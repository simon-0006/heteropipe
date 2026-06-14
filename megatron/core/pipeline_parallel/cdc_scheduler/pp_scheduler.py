import contextlib
import cProfile
import json
import os
import pickle
import pstats
import random
import time
from typing import Dict, Iterator, List, Optional, Tuple, Union
from datetime import timedelta

import numpy as np

from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.pipeline import (
    CPZBUDPipeline,
    CPZBWavePipeline,
    OneChunkPipelineTemplate,
)
from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.subpipeline import DynZBUDSubPipeline
from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.pipeline_config import (
    SystemConfig,
)
from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.auto_schedule import (
    UnidirectionalDynamicBatchSizeZBDependencyGraph,
)
from megatron.core.pipeline_parallel.cdc_scheduler.wgrad_store import WGradStore
from megatron.core.pipeline_parallel.cdc_scheduler.experiment_manager import (
    ExperimentManager,
)
try:
    import pulp
except ImportError:
    pulp = None

import torch
import torch.distributed as dist
import torch.cuda.nvtx as nvtx
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.schedules import (
    check_first_val_step,
    clear_embedding_activation_buffer,
    deallocate_output_tensor,
    finish_embedding_wgrad_compute,
    forward_step,
    backward_step,
    forward_step_subblock,
    backward_step_subblock,
)
from megatron.core.utils import get_model_config, get_model_type
from megatron.training import get_args
from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator import (
    Pipeline,
    get_default_static_schedule,
)
from megatron.core.pipeline_parallel.cdc_scheduler.execution_planner import (
    CommEvent,
    CommEventType,
    ComputeTask,
    ExecutionPlanner,
    TaskEvent,
)
from megatron.core.pipeline_parallel.cdc_scheduler import affine_profiler


_CDC_PP_SCHEDULER = None


def get_cdc_pp_scheduler():
    global _CDC_PP_SCHEDULER
    if _CDC_PP_SCHEDULER is None:
        args = get_args()
        _CDC_PP_SCHEDULER = CDCPPScheduler(args)
    return _CDC_PP_SCHEDULER


def tuple_keys_to_str(d):
    """Recursively converts tuple keys to strings."""
    return {
        str(k): (tuple_keys_to_str(v) if isinstance(v, dict) else v)
        for k, v in d.items()
    }


def str_keys_to_tuple(d):
    """Recursively converts string keys that represent tuples back to tuples."""

    def try_convert_key(k):
        try:
            return eval(k) if k.startswith("(") and k.endswith(")") else k
        except:
            return k

    return {
        try_convert_key(k): (str_keys_to_tuple(v) if isinstance(v, dict) else v)
        for k, v in d.items()
    }


def _floor_small_intercepts(compBias, comp, N, abs_floor_seconds=1e-6):
    """Floor noise-floor intercepts to zero. Without this the integer-scaled
    LP can pick up sub-microsecond regression artefacts and inflate
    coefficient range. Mutates ``compBias`` in place. ``N`` and ``comp`` are
    accepted for symmetry (currently unused — we only floor on absolute
    seconds, not relative size). See issue (i) in the affine-profiler design."""
    for d in range(len(compBias)):
        for op in range(len(compBias[d])):
            if compBias[d][op] < abs_floor_seconds:
                compBias[d][op] = 0.0


def process_pp_stages_per_dc(pp_stages_per_dc, pp_size, num_dc):
    if len(pp_stages_per_dc) == 0:
        # naive split
        ret = [pp_size // num_dc] * num_dc
        for i in range(pp_size % num_dc):
            ret[i] += 1
    elif len(pp_stages_per_dc) == 1:
        ret = [pp_stages_per_dc[0]] * num_dc
    assert (
        sum(ret) == pp_size
    ), f"pp_stages_per_dc {ret} does not sum to pp_size {pp_size}"
    return ret


def _slice_batch(batch, start, end):
    """Slice rows [start, end) out of a base batch. Preserves the pinned
    storage of the source tensor (a view into a pinned tensor is still
    pinned for the purposes of non_blocking H2D transfers)."""
    if isinstance(batch, dict):
        return {k: (v[start:end] if v is not None else None) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(v[start:end] if v is not None else None for v in batch)
    return batch[start:end]


def _concat_batches(pieces):
    """Concatenate aligned slices of the same batch type along dim=0."""
    first = pieces[0]
    if isinstance(first, dict):
        out = {}
        for key in first:
            if first[key] is None:
                out[key] = None
            else:
                out[key] = torch.cat([p[key] for p in pieces], dim=0)
        return out
    if isinstance(first, (list, tuple)):
        out = []
        for i in range(len(first)):
            if first[i] is None:
                out.append(None)
            else:
                out.append(torch.cat([p[i] for p in pieces], dim=0))
        return type(first)(out)
    return torch.cat(pieces, dim=0)


class DynamicMicrobatchIterator:
    """Wraps a data iterator to yield variable-sized microbatches by mb_id.

    The underlying iterator yields batches of size ``micro_batch_size``
    (the equal-split size from args).  This wrapper accumulates enough
    equal-sized batches to fill the global batch, then pre-splits them
    according to ``microbatch_sizes`` into a dict keyed by mb_id.

    Call ``set_next_mb_id(mb_id)`` before each ``next()`` call so that
    the correct chunk is returned regardless of schedule order.

    On pipeline stages that don't consume data (middle stages), the
    underlying iterator yields ``None`` — we just pass those through.

    Chunks are kept at their true variable sizes — no padding. The
    JIT-recv fix in ``schedule_comm_event`` makes that safe at pp >= 4
    (see tmp_sganter/8_4gpu_hang_root_cause.md Update 16).
    """

    def __init__(self, base_iterator, microbatch_sizes: List[int], equal_mb_size: int):
        self.base_iterator = base_iterator
        self.microbatch_sizes = microbatch_sizes
        self.equal_mb_size = equal_mb_size
        self._chunks: Dict[int, Any] = {}  # mb_id -> pre-split chunk
        self._next_mb_id: Optional[int] = None
        self._filled = False
        # Debug fallbacks read once at construction (per-call os.environ.get
        # in __next__ adds ~3 ms / iter at 1600 calls × ~2 µs).
        self._disable_opt1 = os.environ.get("CDC_DISABLE_OPT1", "0") == "1"
        self._disable_opt2 = os.environ.get("CDC_DISABLE_OPT2", "0") == "1"

        # Per-mb chunk specs computed once: each spec is a tuple
        #   (first_base, last_base, base_start_in_first, base_end_in_last,
        #    is_whole_first_base_only)
        # where the consumer reads bases [first_base..last_base], slices the
        # first one at [base_start_in_first:], the last one at [:base_end_in_last],
        # and concatenates the result. When the chunk lives in a single base and
        # spans the whole base, ``is_whole_first_base_only`` lets us skip the
        # slicing entirely (zero-copy passthrough).
        self._chunk_specs = self._build_chunk_specs()
        # Lazy-pull state (Opt 2): we only consume the underlying iterator as
        # the scheduler asks for chunks, spreading the DataLoader's per-batch
        # latency across the iteration instead of bursting all microbatches at
        # once. Bursting was costing ~38 ms/iter in DataLoader-worker lock
        # contention on the [4]*16 benchmark.
        self._base_cache: List[Any] = []      # bases pulled so far (None for non-data stage)
        self._non_data_stage = False

    def _build_chunk_specs(self):
        specs = []
        k = self.equal_mb_size
        cumsum = 0
        for size in self.microbatch_sizes:
            start = cumsum
            end = cumsum + size
            cumsum = end
            first_base = start // k
            last_base = (end - 1) // k
            base_start = start - first_base * k
            base_end = end - last_base * k
            whole_first = first_base == last_base and base_start == 0 and base_end == k
            specs.append((first_base, last_base, base_start, base_end, whole_first))
        return specs

    def _ensure_base_pulled(self, upto_idx):
        """Pull bases from the underlying iterator until index `upto_idx` is
        available in ``self._base_cache``."""
        while len(self._base_cache) <= upto_idx:
            batch = next(self.base_iterator)
            if batch is None:
                self._non_data_stage = True
                # On non-data stages, the iterator yields None forever; cache a
                # single sentinel and refuse to pull more.
                self._base_cache.append(None)
                return
            self._base_cache.append(batch)

    def set_next_mb_id(self, mb_id: int):
        """Set which microbatch id the next ``next()`` call should return."""
        self._next_mb_id = mb_id

    def _refill(self):
        """Pull enough equal-sized batches to cover one global batch, then
        compose per-mb chunks by slicing into the base batches (no upfront
        concat). The earlier "concat all base batches then split" path lost
        the pinned-memory property of the DataLoader output, which forced the
        downstream ``.cuda(non_blocking=True)`` calls in
        ``get_batch_on_this_tp_rank`` to fall back to a synchronous copy
        (~65 ms/iter on the 4×H100 NVL benchmark). Slicing into the original
        base-batch tensors preserves pinning when a chunk lives in a single
        base batch, which is always true for equal-size schedules and most
        chunks of variable-size schedules. Chunks that span multiple base
        batches still need a concat — but that concat is now on much smaller
        slices and only fires for the (usually small number of) cross-base
        chunks."""
        if self._disable_opt1:
            return self._refill_legacy()
        N = sum(self.microbatch_sizes)
        k = self.equal_mb_size
        num_equal_batches = N // k

        batches = []
        for _ in range(num_equal_batches):
            batch = next(self.base_iterator)
            if batch is None:
                # Non-data stage — just yield None for each microbatch
                self._chunks = {i: None for i in range(len(self.microbatch_sizes))}
                self._filled = True
                return
            batches.append(batch)

        self._chunks = {}
        cumsum = 0
        for mb_id, size in enumerate(self.microbatch_sizes):
            start = cumsum
            end = cumsum + size
            cumsum = end
            first_base = start // k
            last_base = (end - 1) // k
            if first_base == last_base:
                base_start = start - first_base * k
                base_end = end - first_base * k
                base_batch = batches[first_base]
                if base_start == 0 and base_end == k:
                    self._chunks[mb_id] = base_batch
                else:
                    self._chunks[mb_id] = _slice_batch(base_batch, base_start, base_end)
            else:
                pieces = []
                for bi in range(first_base, last_base + 1):
                    bs = max(start - bi * k, 0)
                    be = min(end - bi * k, k)
                    pieces.append(_slice_batch(batches[bi], bs, be))
                self._chunks[mb_id] = _concat_batches(pieces)

        self._filled = True

    def _refill_legacy(self):
        """Original concat-then-split refill, kept behind CDC_DISABLE_OPT1=1
        so the new path can be A/B-tested for correctness."""
        N = sum(self.microbatch_sizes)
        num_equal_batches = N // self.equal_mb_size
        batches = []
        for _ in range(num_equal_batches):
            batch = next(self.base_iterator)
            if batch is None:
                self._chunks = {i: None for i in range(len(self.microbatch_sizes))}
                self._filled = True
                return
            batches.append(batch)
        if isinstance(batches[0], dict):
            full_batch = {}
            for key in batches[0]:
                if batches[0][key] is not None:
                    full_batch[key] = torch.cat([b[key] for b in batches], dim=0)
                else:
                    full_batch[key] = None
            self._chunks = {}
            for mb_id, mb_size in enumerate(self.microbatch_sizes):
                chunk = {}
                for key in full_batch:
                    if full_batch[key] is not None:
                        piece, full_batch[key] = (
                            full_batch[key][:mb_size],
                            full_batch[key][mb_size:],
                        )
                        chunk[key] = piece
                    else:
                        chunk[key] = None
                self._chunks[mb_id] = chunk
        elif isinstance(batches[0], (list, tuple)):
            full = [
                torch.cat([b[i] for b in batches], dim=0) if batches[0][i] is not None else None
                for i in range(len(batches[0]))
            ]
            self._chunks = {}
            for mb_id, mb_size in enumerate(self.microbatch_sizes):
                chunk = []
                for i in range(len(full)):
                    if full[i] is not None:
                        piece = full[i][:mb_size]
                        full[i] = full[i][mb_size:]
                        chunk.append(piece)
                    else:
                        chunk.append(None)
                self._chunks[mb_id] = type(batches[0])(chunk)
        else:
            full = torch.cat(batches, dim=0)
            splits = torch.split(full, self.microbatch_sizes, dim=0)
            self._chunks = {i: s for i, s in enumerate(splits)}
        self._filled = True

    def __next__(self):
        if self._disable_opt2:
            # Legacy path: do upfront refill on first __next__.
            if not self._filled:
                self._refill()
            if self._next_mb_id is None:
                raise RuntimeError(
                    "DynamicMicrobatchIterator: next() called without set_next_mb_id(). "
                    "This means data is being consumed outside of schedule_compute_task."
                )
            mb_id = self._next_mb_id
            self._next_mb_id = None
            return self._chunks[mb_id]

        # Opt 2: lazy per-call pull. We only consume base batches up to the
        # last_base required by this microbatch, then materialize the chunk
        # from the cached bases. This spreads DataLoader-worker pressure
        # across the iteration so the prefetch queue can keep up.
        if self._next_mb_id is None:
            raise RuntimeError(
                "DynamicMicrobatchIterator: next() called without set_next_mb_id(). "
                "This means data is being consumed outside of schedule_compute_task."
            )
        mb_id = self._next_mb_id
        self._next_mb_id = None
        first_base, last_base, base_start, base_end, whole_first = self._chunk_specs[mb_id]
        self._ensure_base_pulled(last_base)
        if self._non_data_stage:
            return None
        if whole_first:
            return self._base_cache[first_base]
        if first_base == last_base:
            return _slice_batch(self._base_cache[first_base], base_start, base_end)
        # Multi-base chunk: collect slices and concat.
        pieces = []
        for bi in range(first_base, last_base + 1):
            if bi == first_base:
                bs = base_start
            else:
                bs = 0
            if bi == last_base:
                be = base_end
            else:
                be = self.equal_mb_size
            pieces.append(_slice_batch(self._base_cache[bi], bs, be))
        return _concat_batches(pieces)

    def __iter__(self):
        return self

    def reset_for_next_iteration(self):
        """Reset state for the next training iteration (re-fill from data iterator)."""
        self._chunks = {}
        self._filled = False
        self._base_cache = []
        self._non_data_stage = False


def get_or_set_pp_io_tensor(tensor_dict: Dict, key, config, tensor_shape):
    """Return cached buffer for key, allocating only on miss.

    Optimization 2 (see tmp_sganter/docs/dispatcher_optimizations.md):
    dict.setdefault always evaluates its default expression, so the
    previous implementation allocated a throwaway torch.empty on every
    call even when the cache already had the entry. Switch to an
    explicit miss-check so the allocation only happens when needed.

    Note: pp_scheduler clears entries to None after use (see input_tensors
    cleanup around line 1855). For a None-valued entry, we treat it as a
    miss and re-allocate, matching the original semantics.
    """
    val = tensor_dict.get(key)
    if val is None:
        val = torch.empty(
            tensor_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )
        tensor_dict[key] = val
    return val


class CDCDynamicScheduleGenerator:
    def __init__(
        self,
        args,
        schedule_type: str,
        num_microbatch: int,
        profile_result_path: str = None,
    ) -> None:
        self.args = args
        self.schedule_type = schedule_type
        assert self.schedule_type in [
            "wave",
            "ud",
            "subud",
            "dynamic_mb",
        ], "Currently only support ud, subud, wave, and dynamic_mb schedule"
        self.num_chunks = 2 if self.schedule_type == "wave" else 1
        self.num_microbatch = num_microbatch
        self.profile_result_path = profile_result_path
        self.initialized = False
        self.zero1_dp_modeling = args.zero1_dp_modeling

    def initialize(self):
        # Initialize after profile result is available
        self.initialized = True
        args = self.args
        self.pp_size = args.pipeline_model_parallel_size
        self.latency_delay = 0
        self.bandwidth_delay = 0
        self.injected_latency = np.zeros((self.pp_size, self.pp_size))
        self.injected_bandwidth = np.zeros((self.pp_size, self.pp_size))

        with open(os.path.join(self.profile_result_path, "total.json"), "r") as f:
            profile_result = json.load(f)
        self.T_F_list = np.array(profile_result["T_F"])
        self.T_B_list = np.array(profile_result["T_B"])
        self.T_W_list = np.array(profile_result["T_W"])
        self.T_alpha_matrix = np.array(profile_result["T_alpha"])
        self.T_bw_matrix = np.array(profile_result["T_bw"])
        self.M_F_list = np.array(profile_result["M_F"])
        self.M_B_list = np.array(profile_result["M_B"])
        self.M_W_list = np.array(profile_result["M_W"])
        self.M_Limit_list = np.array(profile_result["M_Limit"])
        self.M_base_list = np.array(profile_result["M_base"])
        self.device_max_mem = float(profile_result["M_dev_max"])
        self.T_DP_list = np.array(profile_result["T_DP"])
        

        assert self.T_F_list.shape == (self.num_chunks, self.pp_size)
        assert self.T_B_list.shape == (self.num_chunks, self.pp_size)
        assert self.T_W_list.shape == (self.num_chunks, self.pp_size)
        assert self.T_alpha_matrix.shape == (self.pp_size, self.pp_size)
        assert self.T_bw_matrix.shape == (self.pp_size, self.pp_size)
        assert self.M_F_list.shape == (self.num_chunks, self.pp_size)
        assert self.M_B_list.shape == (self.num_chunks, self.pp_size)
        assert self.M_W_list.shape == (self.num_chunks, self.pp_size)

        self.pipeline: Pipeline | None = None
        self.microbatch_sizes: Optional[List[int]] = None  # set by dynamic_mb solver

        self.override_M_Limit(args.dynamic_extra_mem_factor)

        if dist.get_rank() == 0:
            self.rank_zero = True
        else:
            self.rank_zero = False
        # self.dump_profile()

    def dump_sys_cfg(self, sys_cfg: SystemConfig) -> None:
        assert self.initialized
        if not self.rank_zero:
            return
        cur_time = time.time()
        with open(
            os.path.join(self.profile_result_path, f"override_cfg_{cur_time}.json"), "w"
        ) as f:
            json.dump(
                {
                    "T_F": tolist_if_needed(sys_cfg.T_F),
                    "T_B": tolist_if_needed(sys_cfg.T_B),
                    "T_W": tolist_if_needed(sys_cfg.T_W),
                    "T_alpha": tolist_if_needed(sys_cfg.T_alpha),
                    "T_beta": tolist_if_needed(sys_cfg.T_beta),
                    "T_DP": tolist_if_needed(sys_cfg.T_DP),
                    "M_F": tolist_if_needed(sys_cfg.M_F),
                    "M_B": tolist_if_needed(sys_cfg.M_B),
                    "M_W": tolist_if_needed(sys_cfg.M_W),
                    "M_Limit": tolist_if_needed(sys_cfg.M_Limit),
                    "num_devices": sys_cfg.num_devices,
                    "num_microbatches": sys_cfg.num_microbatches,
                    "num_chunks": sys_cfg.num_chunks,
                },
                f,
            )

    def override_T_comm(self, latency_seconds=None, bandwidth_seconds=None):
        assert self.initialized
        self.latency_delay = latency_seconds if latency_seconds is not None else 0
        self.bandwidth_delay = bandwidth_seconds if bandwidth_seconds is not None else 0
        self.injected_latency = np.zeros((self.pp_size, self.pp_size))
        self.injected_bandwidth = np.zeros((self.pp_size, self.pp_size))

        pp_stages_per_dc = process_pp_stages_per_dc(
            self.args.pp_stages_per_dc, self.pp_size, self.args.num_dc
        )
        dc_boundaries = [
            sum(pp_stages_per_dc[:i]) for i in range(1, self.args.num_dc + 1)
        ]
        for boundary in dc_boundaries:
            src = (boundary - 1) % self.pp_size
            dst = boundary % self.pp_size
            if latency_seconds is not None:
                self.injected_latency[src, dst] = max(
                    0, latency_seconds - self.T_alpha_matrix[src, dst]
                )
                self.injected_latency[dst, src] = max(
                    0, latency_seconds - self.T_alpha_matrix[dst, src]
                )
            if bandwidth_seconds is not None:
                self.injected_bandwidth[src, dst] = max(
                    0, bandwidth_seconds - self.T_bw_matrix[src, dst]
                )
                self.injected_bandwidth[dst, src] = max(
                    0, bandwidth_seconds - self.T_bw_matrix[dst, src]
                )

    def override_M_Limit(self, extra_mem_factor: float) -> None:
        assert self.initialized
        # M_F if no recompute, (M_F + M_B) if recompute
        for i in range(self.pp_size):
            unit_memory = max([self.M_F_list[chunk][i] for chunk in range(self.num_chunks)] + [self.M_F_list[chunk][i] + self.M_B_list[chunk][i] for chunk in range(self.num_chunks)])
            new_mem_limit = self.pp_size * self.num_chunks * (1 + extra_mem_factor) * unit_memory * 1.02
            self.M_Limit_list[i] = min(new_mem_limit, (self.device_max_mem - self.M_base_list[i]) * 0.96)

    def integerize_sys_cfg(self, sys_cfg: SystemConfig, multiply_factor: int = 1) -> SystemConfig:
        time_candidate = set()
        memory_candidate = set()
        # Assume T_F, T_B, T_W are 2D lists, T_DP is 3D list
        def flatten_to_scalars(x):
            if isinstance(x, np.ndarray):
                # Use .flat for NumPy arrays
                for item in x.flat:
                    yield item
            elif isinstance(x, (list, tuple)):
                for item in x:
                    yield from flatten_to_scalars(item)
            else:
                yield x
        
        for work_list in [
            sys_cfg.T_F,
            sys_cfg.T_B,
            sys_cfg.T_alpha,
            sys_cfg.T_beta,
            sys_cfg.T_W,
            sys_cfg.T_DP,
        ]:
            # add to a set and turn it into a list
            # work_list can be np array or 1d or 2d list
            time_candidate.update(flatten_to_scalars(work_list))
        for work_list in [sys_cfg.M_F, sys_cfg.M_B, sys_cfg.M_W, sys_cfg.M_Limit]:
            memory_candidate.update(flatten_to_scalars(work_list))
        
        time_candidate = list(time_candidate)
        memory_candidate = list(memory_candidate)

        def scale_to_integers_factor(
            candidate_list, target_min_diff=4, max_abs_value=100000
        ):
            values = np.abs(candidate_list)
            sorted_values = np.sort(values)
            # Find minimum non-zero absolute difference between any two values
            diffs = [b - a for a, b in zip(sorted_values, sorted_values[1:])]
            non_zero_diffs = [
                abs(d) for d in diffs if abs(d) > 1e-10
            ]  # Use small epsilon
            min_diff = min(non_zero_diffs) if non_zero_diffs else 1.0
            scaling_reference = min_diff if min_diff > 1e-10 else 1.0
            scaling_factor = target_min_diff / scaling_reference
            if np.max(values) * scaling_factor > max_abs_value:
                scaling_factor = max_abs_value / np.max(values)
            return scaling_factor

        time_scaling_factor = scale_to_integers_factor(time_candidate)
        memory_scaling_factor = scale_to_integers_factor(memory_candidate)

        def scale_list(lst, factor):
            if isinstance(lst, list):
                lst = np.array(lst)
            return ((lst * factor).astype(int) * multiply_factor)

        new_T_F = scale_list(sys_cfg.T_F, time_scaling_factor)
        new_T_B = scale_list(sys_cfg.T_B, time_scaling_factor)
        new_T_alpha = scale_list(sys_cfg.T_alpha, time_scaling_factor)
        new_T_beta = scale_list(sys_cfg.T_beta, time_scaling_factor)
        new_T_W = scale_list(sys_cfg.T_W, time_scaling_factor)
        new_T_DP = scale_list(sys_cfg.T_DP, time_scaling_factor)
        new_M_F = scale_list(sys_cfg.M_F, memory_scaling_factor)
        new_M_B = scale_list(sys_cfg.M_B, memory_scaling_factor)
        new_M_W = np.array([[-new_M_F[i][j] - new_M_B[i][j] for j in range(len(new_M_F[0]))] for i in range(len(new_M_F))], dtype=int)
        new_M_Limit = scale_list(sys_cfg.M_Limit, memory_scaling_factor)
        return SystemConfig(
            T_F=new_T_F,
            T_B=new_T_B,
            T_alpha=new_T_alpha,
            T_beta=new_T_beta,
            T_W=new_T_W,
            M_F=new_M_F,
            M_B=new_M_B,
            M_W=new_M_W,
            M_Limit=new_M_Limit,
            T_DP=new_T_DP,
            num_devices=sys_cfg.num_devices,
            num_microbatches=sys_cfg.num_microbatches,
            num_chunks=sys_cfg.num_chunks,
            zero_1_dp_modeling=self.zero1_dp_modeling,
        ), time_scaling_factor * multiply_factor, memory_scaling_factor * multiply_factor

    def generate_schedule_from_profile(self) -> int:
        assert self.initialized
        ud_solution_time = {4: 1200, 8: 1200, 16: 2400}
        wave_solution_time = {4: 1500, 8: 1500, 16: 2400}
        estimated_runtime = 0
        if self.schedule_type == "wave":
            num_chunks = 2
            sys_cfg = SystemConfig(
                T_F=self.T_F_list,
                T_B=self.T_B_list,
                T_alpha=self.T_alpha_matrix + self.injected_latency,
                T_beta=self.T_bw_matrix + self.injected_bandwidth,
                T_W=self.T_W_list,
                M_F=self.M_F_list,
                M_B=self.M_B_list,
                M_W=self.M_W_list,
                M_Limit=self.M_Limit_list,
                T_DP=self.T_DP_list,
                num_devices=self.pp_size,
                num_microbatches=self.num_microbatch,
                num_chunks=num_chunks,
                zero_1_dp_modeling=self.zero1_dp_modeling,
            )
            sys_cfg, time_factor, mem_factor = self.integerize_sys_cfg(sys_cfg)
            if self.rank_zero:
                self.dump_sys_cfg(sys_cfg)
                if os.path.exists(
                    os.path.join(
                        self.profile_result_path,
                        f"wave_lat{self.latency_delay}_bw{self.bandwidth_delay}.json",
                    )
                ):
                    with open(
                        os.path.join(
                            self.profile_result_path,
                            f"wave_lat{self.latency_delay}_bw{self.bandwidth_delay}.json",
                        ),
                        "r",
                    ) as f:
                        pp = CPZBWavePipeline(sys_cfg)
                        pp.load_schedule_from_dict(json.load(f))
                        pp.solve_dependencies()
                else:
                    pp = CPZBWavePipeline(sys_cfg)
                    pp.schedule(
                        time_limit_sec=wave_solution_time[self.pp_size],
                        relative_gap=0.01,
                        logging=True,
                    )
                    pp.solve_dependencies()
                    pp.print_schedule(
                        name=f"wave_lat{self.latency_delay}_bw{self.bandwidth_delay}",
                        save=True,
                        save_path=self.profile_result_path,
                    )
                    # save schedule as json
                    with open(
                        os.path.join(
                            self.profile_result_path,
                            f"wave_lat{self.latency_delay}_bw{self.bandwidth_delay}.json",
                        ),
                        "w",
                    ) as f:
                        json.dump(pp.store_schedule_to_dict(), f)
            else:
                # wait till the rank 0 save the schedule, sleep 10s for rank 0 to check
                pp: Optional[CPZBWavePipeline] = None
                while pp is None:
                    time.sleep(random.uniform(5, 10))
                    # check if the file exists
                    if os.path.exists(
                        os.path.join(
                            self.profile_result_path,
                            f"wave_lat{self.latency_delay}_bw{self.bandwidth_delay}.json",
                        )
                    ):
                        with open(
                            os.path.join(
                                self.profile_result_path,
                                f"wave_lat{self.latency_delay}_bw{self.bandwidth_delay}.json",
                            ),
                            "r",
                        ) as f:
                            pp = CPZBWavePipeline(sys_cfg)
                            pp.load_schedule_from_dict(json.load(f))
                            pp.solve_dependencies()
            estimated_runtime = pp.get_schedule_time(device_wise=True) / time_factor

        elif self.schedule_type == "ud":
            num_chunks = 1
            sys_cfg = SystemConfig(
                T_F=self.T_F_list,
                T_B=self.T_B_list,
                T_alpha=self.T_alpha_matrix + self.injected_latency,
                T_beta=self.T_bw_matrix + self.injected_bandwidth,
                T_W=self.T_W_list,
                M_F=self.M_F_list,
                M_B=self.M_B_list,
                M_W=self.M_W_list,
                M_Limit=self.M_Limit_list,
                T_DP=self.T_DP_list,
                num_devices=self.pp_size,
                num_microbatches=self.num_microbatch,
                num_chunks=num_chunks,
                zero_1_dp_modeling=self.zero1_dp_modeling,
            )
            sys_cfg, time_factor, mem_factor = self.integerize_sys_cfg(sys_cfg)
            if self.rank_zero:
                self.dump_sys_cfg(sys_cfg)
                if os.path.exists(
                    os.path.join(
                        self.profile_result_path,
                        f"ud_lat{self.latency_delay}_bw{self.bandwidth_delay}.json",
                    )
                ):
                    with open(
                        os.path.join(
                            self.profile_result_path,
                            f"ud_lat{self.latency_delay}_bw{self.bandwidth_delay}.json",
                        ),
                        "r",
                    ) as f:
                        pp = CPZBUDPipeline(sys_cfg)
                        pp.load_schedule_from_dict(json.load(f))
                        pp.solve_dependencies()
                else:
                    pp = CPZBUDPipeline(sys_cfg)
                    pp.schedule(
                        time_limit_sec=ud_solution_time[self.pp_size], relative_gap=0.01, logging=True
                    )
                    pp.solve_dependencies()
                    pp.print_schedule(
                        name=f"ud_lat{self.latency_delay}_bw{self.bandwidth_delay}",
                        save=True,
                        save_path=self.profile_result_path,
                    )
                    # save schedule as json
                    with open(
                        os.path.join(
                            self.profile_result_path,
                            f"ud_lat{self.latency_delay}_bw{self.bandwidth_delay}.json",
                        ),
                        "w",
                    ) as f:
                        json.dump(pp.store_schedule_to_dict(), f)
            else:
                # wait till the rank 0 save the schedule, sleep 10s for rank 0 to check
                pp: Optional[CPZBUDPipeline] = None
                while pp is None:
                    time.sleep(random.uniform(10, 20))
                    # check if the file exists
                    if os.path.exists(
                        os.path.join(
                            self.profile_result_path,
                            f"ud_lat{self.latency_delay}_bw{self.bandwidth_delay}.json",
                        )
                    ):
                        with open(
                            os.path.join(
                                self.profile_result_path,
                                f"ud_lat{self.latency_delay}_bw{self.bandwidth_delay}.json",
                            ),
                            "r",
                        ) as f:
                            pp = CPZBUDPipeline(sys_cfg)
                            pp.load_schedule_from_dict(json.load(f))
                            pp.solve_dependencies()
            estimated_runtime = pp.get_schedule_time(device_wise=True) / time_factor
                            
        elif self.schedule_type == "subud":
            num_chunks = 1
            sys_cfg = SystemConfig(
                T_F=self.T_F_list,
                T_B=self.T_B_list,
                T_alpha=self.T_alpha_matrix + self.injected_latency,
                T_beta=self.T_bw_matrix + self.injected_bandwidth,
                T_W=self.T_W_list,
                M_F=self.M_F_list,
                M_B=self.M_B_list,
                M_W=self.M_W_list,
                M_Limit=self.M_Limit_list,
                T_DP=self.T_DP_list,
                num_devices=self.pp_size,
                num_microbatches=self.num_microbatch,
                num_chunks=num_chunks,
                zero_1_dp_modeling=self.zero1_dp_modeling,
            )
            
            num_subparts = self.args.num_subparts
            sys_cfg, time_factor, mem_factor = self.integerize_sys_cfg(sys_cfg, num_subparts)
            pp = DynZBUDSubPipeline(sys_cfg, num_subparts)
            pp.schedule()
            pp.solve_dependencies()
            
            if self.rank_zero:
                self.dump_sys_cfg(sys_cfg)
                pp.print_schedule(
                    name=f"subud_lat{self.latency_delay}_bw{self.bandwidth_delay}",
                    save=True,
                    save_path=self.profile_result_path,
                )
                # save schedule as json
                with open(
                    os.path.join(
                        self.profile_result_path,
                        f"subud_lat{self.latency_delay}_bw{self.bandwidth_delay}.json",
                    ),
                    "w",
                ) as f:
                    json.dump(pp.store_schedule_to_dict(), f)
            
            estimated_runtime = pp.get_schedule_time(device_wise=True) / time_factor

        elif self.schedule_type == "dynamic_mb":
            num_chunks = 1

            # Derive per-sample compute/comm costs from profiled absolute timings.
            # T_F_list[chunk=0][dev] was measured at microbatch_size = N / num_microbatch.
            N = self.args.global_batch_size // (
                self.args.data_parallel_size if hasattr(self.args, 'data_parallel_size') else 1
            )
            f_profiled = N // self.num_microbatch

            T_alpha_with_inject = self.T_alpha_matrix + self.injected_latency
            T_bw_with_inject = self.T_bw_matrix + self.injected_bandwidth

            schedule_file = os.path.join(self.profile_result_path, "dynamic_mb.json")

            # Rank 0 deletes stale schedule file, then all ranks synchronize.
            # This ensures non-rank-0 workers never read a leftover file from a prior run.
            if self.rank_zero and os.path.exists(schedule_file):
                os.remove(schedule_file)
            dist.barrier()

            if self.rank_zero:
                # --- Only rank 0 solves the MILP ---
                # Dummy T_F/T_B/T_W as Python int lists shaped (num_chunks, num_devices).
                # Actual durations come from comp/compBias/f[mb] in the dynamic solver.
                # M_F + M_B + M_W must sum to 0 per device for the base class assertion.
                dummy_T = [[1] * self.pp_size for _ in range(num_chunks)]
                dummy_M = [[0] * self.pp_size for _ in range(num_chunks)]
                dummy_sys_cfg = SystemConfig(
                    num_devices=self.pp_size,
                    num_microbatches=self.num_microbatch,
                    T_F=dummy_T, T_B=dummy_T, T_W=dummy_T,
                    T_alpha=[[0] * self.pp_size for _ in range(self.pp_size)],
                    M_F=dummy_M,
                    M_B=dummy_M,
                    M_W=dummy_M,
                    M_Limit=[-1] * self.pp_size,
                    num_chunks=num_chunks,
                )

                # Default cost model: single-point scaling
                #   comp[d][op] = T_op[d] / f_profiled       (per-sample slope)
                #   compBias    = 0.0                        (no fixed overhead)
                #   comm[dir]   = avg(T_bw) / f_profiled     (per-sample slope)
                #   commLat     = T_alpha[src][dst]          (per-edge intercept)
                comp = [
                    [
                        float(self.T_F_list[0][d]) / f_profiled,
                        float(self.T_B_list[0][d]) / f_profiled,
                        float(self.T_W_list[0][d]) / f_profiled,
                    ]
                    for d in range(self.pp_size)
                ]
                compBias = [[0.0, 0.0, 0.0] for _ in range(self.pp_size)]

                commLat = T_alpha_with_inject.tolist()
                if self.pp_size > 1:
                    fwd_bw = [float(T_bw_with_inject[d][(d + 1) % self.pp_size]) for d in range(self.pp_size - 1)]
                    bwd_bw = [float(T_bw_with_inject[(d + 1) % self.pp_size][d]) for d in range(self.pp_size - 1)]
                    comm = [np.mean(fwd_bw) / f_profiled, np.mean(bwd_bw) / f_profiled]
                else:
                    comm = [0.0, 0.0]

                # Affine cost model overlay: when --cdc_profile_affine produced
                # fits, replace per-(dev, op) compute slopes + intercepts with
                # the regression results, and average comm slopes from the
                # per-edge fits. First-stage compute (and any cell missing a
                # fit) keeps the single-point fallback above.
                if getattr(self.args, "cdc_profile_affine", False):
                    self._apply_affine_overlays(comp, compBias, comm, commLat, f_profiled=f_profiled)
                    # The affine comm intercepts were measured at profile iter 2, before
                    # stock latency injection begins (cdc_stock_inject_warmup_iters), so
                    # the overlay above erases the injected latency from commLat. Re-apply
                    # it as a floor on the injected edges so the MILP sees the same
                    # boundary latency the runtime will experience.
                    #
                    # EXCEPT: --cdc_dynamic_mb_schedule_lat_ms decouples the latency the
                    # SCHEDULE assumes from the latency the runtime injects. With a fixed
                    # microbatch count (K) the injected cross-boundary latency is paid on a
                    # fixed number of crossings — a near-constant offset the schedule cannot
                    # reduce — yet feeding it into the MILP distorts the compute-balance
                    # objective so the solver picks schedules that are intrinsically slower
                    # at BASE compute and lose to ZBH1 (exp7 / inv7). A value of 0 schedules
                    # compute-only, producing the robust bell schedule (large microbatches
                    # in the pipeline middle, near the slow link) that overlaps the latency
                    # naturally and beats ZBH1 at every injected latency.
                    #   sched_lat_ms <  0 : floor with the real injected latency (legacy)
                    #   sched_lat_ms >= 0 : floor with this assumed latency (0 = compute-only)
                    sched_lat_ms = float(
                        getattr(self.args, "cdc_dynamic_mb_schedule_lat_ms", -1.0)
                    )
                    for _src in range(self.pp_size):
                        for _dst in range(self.pp_size):
                            if self.injected_latency[_src][_dst] > 0:
                                if sched_lat_ms < 0:
                                    floor = float(T_alpha_with_inject[_src][_dst])
                                else:
                                    floor = sched_lat_ms / 1000.0
                                commLat[_src][_dst] = max(
                                    commLat[_src][_dst], floor
                                )

                # Floor very-small intercepts so the integer scaling below
                # doesn't blow up. Anything < 2% of slope*max_in_grid is
                # numerical noise from the regression on a near-perfectly-
                # linear cell — see issue (i) in the design doc.
                _floor_small_intercepts(compBias, comp, N)

                # Integer scaling for the MILP. We use a fixed scale (sec ->
                # microseconds) instead of ``10/min(nonzero)`` so adding new
                # bias terms can't blow up the LP coefficient range.
                scale = 1e6

                comp_scaled = [[v * scale for v in row] for row in comp]
                compBias_scaled = [[v * scale for v in row] for row in compBias]
                comm_scaled = [v * scale for v in comm]
                commLat_scaled = [[v * scale for v in row] for row in commLat]

                print(f"[dynamic_mb] N={N}, f_profiled={f_profiled}, scale={scale:.2f}")
                print(f"[dynamic_mb] comp={comp}, comm={comm}")

                debug_mb_sizes = getattr(self.args, 'cdc_debug_mb_sizes', None)
                if debug_mb_sizes is not None:
                    # Skip MILP — use provided microbatch sizes and build a ZBH1 ordering.
                    mb_sizes = list(debug_mb_sizes)
                    assert len(mb_sizes) == self.num_microbatch, (
                        f"--cdc_debug_mb_sizes has {len(mb_sizes)} entries, "
                        f"expected {self.num_microbatch}"
                    )
                    assert sum(mb_sizes) == N, (
                        f"--cdc_debug_mb_sizes sums to {sum(mb_sizes)}, expected N={N}"
                    )
                    print(f"[dynamic_mb] Using debug microbatch sizes: {mb_sizes}")

                    # Build a ZBH1 schedule to get task ordering.
                    from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.pipeline import (
                        ZBH1Pipeline,
                    )
                    zbh1 = ZBH1Pipeline(dummy_sys_cfg)
                    zbh1.schedule()
                    schedule_dicts = zbh1.store_schedule_to_dict()
                    estimated_runtime_raw = 0
                else:
                    g = UnidirectionalDynamicBatchSizeZBDependencyGraph(
                        system_cfg=dummy_sys_cfg,
                        N=N,
                        comp=comp_scaled,
                        compBias=compBias_scaled,
                        comm=comm_scaled,
                        commLat=commLat_scaled,
                    )
                    g.build_ilp()
                    # Bound the LP search space with a generous cap and a
                    # shape-diversity penalty.
                    #
                    # The cap (max f_i ≤ 2·uniform_size) keeps the MILP
                    # variable space manageable; the runtime cost shaping
                    # is done by the penalty term below.
                    #
                    # The penalty models the unmodeled per-distinct-shape
                    # runtime overhead: cuBLAS kernel-selection lookups
                    # the first time each shape is launched, NCCL stream
                    # multiplexing of variable-size sends, and the Python
                    # dispatcher's per-shape codepaths. Without this term
                    # the LP picked extreme bimodal [1,…,1,BIG] schedules
                    # that look cheap in the cost model but were 5–10 %
                    # slower at runtime. With the penalty, the LP can
                    # still pick a genuinely dynamic schedule when it
                    # saves enough makespan to justify each extra shape,
                    # and falls back to closer-to-uniform schedules when
                    # the comm bottleneck (high injected latency) leaves
                    # nothing for dynamic sizing to exploit.
                    uniform_size = N // self.num_microbatch
                    # Cap at uniform_size + 2 by default: gives the LP room
                    # to play (sizes ∈ {1, ..., 6} at gbs=64, num_mb=16) while
                    # preventing the >1.5×-uniform jumps that hurt runtime.
                    # An explicit non-zero --cdc_dynamic_mb_max_f_cap overrides
                    # the default — set it to N to disable the cap entirely
                    # for investigation runs.
                    arg_cap = int(getattr(self.args,
                                          "cdc_dynamic_mb_max_f_cap", 0) or 0)
                    if arg_cap > 0:
                        max_f_per_mb = arg_cap
                    else:
                        max_f_per_mb = max(2, uniform_size + 2)
                    if uniform_size >= 1:
                        for fv in g.prob_f:
                            fv.upBound = min(int(fv.upBound), max_f_per_mb)

                    if pulp is not None:
                        # Per-distinct-shape penalty in *scaled* (us) units —
                        # ~0.3 ms per extra microbatch shape is enough to
                        # break ties cleanly without preventing the LP from
                        # exploring varied schedules when they pay off.
                        shape_penalty_us = float(
                            getattr(self.args,
                                    "cdc_dynamic_mb_shape_penalty_us",
                                    500.0)
                        )
                        shapes_range = list(range(1, max_f_per_mb + 1))
                        y_s = {
                            s: pulp.LpVariable(
                                f"shape_used_{s}", 0, 1, cat="Binary"
                            )
                            for s in shapes_range
                        }
                        for i in range(self.num_microbatch):
                            z_vars = [
                                pulp.LpVariable(
                                    f"shape_pick_{i}_{s}", 0, 1, cat="Binary"
                                )
                                for s in shapes_range
                            ]
                            # exactly one shape per microbatch
                            g.prob += pulp.lpSum(z_vars) == 1
                            # f_i = sum_s s * z_{i,s}
                            g.prob += g.prob_f[i] == pulp.lpSum(
                                s * z_vars[k] for k, s in enumerate(shapes_range)
                            )
                            for k, s in enumerate(shapes_range):
                                g.prob += y_s[s] >= z_vars[k]
                        g.prob.objective += shape_penalty_us * pulp.lpSum(
                            y_s[s] for s in shapes_range
                        )
                    # 300s: exp6 offline re-solves showed the 600s incumbent is
                    # already found by 300s (bound stalls, incumbent stable).
                    g.solve_ilp(verbose=True, time_limit=300, relative_gap=0.01)

                    mb_sizes = g.get_microbatch_sizes()
                    schedule_blocks = g.get_schedule()

                    if schedule_blocks is None or mb_sizes is None:
                        raise RuntimeError("[dynamic_mb] MILP solver failed to find a solution")

                    chosen_obj = g.get_objective_value()
                    estimated_runtime_raw = chosen_obj / scale if chosen_obj else 0

                    # Safety net: re-solve the LP with f[mb] forced to uniform
                    # N/num_microbatch. If the uniform-mb schedule has lower
                    # predicted makespan in the SAME cost model, the MILP
                    # picked something pathological (typically a side effect
                    # of the time limit on a flat cost surface). Fall back
                    # to uniform to "first do no harm".
                    if (
                        chosen_obj is not None
                        and uniform_size >= 1
                        and uniform_size * self.num_microbatch == N
                    ):
                        g_u = UnidirectionalDynamicBatchSizeZBDependencyGraph(
                            system_cfg=dummy_sys_cfg,
                            N=N,
                            comp=comp_scaled,
                            compBias=compBias_scaled,
                            comm=comm_scaled,
                            commLat=commLat_scaled,
                        )
                        g_u.build_ilp()
                        for fv in g_u.prob_f:
                            fv.lowBound = uniform_size
                            fv.upBound = uniform_size
                        g_u.solve_ilp(verbose=False, time_limit=180,
                                      relative_gap=0.01)
                        uniform_obj = g_u.get_objective_value()
                        u_blocks = g_u.get_schedule()
                        u_sizes = g_u.get_microbatch_sizes()
                        # Safety net with latency-dependent tolerance.
                        # At low / mid injected latency the LP's shape
                        # penalty is enough — only fall back if uniform
                        # is strictly better in the LP. At very high
                        # latency (≥ 50 ms) the comm bottleneck means
                        # any sized variation costs more at runtime than
                        # the LP can see, so use a tolerant comparison
                        # to push toward the ZBH1-ordered uniform
                        # schedule when uniform is even close.
                        inj_lat_ms = float(getattr(
                            self.args, "cdc_stock_inject_latency_ms", 0.0))
                        tol = 0.0  # safety net disabled — always use LP-chosen schedule
                        if (
                            uniform_obj is not None
                            and u_blocks is not None
                            and u_sizes is not None
                            and uniform_obj < chosen_obj * tol
                        ):
                            print(
                                f"[dynamic_mb][safety] uniform schedule wins in LP cost model "
                                f"(uniform={uniform_obj/scale*1e3:.2f}ms < "
                                f"chosen={chosen_obj/scale*1e3:.2f}ms). "
                                f"Falling back to uniform mb sizes + ZBH1 ordering."
                            )
                            # Use uniform mb sizes BUT with the standard
                            # ZBH1 task ordering instead of the LP's own
                            # ordering. At high injected latency the LP's
                            # ordering, even with uniform sizes, can
                            # serialize sends in ways the cost model
                            # doesn't penalise but the NCCL runtime does.
                            # The known-good ZBH1 ordering avoids that.
                            from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.pipeline import (
                                ZBH1Pipeline,
                            )
                            zbh1_fb = ZBH1Pipeline(dummy_sys_cfg)
                            zbh1_fb.schedule()
                            schedule_dicts = zbh1_fb.store_schedule_to_dict()
                            mb_sizes = u_sizes
                            schedule_blocks = None  # signal "use schedule_dicts directly"
                            estimated_runtime_raw = uniform_obj / scale
                        else:
                            uniform_ms = (
                                f"{uniform_obj/scale*1e3:.2f}ms"
                                if uniform_obj is not None
                                else "n/a"
                            )
                            print(
                                f"[dynamic_mb][diag] LP predicts chosen={chosen_obj/scale*1e3:.2f}ms "
                                f"vs uniform={uniform_ms} (safety net disabled; MILP schedule used)"
                            )

                print(f"[dynamic_mb] Solved microbatch sizes: {mb_sizes} (sum={sum(mb_sizes)})", flush=True)

                # Convert to dict format and save (so non-rank-0 can load it).
                # When safety net falls back to ZBH1, schedule_dicts is already
                # populated and schedule_blocks is None; skip the conversion.
                if debug_mb_sizes is None and schedule_blocks is not None:
                    schedule_dicts = [[] for _ in range(self.pp_size)]
                    for dev_blocks in schedule_blocks:
                        for b in dev_blocks:
                            schedule_dicts[b.device_id].append({
                                "task_type": b.task_type,
                                "device_id": b.device_id,
                                "microbatch_id": b.mb_id,
                                "chunk_id": b.chunk_id,
                                "post_send_time": b.post_send_time,
                            })
                with open(schedule_file, "w") as f:
                    json.dump({"schedule": schedule_dicts, "microbatch_sizes": mb_sizes}, f)
            else:
                # --- Non-rank-0: wait for rank 0 to save the schedule ---
                while not os.path.exists(schedule_file):
                    time.sleep(random.uniform(5, 10))

            # --- All ranks: load the schedule and microbatch sizes ---
            with open(schedule_file, "r") as f:
                saved = json.load(f)
            schedule_dicts = saved["schedule"]
            self.microbatch_sizes = saved["microbatch_sizes"]
            # Win 2: keep dyn_loss_scale cache consistent with microbatch_sizes.
            if self.microbatch_sizes is not None:
                _N = sum(self.microbatch_sizes)
                self._dyn_loss_scales = [s / _N for s in self.microbatch_sizes]
            else:
                self._dyn_loss_scales = None

            # Validate the loaded schedule matches the current configuration.
            assert len(schedule_dicts) == self.pp_size, (
                f"[dynamic_mb] Schedule file has {len(schedule_dicts)} devices, "
                f"expected {self.pp_size}"
            )
            assert len(self.microbatch_sizes) == self.num_microbatch, (
                f"[dynamic_mb] Schedule file has {len(self.microbatch_sizes)} microbatches, "
                f"expected {self.num_microbatch}"
            )
            assert sum(self.microbatch_sizes) == N, (
                f"[dynamic_mb] Microbatch sizes sum to {sum(self.microbatch_sizes)}, "
                f"expected N={N}"
            )

            equal_sys_cfg = SystemConfig(
                T_F=self.T_F_list,
                T_B=self.T_B_list,
                T_alpha=T_alpha_with_inject,
                T_beta=T_bw_with_inject,
                T_W=self.T_W_list,
                M_F=self.M_F_list,
                M_B=self.M_B_list,
                M_W=self.M_W_list,
                M_Limit=self.M_Limit_list,
                T_DP=self.T_DP_list,
                num_devices=self.pp_size,
                num_microbatches=self.num_microbatch,
                num_chunks=num_chunks,
                zero_1_dp_modeling=self.zero1_dp_modeling,
            )
            equal_sys_cfg, time_factor, mem_factor = self.integerize_sys_cfg(equal_sys_cfg)
            pp = OneChunkPipelineTemplate(equal_sys_cfg)
            pp.load_schedule_from_dict(schedule_dicts)
            pp.solve_dependencies()

            if self.rank_zero:
                pp.print_schedule(
                    name="dynamic_mb",
                    save=True,
                    save_path=self.profile_result_path,
                )

            estimated_runtime = estimated_runtime_raw if self.rank_zero else 0

        # barrier
        dist.barrier()
        self.pipeline = pp
        return estimated_runtime

    def get_schedule(self) -> Pipeline:
        estimated_runtime = self.generate_schedule_from_profile()
        return self.pipeline, estimated_runtime

    def _apply_affine_overlays(self, comp, compBias, comm, commLat, f_profiled=None):
        """Overlay affine-fit slopes/intercepts onto the single-point cost
        arrays in place. Called only when --cdc_profile_affine was active.

        Compute: for each (chunk, dev) with a fit, replace
            comp[dev][F|B|W]    <- fit.slope
            compBias[dev][F|B|W] <- fit.intercept
        Cells whose fit has ``R² < args.cdc_profile_affine_r2_threshold``
        are **rejected** and keep the single-point fallback already
        populated by the caller. This catches the case where compute is
        launch-overhead-bound at small model sizes and the regression is
        fitting noise rather than signal — applying such a fit would
        give the LP a misleading cost surface.

        Comm: average the per-edge fwd/bwd slopes into ``comm[0/1]``
        (R²-gated per edge), and overwrite ``commLat[src][dst]`` with
        the per-edge intercepts.

        If ``f_profiled`` is provided we also run a sanity check: at the
        profile-iter microbatch size, the affine prediction should be
        close to the canonical ``T_op[d]`` measurement. Disagreements
        above ``args.cdc_profile_affine_sanity_tol`` trigger a warning
        log line but do not change behaviour — the R² threshold is the
        actual gate."""
        r2_threshold = float(getattr(self.args, "cdc_profile_affine_r2_threshold", 0.7))
        sanity_tol = float(getattr(self.args, "cdc_profile_affine_sanity_tol", 0.20))

        compute_fits = affine_profiler.load_affine_compute(
            self.profile_result_path, self.pp_size
        )
        comm_fits = affine_profiler.load_affine_comm(self.profile_result_path)

        op_idx = {"F": 0, "B": 1, "W": 2}
        applied = 0
        rejected_lowr2 = 0
        sanity_warnings = 0
        T_op_lists = {"F": self.T_F_list, "B": self.T_B_list, "W": self.T_W_list}
        if compute_fits is not None:
            for (chunk_id, dev), per_op in compute_fits.items():
                if chunk_id != 0:
                    # dynamic_mb only solves with num_chunks=1.
                    continue
                for op in ("F", "B", "W"):
                    cell = per_op.get(op) if isinstance(per_op, dict) else None
                    if not cell:
                        continue
                    fit = cell.get("fit") or {}
                    slope = float(fit.get("slope", 0.0))
                    intercept = float(fit.get("intercept", 0.0))
                    r2 = fit.get("r2")
                    n_points = int(fit.get("n_points", 0))
                    # Reject low-confidence fits: a flat curve produces tiny
                    # slope + large intercept that the LP would interpret as
                    # "this task has a big fixed cost per dispatch", which
                    # skews schedules toward concentration. Better to keep
                    # the canonical single-point estimate in those cases.
                    if n_points >= 2 and r2 is not None and float(r2) < r2_threshold:
                        rejected_lowr2 += 1
                        # At low R² the compute is launch-overhead-bound and
                        # the slope is meaningless — BUT the intercept is
                        # essentially the floor measurement, which is real.
                        # If we leave compBias=0 and keep the single-point
                        # slope (T_op/f_profiled), the LP sees "compute is
                        # purely linear from zero", which dramatically
                        # underestimates the cost of small microbatches and
                        # makes the MILP pick tiny mbs that are actually
                        # expensive at runtime. Fix: keep the intercept as
                        # compBias and derive a slope so the model passes
                        # through the canonical T_op at f_profiled.
                        used_intercept = False
                        if intercept > 0.0 and f_profiled is not None and f_profiled > 0:
                            try:
                                canonical_T = float(T_op_lists[op][0][dev])
                                if canonical_T > intercept:
                                    derived_slope = (canonical_T - intercept) / f_profiled
                                    comp[dev][op_idx[op]] = derived_slope
                                    compBias[dev][op_idx[op]] = intercept
                                    used_intercept = True
                            except (IndexError, ValueError, TypeError):
                                pass
                        print(
                            f"[dynamic_mb][affine] reject fit dev={dev} op={op}: "
                            f"R²={float(r2):.3f} < threshold={r2_threshold:.2f} "
                            f"(used affine intercept={intercept*1e3:.2f}ms as fixed cost)"
                            if used_intercept
                            else f"[dynamic_mb][affine] reject fit dev={dev} op={op}: "
                                 f"R²={float(r2):.3f} < threshold={r2_threshold:.2f} "
                                 f"(keeping single-point fallback)"
                        )
                        continue
                    if slope > 0.0:
                        comp[dev][op_idx[op]] = slope
                    if intercept > 0.0:
                        compBias[dev][op_idx[op]] = intercept
                    applied += 1

                    # Sanity check: the affine prediction at the profile-iter
                    # mbs should be close to the canonical T_op measurement.
                    if f_profiled is not None and f_profiled > 0:
                        try:
                            predicted = slope * f_profiled + intercept
                            canonical = float(T_op_lists[op][0][dev])
                            if canonical > 0:
                                rel_err = abs(predicted - canonical) / canonical
                                if rel_err > sanity_tol:
                                    sanity_warnings += 1
                                    print(
                                        f"[dynamic_mb][affine] sanity check dev={dev} op={op}: "
                                        f"affine predicts {predicted*1e6:.0f}us at mbs={f_profiled}, "
                                        f"canonical T_op was {canonical*1e6:.0f}us "
                                        f"(rel_err={rel_err*100:.1f}% > tol={sanity_tol*100:.0f}%)"
                                    )
                        except (IndexError, ValueError, TypeError):
                            pass
            print(
                f"[dynamic_mb][affine] compute fits: applied={applied}, "
                f"rejected_lowr2={rejected_lowr2}, sanity_warnings={sanity_warnings} "
                f"(threshold R²={r2_threshold:.2f})"
            )
        else:
            print(
                "[dynamic_mb][affine] flag set but no affine_profile_rank*.json found; "
                "compute keeps single-point scaling"
            )

        if comm_fits is not None and self.pp_size > 1:
            fwd_slopes: List[float] = []
            bwd_slopes: List[float] = []
            comm_rejected = 0
            for edge_str, payload in comm_fits.get("fwd_per_edge", {}).items():
                src = int(edge_str)
                dst = src + 1
                fit = payload.get("fit") or {}
                r2 = fit.get("r2")
                n_points = int(fit.get("n_points", 0))
                if n_points >= 2 and r2 is not None and float(r2) < r2_threshold:
                    comm_rejected += 1
                    print(
                        f"[dynamic_mb][affine] reject comm fwd edge {src}->{dst}: "
                        f"R²={float(r2):.3f} < threshold={r2_threshold:.2f}"
                    )
                    continue
                if fit.get("slope") is not None:
                    fwd_slopes.append(float(fit["slope"]))
                if fit.get("intercept") is not None and float(fit["intercept"]) > 0:
                    commLat[src][dst] = float(fit["intercept"])
            for edge_str, payload in comm_fits.get("bwd_per_edge", {}).items():
                src = int(edge_str)
                dst = src + 1
                fit = payload.get("fit") or {}
                r2 = fit.get("r2")
                n_points = int(fit.get("n_points", 0))
                if n_points >= 2 and r2 is not None and float(r2) < r2_threshold:
                    comm_rejected += 1
                    print(
                        f"[dynamic_mb][affine] reject comm bwd edge {dst}->{src}: "
                        f"R²={float(r2):.3f} < threshold={r2_threshold:.2f}"
                    )
                    continue
                if fit.get("slope") is not None:
                    bwd_slopes.append(float(fit["slope"]))
                if fit.get("intercept") is not None and float(fit["intercept"]) > 0:
                    commLat[dst][src] = float(fit["intercept"])
            if fwd_slopes:
                comm[0] = float(np.mean(fwd_slopes))
            if bwd_slopes:
                comm[1] = float(np.mean(bwd_slopes))
            print(
                f"[dynamic_mb][affine] applied comm fits for "
                f"{len(fwd_slopes)} fwd / {len(bwd_slopes)} bwd edges "
                f"(rejected_lowr2={comm_rejected})"
            )
        else:
            print(
                "[dynamic_mb][affine] flag set but no affine_profile_comm.json found; "
                "comm keeps single-point scaling"
            )

    def update_latency_bandwidth_seconds(
        self, latency_seconds=None, bandwidth_seconds=None
    ):
        self.override_T_comm(latency_seconds, bandwidth_seconds)

    def apply_stock_inject_latency(self, latency_seconds: float, link_mode: str) -> None:
        """Add the artificial latency injected by `--cdc_stock_inject_latency_ms`
        into ``self.injected_latency`` so the MILP sees it as part of T_alpha.

        Mirrors the boundary logic used at runtime by
        ``CDCPPScheduler._isend_with_optional_stock_spin`` so the LP cost
        model agrees with what actually happens during training.

        Unlike ``override_T_comm`` (which uses ``self.args.num_dc`` and
        therefore requires --num_dc>1 to inject anything), this works
        without --num_dc / --pp_stages_per_dc by defaulting to a 2-DC
        split in half. The mode mirrors the runtime flag:
          - "all": every adjacent edge gets the latency
          - "cross_boundary": only the link crossing the (synthetic or
            user-set) DC boundary
          - "none": no injection
        """
        if latency_seconds <= 0 or link_mode == "none":
            return
        if not self.initialized:
            return
        # Boundary identification — same logic as the runtime injection.
        if link_mode == "all":
            boundary_edges = [(i, i + 1) for i in range(self.pp_size - 1)]
        elif link_mode == "cross_boundary":
            if self.pp_size <= 1:
                return
            n_dc = max(1, int(getattr(self.args, "num_dc", 1)))
            stages_per_dc = process_pp_stages_per_dc(
                getattr(self.args, "pp_stages_per_dc", []),
                self.pp_size,
                n_dc if n_dc > 1 else 2,
            )
            boundaries = [sum(stages_per_dc[:i]) for i in range(1, len(stages_per_dc) + 1)]
            boundary_edges = []
            for b in boundaries:
                if 1 <= b < self.pp_size:  # skip wrap-around
                    boundary_edges.append((b - 1, b))
        else:
            return
        for src, dst in boundary_edges:
            # `injected_latency` is added to `T_alpha_matrix` later; here we
            # ensure it reflects the runtime delay. Use max(0, ...) in case
            # the profiled NVLink time is already higher (rare).
            self.injected_latency[src, dst] = max(
                self.injected_latency[src, dst],
                latency_seconds - self.T_alpha_matrix[src, dst],
                0.0,
            )
            self.injected_latency[dst, src] = max(
                self.injected_latency[dst, src],
                latency_seconds - self.T_alpha_matrix[dst, src],
                0.0,
            )


class CDCPPScheduler:
    """

    Notice that chunk_id == virtual_pipeline_model_parallel_rank
    TODO: support checkpointing

    """

    def __init__(self, args) -> None:
        self.args = args
        self.config = None
        self.use_static_schedule = False
        self.use_dynamic_schedule = False
        self.pp_schedule: Pipeline = None

        self.num_subparts = args.num_subparts
        if self.num_subparts > 1:
            assert args.dynamic_schedule == "subud"
            self.subblock_scheduling = True
        else:
            self.subblock_scheduling = False

        self.cdc_verbose_print = args.cdc_verbose_print
        self.cdc_print_rank = args.cdc_print_rank

        pp_size = args.pipeline_model_parallel_size
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.pp_rank = pp_rank

        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.dp_rank = parallel_state.get_data_parallel_rank()

        num_microbatch = get_num_microbatches()
        self.num_microbatch = num_microbatch
        
        exp_dir = args.tensorboard_dir

        # static schedule
        if args.static_schedule is not None:
            assert args.dynamic_schedule is None
            self.use_static_schedule = True
            self.pp_schedule = get_default_static_schedule(
                args.static_schedule, pp_size, num_microbatch
            )
            self.pp_schedule_generator = None
        else:
            assert args.dynamic_schedule is not None
            self.use_dynamic_schedule = True
            self.dynamic_schedule_type = args.dynamic_schedule
            self.pp_schedule_generator = CDCDynamicScheduleGenerator(
                args,
                self.dynamic_schedule_type,
                num_microbatch,
                exp_dir,
            )
            # start with profile schedule
            self.pp_schedule = get_default_static_schedule(
                "ZBH1" if self.dynamic_schedule_type in ["ud", "subud", "dynamic_mb"] else "ZBV",
                pp_size,
                num_microbatch,
            )
            if self.subblock_scheduling:
                for dev_task_list in self.pp_schedule.device_scheduled_tasks:
                    for dev_task in dev_task_list:
                        dev_task.subpart_start = 0
                        dev_task.subpart_end = self.num_subparts
                        dev_task.num_subparts = self.num_subparts

        self.exp_manager = ExperimentManager(
            args,
            self.tp_rank,
            self.dp_rank,
            self.pp_rank,
            self.pp_schedule.sys_config.num_chunks,
            exp_dir=exp_dir,
        )
        self.cdc_print(self.exp_manager.print_expertiment_info(), rank=0)
        
        if self.tp_rank == 0 and self.dp_rank == 0 and self.pp_rank == 0:
            self.pp_schedule.print_schedule(name='schedule_init', save=True, save_path=self.exp_manager.profile_result_path)

        self.pp_execution_planner = ExecutionPlanner(self.pp_schedule)
        self.pp_execution_planner.generate_execution_plan()
        self.pp_execution_plan: List[List[ComputeTask]] = (
            self.pp_execution_planner.execution_plan
        )

        self.pp_execution_plan_cur_device: List[ComputeTask] = self.pp_execution_plan[
            pp_rank
        ]

        self.cdc_print(
            f"execution_plan: \n {self.pp_execution_planner.print_execution_plan()}",
            rank=0,
            verbose=2,
        )

        self.cdc_print(f"layer distribution: {self.get_num_layers_in_chunk()}")

        # cross-DC
        self.num_dc = args.num_dc
        # tuple of (ratio to F stage time, time in seconds)
        self.injected_latency_delay = (0, 0)
        self.injected_bandwidth_delay = (0, 0)

        # decide whether to insert latency.
        self.cdc_recv_prev = False
        self.cdc_recv_next = False
        self.cdc_send_prev = False
        self.cdc_send_next = False

        if self.num_dc > 1:
            self.pp_stages_per_dc = process_pp_stages_per_dc(
                args.pp_stages_per_dc, pp_size, self.num_dc
            )

            # check if ocurrent rank on the boundary of DCs
            dc_boundaries = [
                sum(self.pp_stages_per_dc[:i]) for i in range(1, self.num_dc + 1)
            ]
            if pp_rank + 1 in dc_boundaries:
                # check if any recv next events in the plan
                for task in self.pp_execution_plan_cur_device:
                    for event in task.pre_events + task.post_events:
                        if (
                            isinstance(event, CommEvent)
                            and event.type == CommEventType.POST_RECV_NEXT
                        ):
                            self.cdc_recv_next = True
                        if (
                            isinstance(event, CommEvent)
                            and event.type == CommEventType.POST_SEND_NEXT
                        ):
                            self.cdc_send_next = True
                        if self.cdc_recv_next and self.cdc_send_next:
                            break
            if pp_rank in [x % pp_size for x in dc_boundaries]:
                # check if any recv prev events in the plan
                for task in self.pp_execution_plan_cur_device:
                    for event in task.pre_events + task.post_events:
                        if (
                            isinstance(event, CommEvent)
                            and event.type == CommEventType.POST_RECV_PREV
                        ):
                            self.cdc_recv_prev = True
                        if (
                            isinstance(event, CommEvent)
                            and event.type == CommEventType.POST_SEND_PREV
                        ):
                            self.cdc_send_prev = True
                        if self.cdc_recv_prev and self.cdc_send_prev:
                            break
            self.cdc_print(
                f"delay injection: recv_prev {self.cdc_recv_prev}, recv_next {self.cdc_recv_next}, send_prev {self.cdc_send_prev}, send_next {self.cdc_send_next}",
            )

        self.exp_manager.cdc_comm_profiles = self.pp_benchmark()

        # One-shot guard for the affine profiler. The actual sweep runs from
        # forward_backward_func once the model + iterator are live and right
        # before the dynamic_mb LP solves, so that the LP can pick up the fits.
        self._affine_profile_done = False

        self.wgrad_split = any(
            [task.task_desc.type == "W" for task in self.pp_execution_plan_cur_device]
        )

        self.wgrad_store = WGradStore() if self.wgrad_split else None

        # p2p handles, (mb, chunk, type)
        self.send_next_reqs: Dict[Tuple, dist.Work] = {}
        self.recv_next_reqs: Dict[Tuple, dist.Work] = {}
        self.send_prev_reqs: Dict[Tuple, dist.Work] = {}
        self.recv_prev_reqs: Dict[Tuple, dist.Work] = {}

        # cached tensors (mb, chunk)
        self.input_tensors: Dict[Tuple, torch.Tensor] = {}
        self.output_tensors: Dict[Tuple, torch.Tensor] = {}
        self.output_tensor_grads: Dict[Tuple, torch.Tensor] = {}
        self.input_tensor_grads: Dict[Tuple, torch.Tensor] = {}

        # token count
        self.total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

        # dynamic microbatch sizes (None = all equal, set after solver runs)
        self.microbatch_sizes: Optional[List[int]] = None
        # Win 2 cache (recomputed when microbatch_sizes is assigned).
        self._dyn_loss_scales: Optional[List[float]] = None
        # Win 4 cache (per-mb tensor_shape; recomputed once per call to
        # forward_backward_func when microbatch_sizes is set).
        self._cached_tensor_shapes: Optional[List[List[int]]] = None
        # Win 3: cached "data iterator was wrapped this iter" flag, set in
        # forward_backward_func once per iter so per-task hot loop doesn't
        # need a per-task isinstance() check.
        self._data_iter_is_dynamic: bool = False

        # NOTE: heteropipe used to create dedicated 2-rank P2P ProcessGroups
        # here when use_dynamic_schedule and pp_size > 2, as a workaround for
        # the 4-GPU dangling-recv NCCL hang (see tmp_sganter/docs/root_cause.md).
        # That workaround was removed once JIT-recv fixed the actual root cause:
        # dynamic_mb now uses the same `parallel_state.get_pipeline_extra_*_group()`
        # NCCL communicators as the static (ZBH1, 1F1B, ...) schedules.

        # Optimization 1 (see tmp_sganter/docs/dispatcher_optimizations.md):
        # cache parallel_state lookups that are constant after init.
        # schedule_comm_event used to call these on every event (~300 events/iter
        # = ~1800 cross-module function calls per iteration that all return
        # constants). Now they're computed once.
        self._cached_next_rank = parallel_state.get_pipeline_model_parallel_next_rank()
        self._cached_prev_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
        self._cached_extra_send_next_group = parallel_state.get_pipeline_extra_send_next_group()
        self._cached_extra_recv_next_group = parallel_state.get_pipeline_extra_recv_next_group()
        self._cached_extra_send_prev_group = parallel_state.get_pipeline_extra_send_prev_group()
        self._cached_extra_recv_prev_group = parallel_state.get_pipeline_extra_recv_prev_group()

        # Stock-PyTorch artificial latency injection (sender-side `torch.cuda._sleep`).
        # See tmp_sganter/docs/comm_delay_investigation.md for rationale.
        # Why sender-side: putting the spin on the receiver's WAIT path adds
        # latency_ms on TOP of any prior compute, which overcounts and kills
        # the overlap the MILP modeled. On the sender's send stream, the spin
        # runs concurrently with default-stream compute, and the receiver sees
        # the data at the actual link arrival time (max(0, lat - elapsed)).
        self._stock_inject_lat_ms = float(getattr(args, "cdc_stock_inject_latency_ms", 0.0))
        self._stock_inject_warmup_iters = int(getattr(args, "cdc_stock_inject_warmup_iters", 10))
        self._stock_inject_link_mode = str(getattr(args, "cdc_stock_inject_link", "cross_boundary"))
        self._stock_inject_enabled = (
            self._stock_inject_lat_ms > 0.0 and self._stock_inject_link_mode != "none"
        )
        # Per-direction dedicated CUDA streams for the spin+isend kernels.
        # Lazily created so non-CUDA paths don't pay for them.
        self._stock_inject_send_next_stream: Optional[torch.cuda.Stream] = None
        self._stock_inject_send_prev_stream: Optional[torch.cuda.Stream] = None
        # cycles_per_ms calibrated once on first use. Hardcoded to a calibrated
        # H100 NVL value; auto-calibrates on other GPUs in case anyone moves this.
        self._stock_inject_cycles_per_ms: float = 1_784_909.0
        self._stock_inject_calibrated: bool = False
        # Identify which sends should be delayed.
        # cross_boundary: only the link to the rank in a different DC.
        # all: every outgoing send.
        # none: nothing (already gated above).
        self._stock_inject_delay_next = False
        self._stock_inject_delay_prev = False
        if self._stock_inject_enabled:
            if self._stock_inject_link_mode == "all":
                self._stock_inject_delay_next = True
                self._stock_inject_delay_prev = True
            elif self._stock_inject_link_mode == "cross_boundary":
                # Reuse the same DC-boundary logic the existing flag uses
                # (computed below in the num_dc > 1 block). We re-compute it
                # here independently so the new flag works without --num_dc.
                # If pp_size == 1, there are no inter-rank links anyway.
                # NOTE: this runs inside CDCPPScheduler.__init__, where
                # `self.pp_size` is NOT set yet (only `self.pp_rank` is set
                # at line 1289). Use the local `pp_size` from the enclosing
                # scope (derived from args.pipeline_model_parallel_size).
                if pp_size > 1:
                    # Default: place the slow link between the last "half" of
                    # the pipeline and the rest (mirrors --pp_stages_per_dc with
                    # 2 DCs split in half). User overrides via --num_dc and
                    # --pp_stages_per_dc if they want a different boundary.
                    n_dc = max(1, int(getattr(args, "num_dc", 1)))
                    stages_per_dc = process_pp_stages_per_dc(
                        getattr(args, "pp_stages_per_dc", []),
                        pp_size,
                        n_dc if n_dc > 1 else 2,  # default to 2-DC split if user didn't set
                    )
                    boundaries = [sum(stages_per_dc[:i]) for i in range(1, len(stages_per_dc) + 1)]
                    # pp_rank's send-to-next is at the boundary if (pp_rank + 1) is a boundary.
                    if (pp_rank + 1) in boundaries:
                        self._stock_inject_delay_next = True
                    # pp_rank's send-to-prev is at the boundary if pp_rank is a boundary
                    # (i.e. it's the first rank in its DC, sending back to the prev DC).
                    if pp_rank in [b % pp_size for b in boundaries]:
                        self._stock_inject_delay_prev = True
        # Iteration counter for warmup skipping. Read from the experiment manager
        # at injection time so we honor the current training iteration.
        self.cdc_print(
            f"stock latency injection: enabled={self._stock_inject_enabled} "
            f"lat_ms={self._stock_inject_lat_ms} link_mode={self._stock_inject_link_mode} "
            f"delay_next={self._stock_inject_delay_next} delay_prev={self._stock_inject_delay_prev} "
            f"warmup_iters={self._stock_inject_warmup_iters}",
        )

        # grad sync
        self.no_sync_func = None
        self.no_sync_context = None

        # Profiling instrumentation (env-var gated):
        #   CDC_BYPASS_DYNAMIC_ITERATOR=1 — skip DynamicMicrobatchIterator wrap
        #     and per-task / per-event dynamic-mode branches. Only meaningful
        #     when microbatch_sizes happens to match micro_batch_size for every
        #     entry (e.g. --cdc_debug_mb_sizes 4 4 ...). Loss values will match
        #     the static path under that condition; under variable mbs they
        #     will not, so do not use this for real training.
        #   CDC_PROFILE_DISPATCHER=1 — wrap the forward_backward_func body in
        #     cProfile. Stats are collected between iter
        #     CDC_PROFILE_WARMUP (default 10) and CDC_PROFILE_LAST (default 110)
        #     and dumped to CDC_PROFILE_OUT (default
        #     /tmp/cdc_profile_rank<R>.prof) on the last enabled iter.
        self._bypass_dynamic_iterator = bool(int(os.environ.get("CDC_BYPASS_DYNAMIC_ITERATOR", "0")))
        self._profile_dispatcher = bool(int(os.environ.get("CDC_PROFILE_DISPATCHER", "0")))
        self._profile_warmup = int(os.environ.get("CDC_PROFILE_WARMUP", "10"))
        self._profile_last = int(os.environ.get("CDC_PROFILE_LAST", "110"))
        self._profile_out = os.environ.get(
            "CDC_PROFILE_OUT", f"/tmp/cdc_profile_rank{self.pp_rank}.prof"
        )
        self._profile_obj: Optional[cProfile.Profile] = None
        self._profile_iter_counter = 0
        self._profile_dumped = False

        self.validate_args()

        if self.exp_manager.dump_execution_plan_on_this_rank():
            with open(
                os.path.join(
                    self.exp_manager.profile_result_path, "exec_plan_init.log"
                ),
                "w",
            ) as f:
                f.write(self.pp_execution_planner.print_execution_plan())

    def update_schedule_with_latency_bandwidth(self):
        if self.exp_manager.profile_result is None:
            return
        # For latency injection experiments, update injected delays
        if hasattr(self.exp_manager, 'cdc_exp_override_iter_map'):
            latency_sec, bandwidth_sec = (
                self.exp_manager.get_injected_latency_bandwidth_delay_seconds()
            )
            latency_as_F_stage, bandwidth_as_F_stage = (
                self.exp_manager.get_injected_latency_bandwidth_delay_as_F_stage()
            )
            if not self.exp_manager.need_schedule_update_in_current_iter():
                return
            self.injected_latency_delay = (latency_as_F_stage, latency_sec)
            self.injected_bandwidth_delay = (bandwidth_as_F_stage, bandwidth_sec)
        elif not self.exp_manager.profiling_done() or hasattr(self, '_dynamic_schedule_generated'):
            return
        elif self.use_static_schedule:
            # Static schedule with no experiment — no re-generation needed.
            return
        else:
            # dynamic_mb or other non-experiment modes: generate schedule once after profiling
            self._dynamic_schedule_generated = True
            self.injected_latency_delay = (0, 0.0)
            self.injected_bandwidth_delay = (0, 0.0)
        self.cdc_print(
            f"Delay Config Update: latency {self.injected_latency_delay[0]} F stage, {self.injected_latency_delay[1]} seconds; bandwidth {self.injected_bandwidth_delay[0]} F stage, {self.injected_bandwidth_delay[1]} seconds",
            rank=0,
        )
        
        estimated_runtime = 0

        if self.use_static_schedule:
            with open(os.path.join(self.exp_manager.profile_result_path, "total.json"), "r") as f:
                profile_result = json.load(f)
            T_F_list = np.array(profile_result["T_F"])
            T_B_list = np.array(profile_result["T_B"])
            T_W_list = np.array(profile_result["T_W"])
            T_DP_list = np.array(profile_result["T_DP"])
            M_F_list = np.array(profile_result["M_F"])
            M_B_list = np.array(profile_result["M_B"])
            M_W_list = np.array(profile_result["M_W"])
            M_Limit_list = np.array(profile_result["M_Limit"])
            pp_size = self.args.pipeline_model_parallel_size
            new_latency_matrix = np.zeros((pp_size, pp_size))
            new_bandwidth_matrix = np.zeros((pp_size, pp_size))
            
            pp_stages_per_dc = process_pp_stages_per_dc(
                self.args.pp_stages_per_dc, pp_size, self.args.num_dc
            )
            dc_boundaries = [
                sum(pp_stages_per_dc[:i]) for i in range(1, self.args.num_dc + 1)
            ]
            for boundary in dc_boundaries:
                src = (boundary - 1) % pp_size
                dst = boundary % pp_size
                new_latency_matrix[src, dst] = latency_sec
                new_latency_matrix[dst, src] = latency_sec
                new_bandwidth_matrix[src, dst] = bandwidth_sec
                new_bandwidth_matrix[dst, src] = bandwidth_sec
            sys_cfg = SystemConfig(
                T_F=T_F_list,
                T_B=T_B_list,
                T_alpha=new_latency_matrix,
                T_beta=new_bandwidth_matrix,
                T_W=T_W_list,
                M_F=M_F_list,
                M_B=M_B_list,
                M_W=M_W_list,
                M_Limit=M_Limit_list,
                T_DP=T_DP_list,
                num_devices=pp_size,
                num_microbatches=self.num_microbatch,
                num_chunks=self.pp_schedule.sys_config.num_chunks,
                two_dc=False,                
            )
            pipeline = get_default_static_schedule(
                self.args.static_schedule, pp_size, self.num_microbatch, not_to_solve_deps=True
            )
            pipeline.sys_config = sys_cfg
            pipeline.solve_dependencies()
            estimated_runtime = pipeline.get_schedule_time(device_wise=True)          
            self.pp_schedule = pipeline 

        else:
            # dynamic schedule
            self.pp_schedule_generator.update_latency_bandwidth_seconds(
                self.injected_latency_delay[1], self.injected_bandwidth_delay[1]
            )
            # Plumb the stock-PyTorch artificial latency into the LP cost
            # model. Without this the MILP profiles at iter 2 (fast NVLink),
            # picks a schedule with many small microbatches that don't
            # amortize the per-send latency, and loses to ZBH1 at runtime.
            # See tmp_sganter/logs/injected_latency/RESULTS_pp4.md.
            if self._stock_inject_enabled:
                self.pp_schedule_generator.apply_stock_inject_latency(
                    latency_seconds=self._stock_inject_lat_ms / 1000.0,
                    link_mode=self._stock_inject_link_mode,
                )
                self.cdc_print(
                    f"stock latency injection: LP T_alpha now includes "
                    f"{self._stock_inject_lat_ms}ms on "
                    f"{self._stock_inject_link_mode} edges",
                    rank=0,
                )
            self.pp_schedule, estimated_runtime = self.pp_schedule_generator.get_schedule()
            # Propagate dynamic microbatch sizes if available
            if self.pp_schedule_generator.microbatch_sizes is not None:
                self.microbatch_sizes = self.pp_schedule_generator.microbatch_sizes
                # Win 2: cache dyn_loss_scale array (one division per mb instead
                # of recomputing sum(microbatch_sizes) / per-mb division each task).
                _N = sum(self.microbatch_sizes)
                self._dyn_loss_scales = [s / _N for s in self.microbatch_sizes]
            else:
                self._dyn_loss_scales = None

        if (self.use_static_schedule and self.args.enable_prefetch_opt) or self.use_dynamic_schedule:
            self.pp_execution_planner = ExecutionPlanner(self.pp_schedule)
            self.pp_execution_planner.generate_execution_plan()
            self.pp_execution_plan: List[List[ComputeTask]] = (
                self.pp_execution_planner.execution_plan
            )

            self.pp_execution_plan_cur_device: List[ComputeTask] = (
                self.pp_execution_plan[self.pp_rank]
            )
            
            # self.pp_schedule.print_schedule(name=f'schedule_lat{latency_as_F_stage}_bw{bandwidth_as_F_stage}', save=True, save_path=self.exp_manager.profile_result_path)

            self.cdc_print(
                f"updated execution_plan: \n {self.pp_execution_planner.print_execution_plan()}",
                rank=0,
                verbose=2,
            )
            if self.exp_manager.dump_execution_plan_on_this_rank():
                with open(
                    os.path.join(
                        self.exp_manager.profile_result_path,
                        f"exec_plan_lat{latency_as_F_stage}_bw{bandwidth_as_F_stage}.log",
                    ),
                    "w",
                ) as f:
                    f.write(self.pp_execution_planner.print_execution_plan())

        dist.barrier()

        # Note: heteropipe used to do a parity-based 4-phase eager P2P warmup
        # here to force NCCL to materialize the dedicated 2-rank communicators
        # before the first real iteration. With the dedicated groups removed
        # (JIT-recv made them unnecessary, see tmp_sganter/docs/root_cause.md),
        # the schedule reuses the same `parallel_state.get_pipeline_extra_*_group()`
        # NCCL communicators that ZBH1/1F1B already use, which are initialized
        # by Megatron's own setup. No additional warmup needed.

        self.exp_manager.exp_logging_perf_model_iter_time[(self.injected_latency_delay, self.injected_bandwidth_delay)] = estimated_runtime

    def clean_up(self):
        # get ready for the next iteration
        self.send_next_reqs.clear()
        self.recv_next_reqs.clear()
        self.send_prev_reqs.clear()
        self.recv_prev_reqs.clear()

        self.input_tensors.clear()
        self.output_tensors.clear()
        self.output_tensor_grads.clear()
        self.input_tensor_grads.clear()

        self.total_num_tokens.zero_()

        self.no_sync_func = None
        self.no_sync_context = None

    def validate_args(self):
        args = self.args
        # validate args
        assert args.enable_cdcpp_scheduler, "CDCPPScheduler is not enabled"
        assert (
            args.pipeline_model_parallel_size > 1
        ), "CDCPPScheduler is only for pipeline parallelism"
        assert (
            args.pipeline_model_parallel_size % 2 == 0
        ), "Check _initialize_pp_extra_groups_communicators()"
        assert not args.defer_embedding_wgrad_compute
        assert not args.variable_seq_lengths

        if hasattr(args, 'dynamic_schedule') and args.dynamic_schedule == 'dynamic_mb':
            assert args.tensor_model_parallel_size == 1, (
                "dynamic_mb schedule currently requires tensor_model_parallel_size=1"
            )

        if self.use_static_schedule:
            pass

        assert (
            not args.align_grad_reduce
        ), "align_grad_reduce is not supported, therefore grad_sync_func must be None"
        if hasattr(args, "grad_sync_func"):
            # grad_sync_func may not be set now. No align_grad_reduce should be enough.
            assert args.grad_sync_func is None

        assert (
            not args.align_param_gather
        ), "align_param_gather is not supported, therefore param_sync_func must be None"
        if hasattr(args, "param_sync_func"):
            # param_sync_func may not be set now. No align_param_gather should be enough.
            assert args.param_sync_func is None

        assert (
            not args.defer_embedding_wgrad_compute
        ), "defer_embedding_wgrad_compute is not supported"

        num_chunks = self.pp_schedule.sys_config.num_chunks
        vp_size = args.virtual_pipeline_model_parallel_size or 1
        assert (
            vp_size == num_chunks
        ), "For compatibility, virtual_pipeline_model_parallel_size is equivalent to number of chunks"
        args.virtual_pipeline_model_parallel_size = num_chunks
        from megatron.core import parallel_state as mpu
        mpu.set_virtual_pipeline_model_parallel_world_size(num_chunks)
        mpu.set_virtual_pipeline_model_parallel_rank(0)

        if self.wgrad_split and not (hasattr(args, 'dynamic_schedule') and args.dynamic_schedule == 'dynamic_mb'):
            assert (
                args.gradient_accumulation_fusion
            ), "W-grad split requires gradient accumulation fusion"

    def update_args_and_config(self, args, config):
        # Megatron may add/modify attributes in args and config after initialization.
        self.args = args
        self.config = config
        self.config.deallocate_pipeline_outputs = False

    def get_wgrad_store(self):
        return self.wgrad_store

    def get_layer_offset(self, dev_id, chunk_id):
        layers_list = self.get_num_layers_in_chunk(dev_id=None, chunk_id=None)
        execution_order = self.pp_schedule.get_pipeline_execution_order()
        idx_in_order = execution_order.index((dev_id, chunk_id))
        return sum(layers_list[:idx_in_order])

    def get_num_layers_in_chunk(self, dev_id=None, chunk_id=None):
        # now we treat vocab and lm head as one layer each. and add unbalanced layers to last several chunks
        assert self.args.head_tail_as_one_layer
        num_layer = self.args.num_layers
        if self.args.head_tail_as_one_layer:
            num_layer = num_layer - 2
        execution_order = self.pp_schedule.get_pipeline_execution_order()
        total_chunks = len(execution_order)

        assert total_chunks >= 2, "CDC scheduler should be enabled with pp >= 2"
        if total_chunks == 2:
            layer_list = [num_layer // 2, num_layer - num_layer // 2]
        else:
            head_tail_layers = (num_layer + 2) // total_chunks - 1
            rest_layers = num_layer - head_tail_layers * 2
            rest_chunks = total_chunks - 2
            remainder = rest_layers % rest_chunks
            layer_list = [rest_layers // rest_chunks] * rest_chunks
            for i in range(remainder):
                # add to last several chunks
                layer_list[-(i + 1)] += 1
            layer_list = [head_tail_layers] + layer_list + [head_tail_layers]

        assert (dev_id is not None or chunk_id is not None) or (
            dev_id is None and chunk_id is None
        )
        if dev_id is not None and chunk_id is not None:
            return layer_list[execution_order.index((dev_id, chunk_id))]
        else:
            return layer_list

    def _is_last_microbatch_for_model_chunk(
        self, compute_task: ComputeTask, number_of_microbatches
    ):
        # To enable grad sync in backward.
        has_w_blocks = self.wgrad_split

        if has_w_blocks:
            return (
                compute_task.task_desc.mb_id == number_of_microbatches - 1
                and compute_task.task_desc.type == "W"
            )
        else:
            return (
                compute_task.task_desc.mb_id == number_of_microbatches - 1
                and compute_task.task_desc.type == "B"
            )

    def schedule_comm_event(self, event: CommEvent, config, tensor_shape, forward_only=False):
        # Optimization 1: read cached values instead of calling parallel_state
        # lookups every event. These are constant for the lifetime of this
        # scheduler. See tmp_sganter/docs/dispatcher_optimizations.md.
        next_rank = self._cached_next_rank
        prev_rank = self._cached_prev_rank

        # With dynamic microbatch sizes, the P2P shape is the per-mb size.
        # The 4-GPU dangling-recv NCCL deadlock with heterogeneous P2P shapes
        # (see tmp_sganter/8_4gpu_hang_root_cause.md Update 15) is fixed by
        # the JIT-recv path below: irecv is deferred to WAIT time so no recv
        # kernel sits dangling on a NCCL stream during cuBLAS first-init's
        # device-wide sync.
        # Win 4: per-event tensor_shape override now reads from a per-mb
        # precomputed list (built once per call to forward_backward_func)
        # instead of allocating a new list on every event.
        if self._cached_tensor_shapes is not None and not forward_only:
            tensor_shape = self._cached_tensor_shapes[event.mb_id]

        # Optimization 1: cached at init time. Same NCCL groups for static and
        # dynamic schedules (the heteropipe dedicated 2-rank P2P groups were
        # removed once JIT-recv made them unnecessary).
        send_next_group = self._cached_extra_send_next_group
        recv_next_group = self._cached_extra_recv_next_group
        send_prev_group = self._cached_extra_send_prev_group
        recv_prev_group = self._cached_extra_recv_prev_group

        assert event.task_type in ["F", "B"]

        if event.task_type == "F":
            send_buffer = get_or_set_pp_io_tensor(
                self.output_tensors, (event.mb_id, event.chunk_id), config, tensor_shape
            )
            recv_buffer = get_or_set_pp_io_tensor(
                self.input_tensors, (event.mb_id, event.chunk_id), config, tensor_shape
            )
        else:
            send_buffer = get_or_set_pp_io_tensor(
                self.input_tensor_grads,
                (event.mb_id, event.chunk_id),
                config,
                tensor_shape,
            )
            recv_buffer = get_or_set_pp_io_tensor(
                self.output_tensor_grads,
                (event.mb_id, event.chunk_id),
                config,
                tensor_shape,
            )

        if event.type == CommEventType.LOCAL_COPY:
            if event.task_type == "F":
                with torch.no_grad():
                    recv_buffer.copy_(
                        self.output_tensors[
                            (event.mb_id, event.prev_task_chunk_id)
                        ].detach()
                    )
            else:
                with torch.no_grad():
                    recv_buffer.copy_(
                        self.input_tensor_grads[
                            (event.mb_id, event.prev_task_chunk_id)
                        ].detach()
                    )
        elif event.type == CommEventType.POST_SEND_NEXT:
            self.send_next_reqs[(event.mb_id, event.chunk_id, event.task_type)] = (
                self._isend_with_optional_stock_spin(
                    send_buffer,
                    next_rank,
                    group=send_next_group,
                    direction="next",
                    bandwidth_delay_ms=self.injected_bandwidth_delay[1] * 1000 if self.cdc_send_next else 0,
                )
            )
        elif event.type == CommEventType.POST_RECV_NEXT:
            # Defer the irecv until WAIT_RECV_NEXT to avoid dangling recv kernels
            # on the NCCL stream (which deadlock cuBLAS first-init device-wide
            # sync — see tmp_sganter/8_4gpu_hang_root_cause.md Update 15).
            _buf = recv_buffer
            _grp = recv_next_group
            _peer = next_rank
            _bw = self.injected_bandwidth_delay[1] * 1000 if self.cdc_recv_next else 0
            self.recv_next_reqs[(event.mb_id, event.chunk_id, event.task_type)] = (
                lambda: self.irecv(_buf, _peer, group=_grp, bandwidth_delay_ms=_bw)
            )
        elif event.type == CommEventType.POST_SEND_PREV:
            self.send_prev_reqs[(event.mb_id, event.chunk_id, event.task_type)] = (
                self._isend_with_optional_stock_spin(
                    send_buffer,
                    prev_rank,
                    group=send_prev_group,
                    direction="prev",
                    bandwidth_delay_ms=self.injected_bandwidth_delay[1] * 1000 if self.cdc_send_prev else 0,
                )
            )
        elif event.type == CommEventType.POST_RECV_PREV:
            # Same JIT-recv pattern as POST_RECV_NEXT.
            _buf = recv_buffer
            _grp = recv_prev_group
            _peer = prev_rank
            _bw = self.injected_bandwidth_delay[1] * 1000 if self.cdc_recv_prev else 0
            self.recv_prev_reqs[(event.mb_id, event.chunk_id, event.task_type)] = (
                lambda: self.irecv(_buf, _peer, group=_grp, bandwidth_delay_ms=_bw)
            )
        elif event.type == CommEventType.WAIT_SEND_NEXT:
            handle = self.send_next_reqs[(event.mb_id, event.chunk_id, event.task_type)]
            assert handle is not None
            handle.wait()
        elif event.type == CommEventType.WAIT_RECV_NEXT:
            handle = self.recv_next_reqs[(event.mb_id, event.chunk_id, event.task_type)]
            assert handle is not None
            if callable(handle):
                # JIT-recv mode: materialize the deferred irecv now.
                handle = handle()
                self.recv_next_reqs[(event.mb_id, event.chunk_id, event.task_type)] = handle
            if self.cdc_recv_next:
                assert hasattr(
                    handle, "wait_with_lat_delay_in_ms"
                ), "Latency injection requires custom pytorch build for wait_with_lat_delay_in_ms"
                # if only bandwidth delay injection, still need this api to inject spin kernel on default stream.
                handle.wait_with_lat_delay_in_ms(
                    timedelta(milliseconds=self.injected_latency_delay[1] * 1000)
                )
            else:
                handle.wait()
        elif event.type == CommEventType.WAIT_SEND_PREV:
            handle = self.send_prev_reqs[(event.mb_id, event.chunk_id, event.task_type)]
            assert handle is not None
            handle.wait()
        elif event.type == CommEventType.WAIT_RECV_PREV:
            handle = self.recv_prev_reqs[(event.mb_id, event.chunk_id, event.task_type)]
            assert handle is not None
            if callable(handle):
                # JIT-recv mode: materialize the deferred irecv now.
                handle = handle()
                self.recv_prev_reqs[(event.mb_id, event.chunk_id, event.task_type)] = handle
            if self.cdc_recv_prev:
                assert hasattr(
                    handle, "wait_with_lat_delay_in_ms"
                ), "Latency injection requires custom pytorch build for wait_with_lat_delay_in_ms"
                # if only bandwidth delay injection, still need this api to inject spin kernel on default stream.
                handle.wait_with_lat_delay_in_ms(
                    timedelta(milliseconds=self.injected_latency_delay[1] * 1000)
                )
            else:
                handle.wait()
        else:
            raise NotImplementedError()

    def schedule_event(
        self, event: TaskEvent, config, tensor_shape, forward_only, num_microbatches
    ):
        if isinstance(event, CommEvent):
            if forward_only:
                mb_id = event.mb_id
                task_type = event.task_type
                if mb_id >= num_microbatches or task_type != "F":
                    return
            self.schedule_comm_event(event, config, tensor_shape, forward_only)
        else:
            raise NotImplementedError()

    def schedule_compute_task(
        self,
        compute_task: ComputeTask,
        model,
        data_iterator,
        forward_step_func,
        tensor_shape,
        forward_data_store,
        collect_non_loss_data,
        first_val_step,
        forward_only,
        num_microbatches,
    ):
        config = get_model_config(model[0])

        for pre_event in compute_task.pre_events:
            self.cdc_print(f"pre_event: {pre_event}", verbose=2)
            self.schedule_event(
                pre_event, config, tensor_shape, forward_only, num_microbatches
            )


        _ = torch.empty(1, device=torch.cuda.current_device()) + 1
        task_type = compute_task.task_desc.type
        chunk_id = compute_task.task_desc.chunk_id
        mb_id = compute_task.task_desc.mb_id
        # Important
        parallel_state.set_virtual_pipeline_model_parallel_rank(chunk_id)

        is_first_stage = parallel_state.is_pipeline_first_stage()
        is_last_stage = parallel_state.is_pipeline_last_stage()

        task_type_to_int = {"F": 0, "B": 1, "W": 2}

        if self.exp_manager.profile_in_current_iter():
            if self.exp_manager.cdc_base_memory < 0:
                self.exp_manager.cdc_base_memory = torch.cuda.memory_allocated(
                    device=torch.cuda.current_device()
                )
            # sync default stream
            torch.cuda.default_stream(torch.cuda.current_device()).synchronize()
            mem_before = torch.cuda.memory_allocated(device=torch.cuda.current_device())
            # high precision timer on cpu
            torch.cuda.default_stream(torch.cuda.current_device()).synchronize()
            time_before = time.perf_counter()

        if self.exp_manager.record_schedule_start_in_current_iter():
            assert (
                not self.exp_manager.profile_in_current_iter()
            ), "No profiling when benchmarking schedule runtime"
            # start timing before first compute task
            torch.cuda.default_stream(torch.cuda.current_device()).synchronize()
            self.exp_manager.exp_logging_iter_time[
                (self.injected_latency_delay, self.injected_bandwidth_delay)
            ].append(time.perf_counter())

        if task_type == "F" and (not forward_only or mb_id < num_microbatches):
            if not self.subblock_scheduling:
                self.cdc_print(
                    f"forward_step mb_id: {mb_id}, chunk_id: {chunk_id}", verbose=2
                )
            else:
                self.cdc_print(
                    f"forward_step mb_id: {mb_id}, chunk_id: {chunk_id}, subpart: {compute_task.task_desc.subpart_start}-{compute_task.task_desc.subpart_end}",
                    verbose=2,
                )
            
            with nvtx.range(f"Dev{self.pp_rank} F: {mb_id} chunk: {chunk_id}"):
                if not self.subblock_scheduling:
                    # Win 2: read precomputed loss_scale instead of recomputing
                    # sum(microbatch_sizes) / per-mb division each task.
                    dyn_loss_scale = (
                        self._dyn_loss_scales[mb_id]
                        if self._dyn_loss_scales is not None
                        else None
                    )

                    # Win 3: cached boolean instead of per-task isinstance().
                    if self._data_iter_is_dynamic:
                        data_iterator[chunk_id].set_next_mb_id(mb_id)

                    self.output_tensors[(mb_id, chunk_id)], num_tokens = forward_step(
                        forward_step_func=forward_step_func,
                        data_iterator=data_iterator[chunk_id],
                        model=model[chunk_id],
                        num_microbatches=num_microbatches,
                        input_tensor=self.input_tensors[(mb_id, chunk_id)]
                        if not is_first_stage
                        else None,
                        forward_data_store=forward_data_store,
                        config=config,
                        collect_non_loss_data=collect_non_loss_data,
                        checkpoint_activations_microbatch=None,  # max_outstanding_backprops, num_microbatches_with_partial_activation_checkpoints
                        is_first_microbatch=check_first_val_step(
                            first_val_step, forward_only, mb_id == 0
                        ),
                        current_microbatch=mb_id,
                        encoder_decoder_xattn=False,
                        loss_scale=dyn_loss_scale,
                    )
                    self.total_num_tokens += num_tokens.item()
                else:
                    subpart_start = compute_task.task_desc.subpart_start
                    subpart_end = compute_task.task_desc.subpart_end
                    num_subparts = compute_task.task_desc.num_subparts
                    for subpart_idx in range(subpart_start, subpart_end):
                        first_subpart = subpart_idx == 0
                        last_subpart = subpart_idx == num_subparts - 1
                        output, token = forward_step_subblock(
                            subpart_idx=subpart_idx,
                            num_subparts=num_subparts,
                            microbatch_idx=mb_id,
                            forward_step_func=forward_step_func,
                            data_iterator=data_iterator[chunk_id],
                            model=model[chunk_id],
                            num_microbatches=num_microbatches,
                            input_tensor=self.input_tensors[(mb_id, chunk_id)] if (not is_first_stage and first_subpart) else None,
                            forward_data_store=forward_data_store,
                            config=config,
                            collect_non_loss_data=collect_non_loss_data,
                            checkpoint_activations_microbatch=None,  # max_outstanding_backprops, num_microbatches_with_partial_activation_checkpoints
                            is_first_microbatch=check_first_val_step(
                                first_val_step, forward_only, mb_id == 0
                            ),
                            current_microbatch=mb_id,
                        )
                        if last_subpart:
                            self.output_tensors[(mb_id, chunk_id)] = output
                            self.total_num_tokens += token.item()

        elif task_type == "B" and not forward_only:
            # Only training. In eval, we skip backward.

            if not self.subblock_scheduling:
                self.cdc_print(
                    f"backward_step mb_id: {mb_id}, chunk_id: {chunk_id}", verbose=2
                )
            else:
                self.cdc_print(
                    f"backward_step mb_id: {mb_id}, chunk_id: {chunk_id}, subpart: {compute_task.task_desc.subpart_start}-{compute_task.task_desc.subpart_end}",
                    verbose=2,
                )
            with nvtx.range(f"Dev{self.pp_rank} B: {mb_id} chunk: {chunk_id}"):
                # enable grad sync for the last microbatch
                if self._is_last_microbatch_for_model_chunk(
                    compute_task, num_microbatches
                ):
                    self.enable_grad_sync(chunk_id)
                    self.cdc_print(f"enable_grad_sync for last microbatch, task: {mb_id}, chunk: {chunk_id}", verbose=2)

                if not self.subblock_scheduling:
                    output_tensor_grad = (
                        self.output_tensor_grads[(mb_id, chunk_id)]
                        if not is_last_stage
                        else None
                    )
                    self.input_tensor_grads[(mb_id, chunk_id)] = backward_step(
                        input_tensor=self.input_tensors[(mb_id, chunk_id)],
                        output_tensor=self.output_tensors[(mb_id, chunk_id)],
                        output_tensor_grad=output_tensor_grad,
                        model_type=get_model_type(model[chunk_id]),
                        config=config,
                    )
                    # release tensors
                    self.input_tensors[(mb_id, chunk_id)] = None
                    self.output_tensors[(mb_id, chunk_id)] = None
                    self.output_tensor_grads[(mb_id, chunk_id)] = None

                    if is_first_stage:
                        self.input_tensor_grads[(mb_id, chunk_id)] = None

                    if self.wgrad_store is not None:
                        self.wgrad_store.finish_collection_wgrad_block()

                else:
                    num_subparts = compute_task.task_desc.num_subparts
                    subpart_start = num_subparts - compute_task.task_desc.subpart_start - 1
                    subpart_end = num_subparts - compute_task.task_desc.subpart_end - 1
                    
                    for subpart_idx in range(subpart_start, subpart_end, -1):
                        first_subpart = subpart_idx == 0
                        last_subpart = subpart_idx == num_subparts - 1
                        output_tensor_grad = (
                            self.output_tensor_grads[(mb_id, chunk_id)]
                            if last_subpart and not is_last_stage
                            else None
                        )
                        input_grad = backward_step_subblock(
                            subpart_idx=subpart_idx,
                            num_subparts=num_subparts,
                            microbatch_idx=mb_id,
                            model=model[chunk_id],
                            input_tensor=self.input_tensors[(mb_id, chunk_id)] if first_subpart else None,
                            output_tensor=self.output_tensors[(mb_id, chunk_id)] if last_subpart else None,
                            output_tensor_grad=output_tensor_grad,
                            config=config,
                        )
                        if first_subpart:
                            self.input_tensor_grads[(mb_id, chunk_id)] = input_grad
                        
                            # release tensors
                            self.input_tensors[(mb_id, chunk_id)] = None
                            self.output_tensors[(mb_id, chunk_id)] = None
                            self.output_tensor_grads[(mb_id, chunk_id)] = None

                            if is_first_stage:
                                self.input_tensor_grads[(mb_id, chunk_id)] = None

                            if self.wgrad_store is not None:
                                self.wgrad_store.finish_collection_wgrad_block()

                
                self.disable_grad_sync(chunk_id)

        elif task_type == "W" and not forward_only:
            assert self.wgrad_store is not None

            if not self.subblock_scheduling:
                self.cdc_print(
                    f"wgrad_step mb_id: {mb_id}, chunk_id: {chunk_id}", verbose=2
                )
            else:
                self.cdc_print(
                    f"wgrad_step mb_id: {mb_id}, chunk_id: {chunk_id}, subpart: {compute_task.task_desc.subpart_start}-{compute_task.task_desc.subpart_end}",
                    verbose=2,
                )
            with nvtx.range(f"Dev{self.pp_rank} W: {mb_id} chunk: {chunk_id}"):
                if not self.subblock_scheduling:
                    self.wgrad_store.compute_wgrad_block()
                    if self._is_last_microbatch_for_model_chunk(
                        compute_task, num_microbatches
                    ):
                        model_chunk = model[chunk_id]
                        model_chunk.start_grad_sync()
                else:
                    num_subparts = compute_task.task_desc.num_subparts
                    subpart_start = compute_task.task_desc.subpart_start
                    subpart_end = compute_task.task_desc.subpart_end
                    self.wgrad_store.compute_wgrad_subblock(subpart_end-subpart_start, num_subparts)
                    last_subpart = subpart_end == num_subparts
                    if last_subpart and self._is_last_microbatch_for_model_chunk(
                        compute_task, num_microbatches
                    ):
                        model_chunk = model[chunk_id]
                        model_chunk.start_grad_sync()


        if self.exp_manager.profile_in_current_iter():
            # sync default stream
            torch.cuda.default_stream(torch.cuda.current_device()).synchronize()
            time_after = time.perf_counter()
            mem_after = torch.cuda.memory_allocated(device=torch.cuda.current_device())
            self.exp_manager.cdc_compute_profile_dict[
                (mb_id, chunk_id, task_type_to_int[task_type])
            ] = [
                time_after - time_before,
                mem_before,
                mem_after,
            ]

        for post_event in compute_task.post_events:
            self.cdc_print(f"post_event: {post_event}", verbose=2)
            self.schedule_event(
                post_event, config, tensor_shape, forward_only, num_microbatches
            )

    def deallocate_tensor_in_dicts(self):
 
        # output tensors and input tensor grads
        for (mb_id, chunk_id, task_type), handle in list(
            self.send_next_reqs.items()
        ) + list(self.send_prev_reqs.items()):
            if task_type == "F":
                if (
                    self.output_tensors[(mb_id, chunk_id)] is not None
                    and handle is not None
                    and handle.is_completed()
                ):
                    self.cdc_print(
                        f"deallocate_output_tensor: {mb_id}, {chunk_id}", verbose=2
                    )
                    deallocate_output_tensor(
                        self.output_tensors[(mb_id, chunk_id)],
                        self.config.deallocate_pipeline_outputs,
                    )
            elif task_type == "B":
                if (
                    self.input_tensor_grads[(mb_id, chunk_id)] is not None
                    and handle is not None
                    and handle.is_completed()
                ):
                    self.cdc_print(
                        f"releasing input grad ref: {mb_id}, {chunk_id}", verbose=2
                    )
                    self.input_tensor_grads[(mb_id, chunk_id)] = None

    def setup_grad_sync(self):
        # grad sync
        # Disable async grad reductions
        assert self.config is not None
        num_chunks = self.pp_schedule.sys_config.num_chunks
        if self.config.no_sync_func is None:
            from contextlib import nullcontext
            self.no_sync_func = [nullcontext] * num_chunks
        elif isinstance(self.config.no_sync_func, list):
            self.no_sync_func = self.config.no_sync_func
        else:
            self.no_sync_func = [self.config.no_sync_func,]

        assert len(self.no_sync_func) == num_chunks
        self.cdc_print(f"no_sync_func: {self.no_sync_func}", verbose=2)

        self.no_sync_context = [None] * num_chunks

    def disable_grad_sync(self, chunk_id):
        """Disable asynchronous grad reductions"""
        if self.no_sync_context[chunk_id] is None:
            self.no_sync_context[chunk_id] = self.no_sync_func[chunk_id]()
            self.no_sync_context[chunk_id].__enter__()

    def enable_grad_sync(self, chunk_id):
        """Enable asynchronous grad reductions"""
        if self.no_sync_context[chunk_id] is not None:
            self.no_sync_context[chunk_id].__exit__(None, None, None)
            self.no_sync_context[chunk_id] = None

    def forward_backward_func(
        self,
        *,
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,  # unused
        micro_batch_size: int,  # unused
        decoder_seq_length: int = None,  # unused
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
        first_val_step: bool = None,
    ):
        # self.cdc_print(f'first_stage (virtual): {parallel_state.is_pipeline_first_stage(ignore_virtual=True)} ({parallel_state.is_pipeline_first_stage()}), last_stage (virtual): {parallel_state.is_pipeline_last_stage(ignore_virtual=True)} ({parallel_state.is_pipeline_last_stage()})')

        if not forward_only:
            assert num_microbatches == self.pp_schedule.sys_config.num_microbatches
        else:
            assert num_microbatches <= self.pp_schedule.sys_config.num_microbatches

        if self.pp_schedule.has_multiple_chunks():
            assert isinstance(model, list), "Model has multiple chunks"
            assert [isinstance(chunk, torch.nn.Module) for chunk in model]
            assert isinstance(
                data_iterator, list
            ), "Expect each chunk to have its own data iterator"
            config = get_model_config(model[0])
        else:
            if isinstance(model, list):
                assert len(model) == 1, "Model should only have one chunk"
            else:
                model = [model]
            assert isinstance(model[0], torch.nn.Module)
            if isinstance(data_iterator, list):
                assert (
                    len(data_iterator) == 1
                ), "Data iterator should only have one chunk"
            else:
                data_iterator = [data_iterator]
            config = get_model_config(model[0])

        assert all(
            [get_model_type(chunk) != ModelType.encoder_and_decoder for chunk in model]
        )

        if self.exp_manager.profile_in_current_iter():
            if len(self.exp_manager.cdc_chunk_parameters) == 0:
                for chunk_id, chunk in enumerate(model):
                    assert len(model) == self.pp_schedule.sys_config.num_chunks
                    # get number
                    num_params = sum(p.numel() for p in chunk.parameters())
                    self.exp_manager.cdc_chunk_parameters[chunk_id] = num_params
                    self.cdc_print(
                        f"model info: chunk {chunk_id} has {num_params} parameters"
                    )
                    self.cdc_print(
                        f"model info :chunk {chunk_id} model: {chunk.__repr__()} "
                    )

            for chunk_id in range(self.pp_schedule.sys_config.num_chunks):
                if chunk_id not in self.exp_manager.cdc_layer_info:
                    first_stage_rank = self.pp_schedule.get_pipeline_first_stage_rank()
                    last_stage_rank = self.pp_schedule.get_pipeline_last_stage_rank()
                    cur_chunk_has_vocab_embedding = (
                        self.pp_rank == first_stage_rank and chunk_id == 0
                    )
                    cur_chunk_has_lm_head = (
                        self.pp_rank == last_stage_rank
                        and chunk_id == self.pp_schedule.sys_config.num_chunks - 1
                    )
                    num_layers = self.get_num_layers_in_chunk(
                        dev_id=self.pp_rank, chunk_id=chunk_id
                    )
                    self.exp_manager.cdc_layer_info[chunk_id] = (
                        cur_chunk_has_vocab_embedding,
                        cur_chunk_has_lm_head,
                        num_layers,
                    )
                    self.cdc_print(
                        f"model info: chunk {chunk_id} has {num_layers} layers, vocab embedding: {cur_chunk_has_vocab_embedding}, lm head: {cur_chunk_has_lm_head}"
                    )

        forward_data_store = []

         # # Needed only when gradients are finalized in M-Core
        if config.finalize_model_grads_func is not None and not forward_only:
            embedding_module = clear_embedding_activation_buffer(config, model)


        self.setup_grad_sync()

        for chunk_id in range(self.pp_schedule.sys_config.num_chunks):
            self.disable_grad_sync(chunk_id)
            # is_last_microbatch is set to True by zero_grad_buffer() before this call, so no assertion needed

        # Compute adjusted seq_length (shared across all microbatches)
        adjusted_seq_length = seq_length // parallel_state.get_context_parallel_world_size()
        if config.sequence_parallel:
            adjusted_seq_length = (
                adjusted_seq_length // parallel_state.get_tensor_model_parallel_world_size()
            )
        dtype_size = torch.tensor([], dtype=config.pipeline_dtype).element_size()

        # Always define the equal-size tensor_shape (needed for eval and as fallback).
        tensor_shape = [adjusted_seq_length, micro_batch_size, config.hidden_size]
        self.pp_comm_size_bytes = (
            tensor_shape[0] * tensor_shape[1] * tensor_shape[2] * dtype_size
        )
        if self.microbatch_sizes is not None and not forward_only:
            # Dynamic microbatch sizes — pp_comm_size_bytes uses the largest
            max_mb_size = max(self.microbatch_sizes)
            self.pp_comm_size_bytes = (
                adjusted_seq_length * max_mb_size * config.hidden_size * dtype_size
            )

        # Affine profiling sweep: runs once, after the canonical profile iter
        # has populated total.json and before the dynamic_mb LP consumes its
        # cost data. Gated by --cdc_profile_affine and only meaningful for
        # the dynamic_mb path.
        if (
            getattr(self.args, "cdc_profile_affine", False)
            and not forward_only
            and not self._affine_profile_done
            and self.exp_manager.profiling_done()
            and self.use_dynamic_schedule
            and self.dynamic_schedule_type == "dynamic_mb"
        ):
            self._run_affine_profile(
                model=model,
                config=config,
                adjusted_seq_length=adjusted_seq_length,
                micro_batch_size=micro_batch_size,
            )
            self._affine_profile_done = True

        self.update_schedule_with_latency_bandwidth()

        # Wrap data iterators for dynamic microbatch sizes (must happen after
        # update_schedule_with_latency_bandwidth which may set self.microbatch_sizes).
        # Win 3: store boolean flag once so per-task hot loop can skip a per-task
        # isinstance() check.
        is_dynamic = self.microbatch_sizes is not None and not forward_only
        # Profiling escape hatch: when CDC_BYPASS_DYNAMIC_ITERATOR=1, pretend
        # the schedule is static. Only correct when every entry of
        # microbatch_sizes equals micro_batch_size.
        if self._bypass_dynamic_iterator:
            is_dynamic = False
        self._data_iter_is_dynamic = is_dynamic
        if is_dynamic:
            data_iterator = [
                DynamicMicrobatchIterator(di, self.microbatch_sizes, micro_batch_size)
                for di in data_iterator
            ]
            # Win 4: build the per-mb tensor_shape cache once per iter.
            # schedule_comm_event will read directly from this list per event,
            # avoiding ~300 list() allocations + per-event index lookups.
            self._cached_tensor_shapes = [
                [adjusted_seq_length, s, config.hidden_size]
                for s in self.microbatch_sizes
            ]
            # Per-task placeholder shape (max-size buffer that fits any mb).
            cur_tensor_shape_template = [adjusted_seq_length, max(self.microbatch_sizes), config.hidden_size]
        else:
            self._cached_tensor_shapes = None
            cur_tensor_shape_template = None

        # cProfile instrumentation around the dispatcher (env-var gated).
        profile_this_iter = (
            self._profile_dispatcher
            and not forward_only
            and not self._profile_dumped
            and self._profile_iter_counter >= self._profile_warmup
            and self._profile_iter_counter < self._profile_last
        )
        if profile_this_iter:
            if self._profile_obj is None:
                self._profile_obj = cProfile.Profile()
            self._profile_obj.enable()

        for idx, compute_task in enumerate(self.pp_execution_plan_cur_device):
            # self.cdc_print(f"compute_task: {compute_task}")
            self.exp_manager.exp_logging_first_mb = True if idx == 0 else False

            # tensor_shape: with variable mb sizes, schedule_comm_event overrides
            # tensor_shape[1] per-event from self._cached_tensor_shapes. The base
            # shape here is only used as a fallback for forward-only / static.
            cur_tensor_shape = cur_tensor_shape_template if is_dynamic else tensor_shape

            self.schedule_compute_task(
                compute_task=compute_task,
                model=model,
                data_iterator=data_iterator,
                forward_step_func=forward_step_func,
                tensor_shape=cur_tensor_shape,
                forward_data_store=forward_data_store,
                collect_non_loss_data=collect_non_loss_data,
                first_val_step=first_val_step,
                forward_only=forward_only,
                num_microbatches=num_microbatches,
            )
            self.deallocate_tensor_in_dicts()

        if profile_this_iter:
            self._profile_obj.disable()
            if self._profile_iter_counter == self._profile_last - 1 and not self._profile_dumped:
                self._profile_obj.dump_stats(self._profile_out)
                self._profile_dumped = True
                print(
                    f"[CDC_PROFILE_DISPATCHER] rank{self.pp_rank} dumped cProfile "
                    f"stats covering iters [{self._profile_warmup}, {self._profile_last}) "
                    f"to {self._profile_out}",
                    flush=True,
                )
        if self._profile_dispatcher and not forward_only:
            self._profile_iter_counter += 1

        assert self.wgrad_store is None or self.wgrad_store.is_empty()

        for chunk_id in range(self.pp_schedule.sys_config.num_chunks):
            self.enable_grad_sync(chunk_id)

        if config.finalize_model_grads_func is not None and not forward_only:
            # If defer_embedding_wgrad_compute is enabled we need to do the
            # weight gradient GEMM's here.
            finish_embedding_wgrad_compute(config, embedding_module)

            # Finalize model grads (perform full grad all-reduce / reduce-scatter for
            # data parallelism, layernorm all-reduce for sequence parallelism, and
            # embedding all-reduce for pipeline parallelism).
            config.finalize_model_grads_func(
                model,
                self.total_num_tokens if config.calculate_per_token_loss else None,
            )

        if self.exp_manager.profile_in_current_iter():
            assert len(self.exp_manager.cdc_chunk_parameters) == self.pp_schedule.sys_config.num_chunks
            self.exp_manager.cdc_dp_comm_profiles = self.dp_benchmark(self.exp_manager.cdc_chunk_parameters)
            # write profile result
            if self.exp_manager.cdc_log_profile:
                self.cdc_print(
                    f"cdc_compute_profile_dict: {self.exp_manager.cdc_compute_profile_dict}"
                )
                if self.pp_rank == 0:
                    self.cdc_print(
                        f"cdc_comm_profiles: {self.exp_manager.cdc_comm_profiles}"
                    )
                self.cdc_print(f"cdc_base_memory: {self.exp_manager.cdc_base_memory}")
                self.cdc_print(
                    f"cdc_chunk_parameters: {self.exp_manager.cdc_chunk_parameters}"
                )
                self.cdc_print(f"cdc_dp_comm_profiles: {self.exp_manager.cdc_dp_comm_profiles}")
                self.cdc_print(f"cdc_layer_info: {self.exp_manager.cdc_layer_info}")

                with open(self.exp_manager.profile_result_rank_file, "w") as f:
                    json.dump(
                        {
                            "compute": tuple_keys_to_str(
                                self.exp_manager.cdc_compute_profile_dict
                            ),
                            "comm": self.exp_manager.cdc_comm_profiles,
                            "dp_comm": self.exp_manager.cdc_dp_comm_profiles,
                            "base_mem": self.exp_manager.cdc_base_memory,
                            "params": self.exp_manager.cdc_chunk_parameters,
                            "layer_info": self.exp_manager.cdc_layer_info,
                        },
                        f,
                    )

            dist.barrier()

            # rank 0 concludes the profile result
            num_chunks = self.pp_schedule.sys_config.num_chunks
            if dist.get_rank() == 0:
                json_file_path = self.exp_manager.profile_result_path
                json_results = []
                pp_size = parallel_state.get_pipeline_model_parallel_world_size()
                for i in range(pp_size):
                    with open(os.path.join(json_file_path, f"{i}.json"), "r") as f:
                        json_results.append(json.load(f))

                T_F_list = np.zeros((num_chunks, pp_size))
                T_B_list = np.zeros((num_chunks, pp_size))
                T_W_list = np.zeros((num_chunks, pp_size))
                T_alpha_matrix = np.zeros((pp_size, pp_size))
                T_bw_matrix = np.zeros((pp_size, pp_size))
                T_DP_list = np.zeros((num_chunks, pp_size))
                M_F_list = np.zeros((num_chunks, pp_size))
                M_B_list = np.zeros((num_chunks, pp_size))
                M_W_list = np.zeros((num_chunks, pp_size))
                M_Limit_list = []
                base_mem_list = []
                max_gpu_mem = torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).total_memory
                for i in range(pp_size):
                    compute_profile = str_keys_to_tuple(json_results[i]["compute"])
                    base_mem_list.append(json_results[i]["base_mem"])
                    T_cur_dev = [[[] for _ in range(num_chunks)] for _ in range(3)]
                    M_cur_dev = [[[] for _ in range(num_chunks)] for _ in range(3)]
                    for key, value in compute_profile.items():
                        cur_mb, cur_chunk, cur_type = key
                        compute_time, mem_before, mem_after = value
                        T_cur_dev[cur_type][cur_chunk].append(compute_time)
                        M_cur_dev[cur_type][cur_chunk].append(mem_after - mem_before)
                    for cur_chunk in range(num_chunks):
                        T_F_list[cur_chunk][i] = np.min(T_cur_dev[0][cur_chunk])
                        T_B_list[cur_chunk][i] = np.min(T_cur_dev[1][cur_chunk])
                        if len(T_cur_dev[2][cur_chunk]) > 0:
                            T_W_list[cur_chunk][i] = np.min(T_cur_dev[2][cur_chunk])
                        else:
                            T_W_list[cur_chunk][i] = 0
                    for cur_chunk in range(num_chunks):
                        M_F_list[cur_chunk][i] = np.median(M_cur_dev[0][cur_chunk])
                        if len(M_cur_dev[2][cur_chunk]) > 0:
                            M_W_list[cur_chunk][i] = np.median(M_cur_dev[2][cur_chunk])
                        else:
                            M_W_list[cur_chunk][i] = 0
                        M_B_list[cur_chunk][i] = -M_F_list[cur_chunk][i] - M_W_list[cur_chunk][i]
                        
                    M_Limit_list.append(int((max_gpu_mem - base_mem_list[-1]) * 0.98))
                # T_alpha
                alpha_to_next, alpha_to_prev, beta_to_next, beta_to_prev = json_results[
                    0
                ]["comm"]
                message_size = self.pp_comm_size_bytes
                for i in range(pp_size):
                    T_alpha_matrix[i, (i + 1) % pp_size] = alpha_to_next[i]
                    T_bw_matrix[i, (i + 1) % pp_size] = beta_to_next[i] * message_size
                    T_alpha_matrix[i, (i - 1) % pp_size] = alpha_to_prev[i]
                    T_bw_matrix[i, (i - 1) % pp_size] = beta_to_prev[i] * message_size

                # make T_alpha symmetric
                T_alpha_matrix = (T_alpha_matrix + T_alpha_matrix.T) / 2
                # make T_bw symmetric
                T_bw_matrix = (T_bw_matrix + T_bw_matrix.T) / 2
                
                # DP comm
                T_DP_list = np.array(json_results[0]["dp_comm"])
                with open(os.path.join(json_file_path, "total.json"), "w") as f:
                    json.dump(
                        {
                            "T_F": T_F_list.tolist(),
                            "T_B": T_B_list.tolist(),
                            "T_W": T_W_list.tolist(),
                            "T_alpha": T_alpha_matrix.tolist(),
                            "T_bw": T_bw_matrix.tolist(),
                            "T_DP": T_DP_list.tolist(),
                            "M_F": M_F_list.tolist(),
                            "M_B": M_B_list.tolist(),
                            "M_W": M_W_list.tolist(),
                            "M_Limit": M_Limit_list,
                            "M_base": base_mem_list,
                            "M_dev_max": max_gpu_mem,
                        },
                        f,
                    )
            dist.barrier()
            self.exp_manager.read_profile_result()
            if self.pp_schedule_generator is not None:
                self.pp_schedule_generator.initialize()

        if self.exp_manager.record_schedule_end_in_current_iter():
            
            torch.cuda.default_stream(torch.cuda.current_device()).synchronize()
            self.exp_manager.exp_logging_iter_time[
                (self.injected_latency_delay, self.injected_bandwidth_delay)
            ][-1] = (
                time.perf_counter()
                - self.exp_manager.exp_logging_iter_time[
                    (self.injected_latency_delay, self.injected_bandwidth_delay)
                ][-1]
            )
            self.exp_manager.exp_logging_max_allocated_mem[
                (self.injected_latency_delay, self.injected_bandwidth_delay)
            ].append(torch.cuda.max_memory_allocated())
            torch.cuda.reset_max_memory_allocated()

        if self.exp_manager.write_to_json_in_current_iter():
            # write to json
            timpstamp = int(time.time())
            schedule = (
                self.args.static_schedule
                if self.use_static_schedule
                else self.args.dynamic_schedule
            )
            result_dict = {
                        "iter_time": tuple_keys_to_str(self.exp_manager.exp_logging_iter_time),
                        "max_mem": tuple_keys_to_str(self.exp_manager.exp_logging_max_allocated_mem),
                        "perf_model_time": tuple_keys_to_str(self.exp_manager.exp_logging_perf_model_iter_time),
                        "config": {
                            "schedule": schedule,
                            "TP": self.args.tensor_model_parallel_size,
                            "PP": self.args.pipeline_model_parallel_size,
                            "DP": self.args.data_parallel_size,
                            "seq_len": self.args.seq_length,
                            "GBS": self.args.global_batch_size,
                            "MBS": self.args.micro_batch_size,
                            "n_DC": self.args.num_dc,
                            "cdc_delay": self.args.cdc_latency_bandwidth_delay_as_F_stage,
                            "dyn_extra_mem_factor": self.args.dynamic_extra_mem_factor,
                            "num_layers": self.args.num_layers,
                            "cdc_exp_tf_block_size": self.args.cdc_exp_tf_block_size,
                            "recomputation": self.args.recompute_granularity is not None,
                        },
                    }
            with open(
                os.path.join(
                    self.exp_manager.exp_logging_path, f"exp_{timpstamp}.json"
                ),
                "w",
            ) as f:
                json.dump(
                    result_dict,
                    f,
                )
            # also overwrite the final result json if exist
            with open(
                os.path.join(self.exp_manager.exp_logging_path, "exp_final.json"), "w"
            ) as f:
                json.dump(
                    result_dict,
                    f,
                )
                

        self.clean_up()

        self.cdc_print(f"forward_data_store: {forward_data_store}", verbose=2)
        return forward_data_store

    def get_forward_backward_func(self):
        return self.forward_backward_func

    def send(
        self,
        tensor: torch.Tensor,
        dst: int,
        group: dist.ProcessGroupNCCL | None = None,
        bandwidth_delay_ms: int = 0,
    ):
        prev_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
        next_rank = parallel_state.get_pipeline_model_parallel_next_rank()
        send_prev_group = parallel_state.get_pipeline_extra_send_prev_group()
        send_next_group = parallel_state.get_pipeline_extra_send_next_group()
        if (dst == prev_rank and self.cdc_send_prev and group is send_prev_group) or (
            dst == next_rank and self.cdc_send_next and group is send_next_group
        ):
            bandwidth_delay_to_inject = bandwidth_delay_ms
        else:
            bandwidth_delay_to_inject = 0
        dist.send(tensor, dst, group=group, tag=int(bandwidth_delay_to_inject))
        # dist.send(tensor, dst, group=group, tag=0)

    def _maybe_calibrate_stock_inject_cycles(self) -> None:
        """One-shot calibration of cycles-per-ms for the local SM clock.

        Uses a 100ms `torch.cuda._sleep` and times it with CUDA events. The
        hardcoded fallback (1,784,909 cycles/ms) is correct for H100 NVL —
        this just makes the code portable to other GPUs.
        """
        if self._stock_inject_calibrated:
            return
        if not torch.cuda.is_available():
            self._stock_inject_calibrated = True
            return
        try:
            device = torch.cuda.current_device()
            target_cycles = int(100 * self._stock_inject_cycles_per_ms)
            torch.cuda.synchronize(device)
            # Warmup.
            torch.cuda._sleep(target_cycles)
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            torch.cuda._sleep(target_cycles)
            end.record()
            torch.cuda.synchronize(device)
            measured_ms = start.elapsed_time(end)
            if measured_ms > 0:
                self._stock_inject_cycles_per_ms = float(target_cycles) / measured_ms
        except Exception as exc:  # pragma: no cover - belt-and-suspenders
            self.cdc_print(
                f"stock latency calibration failed ({exc}); using fallback "
                f"{self._stock_inject_cycles_per_ms} cycles/ms",
                verbose=1,
            )
        finally:
            self._stock_inject_calibrated = True
            self.cdc_print(
                f"stock latency injection: calibrated cycles/ms = "
                f"{self._stock_inject_cycles_per_ms:.0f}",
                verbose=1,
            )

    def _stock_inject_active_this_iter(self) -> bool:
        """True iff stock-PyTorch latency injection should fire on this iter."""
        if not self._stock_inject_enabled:
            return False
        cur_iter = getattr(self.args, "curr_iteration", 0)
        return cur_iter >= self._stock_inject_warmup_iters

    def _isend_with_optional_stock_spin(
        self,
        tensor: torch.Tensor,
        dst: int,
        group: dist.ProcessGroupNCCL | None,
        direction: str,  # "next" or "prev"
        bandwidth_delay_ms: int = 0,
    ) -> dist.Work:
        """Wrap `self.isend` so that, when stock-PyTorch latency injection is
        enabled for this direction and we're past warmup, the actual NCCL send
        kernel waits for a `torch.cuda._sleep` spin on a dedicated CUDA stream.

        The spin runs concurrently with default-stream compute — only the
        send kernel waits for it. Receiver naturally sees the data
        `latency_ms` later.
        """
        if not self._stock_inject_active_this_iter():
            return self.isend(
                tensor, dst, group=group, bandwidth_delay_ms=bandwidth_delay_ms
            )
        delay_this_dir = (
            (direction == "next" and self._stock_inject_delay_next)
            or (direction == "prev" and self._stock_inject_delay_prev)
        )
        if not delay_this_dir:
            return self.isend(
                tensor, dst, group=group, bandwidth_delay_ms=bandwidth_delay_ms
            )
        self._maybe_calibrate_stock_inject_cycles()
        device = torch.cuda.current_device()
        if direction == "next":
            stream = self._stock_inject_send_next_stream
            if stream is None:
                stream = torch.cuda.Stream(device=device)
                self._stock_inject_send_next_stream = stream
        else:
            stream = self._stock_inject_send_prev_stream
            if stream is None:
                stream = torch.cuda.Stream(device=device)
                self._stock_inject_send_prev_stream = stream
        cycles = int(self._stock_inject_lat_ms * self._stock_inject_cycles_per_ms)
        # Cross-stream data hazard: `tensor` is being produced by a forward/
        # backward kernel on the default stream. NCCL's backend records its
        # event from the CURRENT stream — if we just enter spin_stream here,
        # NCCL would only wait for the spin, not for the default-stream
        # producer kernel. That's a use-before-write race and shows up as
        # corrupted gradients on the first injection iter. Force the spin
        # stream to wait for the default stream's pending work first, then
        # the spin + isend chain inherits the correct happens-before.
        default_stream = torch.cuda.default_stream(device)
        stream.wait_stream(default_stream)
        with torch.cuda.stream(stream):
            if cycles > 0:
                torch.cuda._sleep(cycles)
            work = self.isend(
                tensor, dst, group=group, bandwidth_delay_ms=bandwidth_delay_ms
            )
        return work

    def isend(
        self,
        tensor: torch.Tensor,
        dst: int,
        group: dist.ProcessGroupNCCL | None = None,
        bandwidth_delay_ms: int = 0,
    ) -> dist.Work:
        prev_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
        next_rank = parallel_state.get_pipeline_model_parallel_next_rank()
        send_prev_group = parallel_state.get_pipeline_extra_send_prev_group()
        send_next_group = parallel_state.get_pipeline_extra_send_next_group()
        if (dst == prev_rank and self.cdc_send_prev and group is send_prev_group) or (
            dst == next_rank and self.cdc_send_next and group is send_next_group
        ):
            bandwidth_delay_to_inject = bandwidth_delay_ms
        else:
            bandwidth_delay_to_inject = 0
        return dist.isend(tensor, dst, group=group, tag=int(bandwidth_delay_to_inject))
        # return dist.isend(tensor, dst, group=group, tag=0)

    def recv(
        self,
        tensor: torch.Tensor,
        src: int,
        group: dist.ProcessGroupNCCL | None = None,
        bandwidth_delay_ms: int = 0,
    ):
        prev_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
        next_rank = parallel_state.get_pipeline_model_parallel_next_rank()
        recv_prev_group = parallel_state.get_pipeline_extra_recv_prev_group()
        recv_next_group = parallel_state.get_pipeline_extra_recv_next_group()
        if (src == prev_rank and self.cdc_recv_prev and group is recv_prev_group) or (
            src == next_rank and self.cdc_recv_next and group is recv_next_group
        ):
            work = dist.irecv(tensor, src, group=group, tag=int(bandwidth_delay_ms))
            assert hasattr(
                work, "wait_with_lat_delay_in_ms"
            ), "Latency injection requires custom pytorch build for wait_with_lat_delay_in_ms"
            work.wait_with_lat_delay_in_ms(
                timedelta(milliseconds=self.injected_latency_delay[1] * 1000)
            )
        else:
            dist.recv(tensor, src, group=group, tag=0)
        # dist.recv(tensor, src, group=group, tag=0)

    def irecv(
        self,
        tensor: torch.Tensor,
        src: int,
        group: dist.ProcessGroupNCCL | None = None,
        bandwidth_delay_ms: int = 0,
    ):
        prev_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
        next_rank = parallel_state.get_pipeline_model_parallel_next_rank()
        recv_prev_group = parallel_state.get_pipeline_extra_recv_prev_group()
        recv_next_group = parallel_state.get_pipeline_extra_recv_next_group()
        if (src == prev_rank and self.cdc_recv_prev and group is recv_prev_group) or (
            src == next_rank and self.cdc_recv_next and group is recv_next_group
        ):
            return dist.irecv(tensor, src, group=group, tag=int(bandwidth_delay_ms))
        else:
            return dist.irecv(tensor, src, group=group, tag=0)
        # return dist.irecv(tensor, src, group=group, tag=0)

    def cdc_print(self, msg: str, rank=None, verbose=1):
        if verbose > self.cdc_verbose_print:
            return

        my_rank = dist.get_rank()
        if rank is not None and my_rank != rank:
            return
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        dp_rank = parallel_state.get_data_parallel_rank()
        if self.cdc_print_rank == -1 or my_rank == self.cdc_print_rank:
            print(
                f"[CDC] Global[{my_rank}] TP[{tp_rank}] PP[{pp_rank}] DP[{dp_rank}]:    {msg}"
            )

    def _run_affine_profile(self, model, config, adjusted_seq_length, micro_batch_size):
        """One-shot affine profiling sweep. Called from forward_backward_func
        between the canonical profile iter and the LP solve. Writes
        per-rank ``affine_profile_rank{N}.json`` and (rank 0)
        ``affine_profile_comm.json`` next to ``total.json`` so the LP loader
        can pick them up.

        Compute is profiled per non-first stage; first-stage compute keeps
        the existing single-point scaling because synthetic dataloader
        equivalents are out of scope here. Communication is profiled per
        directed adjacent edge."""
        args = self.args
        N = args.global_batch_size // (
            args.data_parallel_size if hasattr(args, "data_parallel_size") else 1
        )

        sizes = args.cdc_profile_affine_sizes
        if sizes is None:
            sizes = affine_profiler.default_size_grid(N)
        else:
            sizes = sorted(set(int(s) for s in sizes if 1 <= int(s) <= N))
            if not sizes:
                sizes = affine_profiler.default_size_grid(N)

        warmup = int(getattr(args, "cdc_profile_affine_warmup_iters", 2))
        measure = int(getattr(args, "cdc_profile_affine_measure_iters", 5))
        dtype = config.pipeline_dtype

        self.cdc_print(
            f"[affine] starting sweep on rank {self.pp_rank}: sizes={sizes}, warmup={warmup}, measure={measure}",
            rank=0,
        )
        t0 = time.perf_counter()

        # --- Compute sweep (per chunk on this rank) --------------------------
        is_first = parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        is_last = parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        chunks = model if isinstance(model, list) else [model]
        # Vocab size for synthetic-token construction on the first stage.
        # Pull from config / args if available; default to a small valid value
        # so the embedding lookup gets in-range indices.
        vocab_size = getattr(config, "vocab_size", None) or getattr(args, "padded_vocab_size", None) \
            or getattr(args, "vocab_size", 32000)
        compute_payload = {"chunks": {}}
        for chunk_idx, _chunk in enumerate(chunks):
            cell = affine_profiler.profile_compute_affine(
                chunks=chunks,
                chunk_idx=chunk_idx,
                is_first_stage=is_first,
                is_last_stage=is_last,
                seq_length=adjusted_seq_length,
                hidden_size=config.hidden_size,
                vocab_size=int(vocab_size),
                dtype=dtype,
                sizes=sizes,
                warmup=warmup,
                measure=measure,
            )
            if cell is not None:
                compute_payload["chunks"][str(chunk_idx)] = cell

        # --- Comm sweep (per directed adjacent edge) -------------------------
        comm_payload = None
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            comm_payload = affine_profiler.profile_comm_affine(
                scheduler=self,
                seq_length=adjusted_seq_length,
                hidden_size=config.hidden_size,
                dtype=dtype,
                sizes=sizes,
                warmup=warmup,
                measure=measure,
            )

        # --- Naive single-point reference for plotting comparison -----------
        # The current LP path scales T_F/T_B/T_W at one mbs (=micro_batch_size,
        # unless the profile iter saw a different f). We expose that line on
        # the plot for direct comparison with the affine fit.
        naive_refs: Dict[Tuple[int, str], Tuple[int, float]] = {}
        comp_dict = self.exp_manager.cdc_compute_profile_dict
        for (mb_id, chunk_id, task_type), payload in comp_dict.items():
            if task_type not in ("F", "B"):
                continue
            t_measured = payload[0] if isinstance(payload, (list, tuple)) else payload
            naive_refs.setdefault((int(chunk_id), task_type), (int(micro_batch_size), float(t_measured)))

        # --- Persist + plot --------------------------------------------------
        affine_profiler.write_results_and_plots(
            profile_result_path=self.exp_manager.profile_result_path,
            pp_rank=self.pp_rank,
            pp_size=parallel_state.get_pipeline_model_parallel_world_size(),
            compute=compute_payload if compute_payload["chunks"] else None,
            comm=comm_payload,
            naive_compute_refs=naive_refs,
        )

        dist.barrier()
        self.cdc_print(
            f"[affine] sweep done in {time.perf_counter()-t0:.1f}s on rank {self.pp_rank}",
            rank=0,
        )

    def pp_benchmark(self):
        """
        Return:
            alpha_to_next: [pp_size]
            alpha_to_prev: [pp_size]
            beta_to_next: [pp_size]
            beta_to_prev: [pp_size]
        """
        my_rank = parallel_state.get_pipeline_model_parallel_rank()
        prev_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
        next_rank = parallel_state.get_pipeline_model_parallel_next_rank()
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        # DEBUG/E1: bypass even-pp assertion so we can test pp=3
        # assert pp_size % 2 == 0

        warmup = 2
        num_iters = 10
        message_size_bytes = 2**30

        tensor_small = torch.ones(1, dtype=torch.float32).cuda(
            torch.cuda.current_device()
        )
        tensor_large = torch.ones(message_size_bytes // 4, dtype=torch.float32).cuda(
            torch.cuda.current_device()
        )

        send_next_group = parallel_state.get_pipeline_extra_send_next_group()
        recv_next_group = parallel_state.get_pipeline_extra_recv_next_group()
        send_prev_group = parallel_state.get_pipeline_extra_send_prev_group()
        recv_prev_group = parallel_state.get_pipeline_extra_recv_prev_group()

        torch.cuda.synchronize()
        for _ in range(warmup):
            if my_rank % 2 == 0:
                self.send(tensor_large, next_rank, group=send_next_group)
                self.recv(tensor_large, next_rank, group=recv_next_group)
                self.send(tensor_large, prev_rank, group=send_prev_group)
                self.recv(tensor_large, prev_rank, group=recv_prev_group)
            else:
                self.recv(tensor_large, prev_rank, group=recv_prev_group)
                self.send(tensor_large, prev_rank, group=send_prev_group)
                self.recv(tensor_large, next_rank, group=recv_next_group)
                self.send(tensor_large, next_rank, group=send_next_group)
        torch.cuda.synchronize()

        # [prev, next] x [small, large]
        t_recv_prev = [0.0, 0.0]
        t_recv_next = [0.0, 0.0]
        for idx, tensor in enumerate([tensor_small, tensor_large]):
            for _ in range(num_iters):
                dist.barrier(group=pp_group)
                torch.cuda.synchronize()
                start = time.perf_counter()
                if my_rank % 2 == 0:
                    self.send(tensor, next_rank, group=send_next_group)
                else:
                    self.recv(tensor, prev_rank, group=recv_prev_group)
                torch.cuda.synchronize()
                end = time.perf_counter()
                if my_rank % 2 != 0:
                    t_recv_prev[idx] += (end - start) / num_iters

        for idx, tensor in enumerate([tensor_small, tensor_large]):
            for _ in range(num_iters):
                dist.barrier(group=pp_group)
                torch.cuda.synchronize()
                start = time.perf_counter()
                if my_rank % 2 == 0:
                    self.send(tensor, prev_rank, group=send_prev_group)
                else:
                    self.recv(tensor, next_rank, group=recv_next_group)
                torch.cuda.synchronize()
                end = time.perf_counter()
                if my_rank % 2 != 0:
                    t_recv_next[idx] += (end - start) / num_iters

        for idx, tensor in enumerate([tensor_small, tensor_large]):
            for _ in range(num_iters):
                dist.barrier(group=pp_group)
                torch.cuda.synchronize()
                start = time.perf_counter()
                if my_rank % 2 == 0:
                    self.recv(tensor, next_rank, group=recv_next_group)
                else:
                    self.send(tensor, prev_rank, group=send_prev_group)
                torch.cuda.synchronize()
                end = time.perf_counter()
                if my_rank % 2 == 0:
                    t_recv_next[idx] += (end - start) / num_iters

        for idx, tensor in enumerate([tensor_small, tensor_large]):
            for _ in range(num_iters):
                dist.barrier(group=pp_group)
                torch.cuda.synchronize()
                start = time.perf_counter()
                if my_rank % 2 == 0:
                    self.recv(tensor, prev_rank, group=recv_prev_group)
                else:
                    self.send(tensor, next_rank, group=send_next_group)
                torch.cuda.synchronize()
                end = time.perf_counter()
                if my_rank % 2 == 0:
                    t_recv_prev[idx] += (end - start) / num_iters

        # idx -> idx + 1
        alpha_to_next = torch.zeros(
            pp_size, dtype=torch.float32, device=torch.cuda.current_device()
        )
        beta_to_next = torch.zeros(
            pp_size, dtype=torch.float32, device=torch.cuda.current_device()
        )
        # idx -> idx - 1
        alpha_to_prev = torch.zeros(
            pp_size, dtype=torch.float32, device=torch.cuda.current_device()
        )
        beta_to_prev = torch.zeros(
            pp_size, dtype=torch.float32, device=torch.cuda.current_device()
        )

        # IMPORTANT: only collect on receiver, since latency is only injected on receiver.
        alpha_to_next[(my_rank - 1) % pp_size] = t_recv_prev[0]
        alpha_to_prev[(my_rank + 1) % pp_size] = t_recv_next[0]

        beta_to_next[(my_rank - 1) % pp_size] = (
            t_recv_prev[1] - t_recv_prev[0]
        ) / message_size_bytes
        beta_to_prev[(my_rank + 1) % pp_size] = (
            t_recv_next[1] - t_recv_next[0]
        ) / message_size_bytes

        dist.all_reduce(alpha_to_next, op=dist.ReduceOp.SUM, group=pp_group)
        dist.all_reduce(beta_to_next, op=dist.ReduceOp.SUM, group=pp_group)
        dist.all_reduce(alpha_to_prev, op=dist.ReduceOp.SUM, group=pp_group)
        dist.all_reduce(beta_to_prev, op=dist.ReduceOp.SUM, group=pp_group)

        # average globally
        dist.all_reduce(alpha_to_next, op=dist.ReduceOp.AVG)
        dist.all_reduce(beta_to_next, op=dist.ReduceOp.AVG)
        dist.all_reduce(alpha_to_prev, op=dist.ReduceOp.AVG)
        dist.all_reduce(beta_to_prev, op=dist.ReduceOp.AVG)

        return (
            alpha_to_next.tolist(),
            alpha_to_prev.tolist(),
            beta_to_next.tolist(),
            beta_to_prev.tolist(),
        )
    
    def dp_benchmark(self, chunk_params):
        """
        chunk_params: chunk_id -> num_params
        
        Return:
            T_DP: (num_chunks, pp_size, 2) 3d list
        """
        dp_group = parallel_state.get_data_parallel_group()
        dp_size = parallel_state.get_data_parallel_world_size()
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        if dp_size == 1:
            return [[[0, 0] for _ in range(pp_size)] for _ in range(self.pp_schedule.sys_config.num_chunks)]
        num_chunks = self.pp_schedule.sys_config.num_chunks
        num_iters = 10
        
        T_DP_list = torch.zeros((num_chunks, pp_size, 2), dtype=torch.float32, device=torch.cuda.current_device())
        
        assert self.args.bf16
        assert self.args.use_distributed_optimizer and self.args.overlap_param_gather and self.args.overlap_grad_reduce
        assert len(chunk_params) == num_chunks
        for chunk_id in range(num_chunks):
            chunk_param = chunk_params[chunk_id] // dp_size * dp_size
            param_dtype_size = 2 # bf16
            grad_dtype_size = 4 if self.args.accumulate_allreduce_grads_in_fp32 else 2 # fp32 or bf16, see distributed_data_parallel.py
            param_size = param_dtype_size * chunk_param
            grad_size = grad_dtype_size * chunk_param
            
            threshold = 4 * 2**30 # 4GB

            # param
            test_size = threshold if param_size > threshold else param_size
            temp_tensor = torch.randn(test_size // 2, dtype=torch.bfloat16, device=torch.cuda.current_device())
            temp_tensor_shard = torch.randn(test_size // 2 // dp_size, dtype=torch.bfloat16, device=torch.cuda.current_device())
            test_times = []
            for _ in range(num_iters):
                dist.barrier(group=dp_group)
                torch.cuda.synchronize()
                start = time.perf_counter()
                dist.all_gather_into_tensor(temp_tensor, temp_tensor_shard, group=dp_group)
                torch.cuda.synchronize()
                end = time.perf_counter()
                test_times.append(end - start)
            T_DP_list[chunk_id][pp_rank][0] = np.median(test_times) * (param_size / test_size)
            
            # grad
            test_size = threshold if grad_size > threshold else grad_size
            temp_tensor = torch.randn(test_size // 2, dtype=torch.bfloat16, device=torch.cuda.current_device())
            temp_tensor_shard = torch.randn(test_size // 2 // dp_size, dtype=torch.bfloat16, device=torch.cuda.current_device())
            test_times = []
            for _ in range(num_iters):
                dist.barrier(group=dp_group)
                torch.cuda.synchronize()
                start = time.perf_counter()
                dist.reduce_scatter_tensor(temp_tensor_shard, temp_tensor, group=dp_group)
                torch.cuda.synchronize()
                end = time.perf_counter()
                test_times.append(end - start)
            T_DP_list[chunk_id][pp_rank][1] = np.median(test_times) * (grad_size / test_size)
        
        dist.all_reduce(T_DP_list, op=dist.ReduceOp.AVG, group=dp_group)
        dist.all_reduce(T_DP_list, op=dist.ReduceOp.SUM, group=pp_group)
        
        return T_DP_list.cpu().numpy().tolist()
        

def tolist_if_needed(x):
    return x.tolist() if hasattr(x, 'tolist') else x
