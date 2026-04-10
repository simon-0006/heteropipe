import contextlib
import json
import os
import pickle
import random
import time
from typing import Dict, Iterator, List, Optional, Tuple, Union
from datetime import timedelta

import numpy as np

from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.pipeline import (
    CPZBUDPipeline,
    CPZBWavePipeline,
)
from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.subpipeline import DynZBUDSubPipeline
from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.pipeline_config import (
    SystemConfig,
)
from megatron.core.pipeline_parallel.cdc_scheduler.wgrad_store import WGradStore
from megatron.core.pipeline_parallel.cdc_scheduler.experiment_manager import (
    ExperimentManager,
)
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


def get_or_set_pp_io_tensor(tensor_dict: Dict, key, config, tensor_shape):
    return tensor_dict.setdefault(
        key,
        torch.empty(
            tensor_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        ),
    )


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
        ], "Currently only support ud, subud and wave schedule"
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

        # barrier
        dist.barrier()
        self.pipeline = pp
        return estimated_runtime

    def get_schedule(self) -> Pipeline:
        estimated_runtime = self.generate_schedule_from_profile()
        return self.pipeline, estimated_runtime

    def update_latency_bandwidth_seconds(
        self, latency_seconds=None, bandwidth_seconds=None
    ):
        self.override_T_comm(latency_seconds, bandwidth_seconds)


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
                "ZBH1" if self.dynamic_schedule_type in ["ud", "subud"] else "ZBV",
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

        # grad sync
        self.no_sync_func = None
        self.no_sync_context = None

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
        self.cdc_print(
            f"Delay Config Update: latency {latency_as_F_stage} F stage, {latency_sec} seconds; bandwidth {bandwidth_as_F_stage} F stage, {bandwidth_sec} seconds",
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
            self.pp_schedule, estimated_runtime = self.pp_schedule_generator.get_schedule()
            
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

        if self.wgrad_split:
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

    def schedule_comm_event(self, event: CommEvent, config, tensor_shape):
        send_next_group = parallel_state.get_pipeline_extra_send_next_group()
        recv_next_group = parallel_state.get_pipeline_extra_recv_next_group()
        send_prev_group = parallel_state.get_pipeline_extra_send_prev_group()
        recv_prev_group = parallel_state.get_pipeline_extra_recv_prev_group()

        next_rank = parallel_state.get_pipeline_model_parallel_next_rank()
        prev_rank = parallel_state.get_pipeline_model_parallel_prev_rank()

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
                self.isend(
                    send_buffer,
                    next_rank,
                    group=send_next_group,
                    bandwidth_delay_ms=self.injected_bandwidth_delay[1] * 1000 if self.cdc_send_next else 0,
                )
            )
        elif event.type == CommEventType.POST_RECV_NEXT:
            self.recv_next_reqs[(event.mb_id, event.chunk_id, event.task_type)] = (
                self.irecv(
                    recv_buffer,
                    next_rank,
                    group=recv_next_group,
                    bandwidth_delay_ms=self.injected_bandwidth_delay[1] * 1000 if self.cdc_recv_next else 0,
                )
            )
        elif event.type == CommEventType.POST_SEND_PREV:
            self.send_prev_reqs[(event.mb_id, event.chunk_id, event.task_type)] = (
                self.isend(
                    send_buffer,
                    prev_rank,
                    group=send_prev_group,
                    bandwidth_delay_ms=self.injected_bandwidth_delay[1] * 1000 if self.cdc_send_prev else 0,
                )
            )
        elif event.type == CommEventType.POST_RECV_PREV:
            self.recv_prev_reqs[(event.mb_id, event.chunk_id, event.task_type)] = (
                self.irecv(
                    recv_buffer,
                    prev_rank,
                    group=recv_prev_group,
                    bandwidth_delay_ms=self.injected_bandwidth_delay[1] * 1000 if self.cdc_recv_prev else 0,
                )
            )
        elif event.type == CommEventType.WAIT_SEND_NEXT:
            handle = self.send_next_reqs[(event.mb_id, event.chunk_id, event.task_type)]
            assert handle is not None
            handle.wait()
        elif event.type == CommEventType.WAIT_RECV_NEXT:
            handle = self.recv_next_reqs[(event.mb_id, event.chunk_id, event.task_type)]
            assert handle is not None
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
            self.schedule_comm_event(event, config, tensor_shape)
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

        tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
        tensor_shape[0] = (
            tensor_shape[0] // parallel_state.get_context_parallel_world_size()
        )
        if config.sequence_parallel:
            tensor_shape[0] = (
                tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()
            )

        # multiply tensor shape dims
        self.pp_comm_size_bytes = (
            tensor_shape[0]
            * tensor_shape[1]
            * tensor_shape[2]
            * torch.tensor([], dtype=config.pipeline_dtype).element_size()
        )

        self.update_schedule_with_latency_bandwidth()

        for idx, compute_task in enumerate(self.pp_execution_plan_cur_device):
            # self.cdc_print(f"compute_task: {compute_task}")
            self.exp_manager.exp_logging_first_mb = True if idx == 0 else False
            self.schedule_compute_task(
                compute_task=compute_task,
                model=model,
                data_iterator=data_iterator,
                forward_step_func=forward_step_func,
                tensor_shape=tensor_shape,
                forward_data_store=forward_data_store,
                collect_non_loss_data=collect_non_loss_data,
                first_val_step=first_val_step,
                forward_only=forward_only,
                num_microbatches=num_microbatches,
            )
            self.deallocate_tensor_in_dicts()

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
        assert pp_size % 2 == 0

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
