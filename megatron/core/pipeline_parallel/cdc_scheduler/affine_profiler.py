"""Affine profiler for CrossPipe / dynamic_mb scheduling.

Measures forward / backward (and wgrad if separable) compute time per
(chunk, device) and P2P send/recv time per directed adjacent edge at
multiple microbatch sizes, fits an affine model

    t(mbs) = slope * mbs + intercept

and writes the fits + measurement scatter + visualization plots under
``{profile_result_path}/affine_profile.json`` and
``{profile_result_path}/affine_plots/``.

Activated by ``--cdc_profile_affine``. Coexists with the existing
single-point timing path that feeds the dynamic_mb LP today; the LP
picks up the affine fits when the flag is set, otherwise the original
``T_F[chunk][dev] / f_profiled`` scaling is used.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist


# Fall back gracefully if matplotlib is unavailable in the env.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def default_size_grid(N: int) -> List[int]:
    """Powers of 2 up to N (per-DP global batch size), plus 3 and 6 as
    non-power-of-2 validation points if they fit. Always at least [1, N]."""
    powers = [s for s in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if s <= N]
    extras = [s for s in (3, 6) if s <= N]
    if N not in powers:
        powers.append(N)
    return sorted(set(powers + extras))


def fit_affine(sizes, times) -> Dict:
    """Least-squares affine fit ``t = slope * mbs + intercept``.

    Clamps intercept to >= 0 — a negative intercept would let the LP think
    small microbatches are ~free, which is wrong. Returns a dict with the
    fit + diagnostics + the underlying scatter points (for plotting and
    later inspection)."""
    sizes_arr = np.asarray(sizes, dtype=np.float64)
    times_arr = np.asarray(times, dtype=np.float64)
    n = len(sizes_arr)
    if n == 0:
        return {
            "slope": 0.0,
            "intercept": 0.0,
            "r2": float("nan"),
            "residual_std": 0.0,
            "max_rel_err": 0.0,
            "n_points": 0,
            "points": [],
        }
    if n == 1:
        return {
            "slope": float(times_arr[0] / max(sizes_arr[0], 1.0)),
            "intercept": 0.0,
            "r2": float("nan"),
            "residual_std": 0.0,
            "max_rel_err": 0.0,
            "n_points": 1,
            "points": [(int(sizes_arr[0]), float(times_arr[0]))],
        }
    slope, intercept = np.polyfit(sizes_arr, times_arr, deg=1)
    intercept = max(0.0, float(intercept))
    pred = slope * sizes_arr + intercept
    ss_res = float(np.sum((times_arr - pred) ** 2))
    ss_tot = float(np.sum((times_arr - times_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    residual_std = float(np.std(times_arr - pred))
    rel_err = np.abs(times_arr - pred) / np.maximum(times_arr, 1e-12)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "residual_std": residual_std,
        "max_rel_err": float(np.max(rel_err)),
        "n_points": int(n),
        "points": [(int(s), float(t)) for s, t in zip(sizes_arr, times_arr)],
    }


def _time_cuda_event(fn, warmup: int, measure: int) -> float:
    """Run ``fn()`` ``warmup`` times then ``measure`` times under CUDA events.
    Returns the median elapsed time in seconds."""
    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    times_ms: List[float] = []
    for _ in range(measure):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        times_ms.append(start.elapsed_time(end))
    return float(np.median(times_ms)) / 1000.0


# ---------------------------------------------------------------------------
# Communication profiling
# ---------------------------------------------------------------------------


def _resolve_comm_groups(scheduler) -> Tuple[Optional[object], Optional[object], Optional[object], Optional[object]]:
    """Return (send_next, recv_next, send_prev, recv_prev) groups, preferring
    the dedicated 2-rank groups created for dynamic_mb when available."""
    if getattr(scheduler, "_use_dedicated_p2p_groups", False):
        return (
            scheduler._p2p_send_next_group,
            scheduler._p2p_recv_next_group,
            scheduler._p2p_send_prev_group,
            scheduler._p2p_recv_prev_group,
        )
    from megatron.core import parallel_state

    return (
        parallel_state.get_pipeline_extra_send_next_group(),
        parallel_state.get_pipeline_extra_recv_next_group(),
        parallel_state.get_pipeline_extra_send_prev_group(),
        parallel_state.get_pipeline_extra_recv_prev_group(),
    )


def profile_comm_affine(
    scheduler,
    seq_length: int,
    hidden_size: int,
    dtype: torch.dtype,
    sizes: List[int],
    warmup: int,
    measure: int,
) -> Dict:
    """For each adjacent PP pair (i, i+1), measure:

      - forward send/recv time at activation tensor shape (seq, mbs, hidden)
      - backward send/recv time at gradient tensor shape (same)

    for each mbs in ``sizes``. Returns a dict structured for
    ``affine_profile.json``: per directed edge, per direction, the points and
    fit. The intercept of the fit is the analogue of ``T_alpha`` (latency)
    today, and the slope replaces ``T_bw`` (per-byte bandwidth) scaled by
    ``seq * hidden * dtype_bytes`` to give per-sample comm cost."""
    from megatron.core import parallel_state

    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    my_rank = parallel_state.get_pipeline_model_parallel_rank()
    prev_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
    next_rank = parallel_state.get_pipeline_model_parallel_next_rank()
    pp_group = parallel_state.get_pipeline_model_parallel_group()

    send_next_g, recv_next_g, send_prev_g, recv_prev_g = _resolve_comm_groups(scheduler)
    device = torch.cuda.current_device()

    # For each directed edge index i in [0, pp_size-1), measure time on the
    # receiver. We use a global synchronisation pattern: even ranks send,
    # odd ranks receive (and vice versa) so every directed edge is timed.
    fwd_per_edge: Dict[int, Dict] = {}
    bwd_per_edge: Dict[int, Dict] = {}

    for mbs in sizes:
        shape = (seq_length, mbs, hidden_size)
        tensor = torch.zeros(shape, dtype=dtype, device=device)

        # Forward direction: rank i sends to rank i+1.
        # For each i, even-i pairs and odd-i pairs run in different rounds to
        # avoid one rank's send and recv aliasing on the same group at once.
        for parity in (0, 1):
            for i in range(parity, pp_size - 1, 2):
                if my_rank == i:
                    fn = lambda: scheduler.send(tensor, next_rank, group=send_next_g)
                elif my_rank == i + 1:
                    fn = lambda: scheduler.recv(tensor, prev_rank, group=recv_prev_g)
                else:
                    fn = None

                if fn is not None:
                    t = _time_cuda_event(fn, warmup, measure)
                else:
                    t = 0.0

                # The receiver's measurement is the authoritative one.
                t_t = torch.tensor([t], dtype=torch.float64, device=device)
                if my_rank != i + 1:
                    t_t.zero_()
                dist.all_reduce(t_t, op=dist.ReduceOp.SUM, group=pp_group)
                fwd_per_edge.setdefault(i, {"sizes": [], "times": []})
                fwd_per_edge[i]["sizes"].append(mbs)
                fwd_per_edge[i]["times"].append(float(t_t.item()))

        # Backward direction: rank i+1 sends to rank i.
        for parity in (0, 1):
            for i in range(parity, pp_size - 1, 2):
                if my_rank == i + 1:
                    fn = lambda: scheduler.send(tensor, prev_rank, group=send_prev_g)
                elif my_rank == i:
                    fn = lambda: scheduler.recv(tensor, next_rank, group=recv_next_g)
                else:
                    fn = None

                if fn is not None:
                    t = _time_cuda_event(fn, warmup, measure)
                else:
                    t = 0.0

                t_t = torch.tensor([t], dtype=torch.float64, device=device)
                if my_rank != i:
                    t_t.zero_()
                dist.all_reduce(t_t, op=dist.ReduceOp.SUM, group=pp_group)
                bwd_per_edge.setdefault(i, {"sizes": [], "times": []})
                bwd_per_edge[i]["sizes"].append(mbs)
                bwd_per_edge[i]["times"].append(float(t_t.item()))

        del tensor
        torch.cuda.empty_cache()

    # Fit per edge.
    for d in (fwd_per_edge, bwd_per_edge):
        for i, payload in d.items():
            payload["fit"] = fit_affine(payload["sizes"], payload["times"])

    return {
        "seq_length": int(seq_length),
        "hidden_size": int(hidden_size),
        "dtype_bytes": torch.tensor([], dtype=dtype).element_size(),
        "fwd_per_edge": {str(k): v for k, v in fwd_per_edge.items()},
        "bwd_per_edge": {str(k): v for k, v in bwd_per_edge.items()},
    }


# ---------------------------------------------------------------------------
# Compute profiling
# ---------------------------------------------------------------------------


def _unwrap(m):
    """Strip DDP / FP16 / FSDP wrappers to reach the underlying model."""
    while hasattr(m, "module"):
        m = m.module
    return m


def profile_compute_affine(
    chunks,
    chunk_idx: int,
    is_first_stage: bool,
    is_last_stage: bool,
    seq_length: int,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    sizes: List[int],
    warmup: int,
    measure: int,
) -> Optional[Dict]:
    """Time forward and backward at each mbs in ``sizes`` for one model chunk.

    Returns ``{"F": cell, "B": cell, "W": None}`` where each cell has
    ``sizes``, ``times``, and ``fit``. Returns ``None`` only on hard error.

    **Backward timing without double-counting:** PyTorch consumes the
    autograd graph on ``.backward()``, so each timed backward iteration
    must rebuild the forward. The previous version timed the combined
    fwd+bwd region and reported it as B, which made the LP see B
    overcounted by exactly the forward time. We now time forward-only
    (``t_f``) and forward-then-backward (``t_total``) separately, then
    report ``t_b = max(0, t_total - t_f)``. The subtraction cancels the
    forward cost; the ``max(0, ...)`` clamps the rare case where
    measurement noise dominates a near-zero backward.

    **First-stage coverage:** earlier this function returned ``None`` for
    the first PP stage (it consumes tokens, not activations). Now we
    construct synthetic random token IDs of shape ``(mbs, seq)`` using
    ``vocab_size`` and run the model directly through its embedding
    layer, so the first stage gets the same affine treatment as middle
    stages. Non-first stages still use synthetic hidden-state activations
    plumbed via ``set_input_tensor`` so the model bypasses the embedding
    lookup, exactly like a real intermediate forward step does.

    **Last-stage backward:** the loss function isn't available here, so
    we use ``out.float().sum()`` as a stand-in loss. The kernel cost of
    a sum-reduction is small relative to the LM-head matmul, so the
    measurement is dominated by the same matmuls the real backward
    would run."""
    model = chunks[chunk_idx]
    device = torch.cuda.current_device()

    # Disable any DDP / FSDP grad-sync hooks for the synthetic backward.
    no_sync_ctx = None
    if hasattr(model, "no_sync"):
        try:
            no_sync_ctx = model.no_sync()
            no_sync_ctx.__enter__()
        except Exception:
            no_sync_ctx = None

    forward_records: Dict[int, float] = {}
    backward_records: Dict[int, float] = {}

    _set_input_tensor = getattr(_unwrap(model), "set_input_tensor", None)

    try:
        for mbs in sizes:
            # Megatron's batch dim is dim 0 on tokens / position_ids;
            # the activation tensor between stages is (seq, mbs, hidden).
            tokens = torch.randint(
                low=0, high=max(2, int(vocab_size)),
                size=(mbs, seq_length),
                dtype=torch.long, device=device,
            )
            position_ids = (
                torch.arange(0, seq_length, dtype=torch.long, device=device)
                .unsqueeze(0).expand(mbs, -1)
            )
            attention_mask = (
                torch.ones((1, 1, seq_length, seq_length), dtype=torch.bool, device=device)
                .tril().logical_not()
            )

            # Activation input for non-first stages. None for the first stage
            # (model's embedding lookup uses the synthetic tokens instead).
            if is_first_stage:
                inp = None
            else:
                inp = torch.randn(
                    (seq_length, mbs, hidden_size),
                    dtype=dtype, device=device, requires_grad=True,
                )

            # Upstream gradient for non-last stages — same shape as the
            # model's hidden-state output. Last stage uses sum() backward.
            grad_out = None
            if not is_last_stage:
                grad_out = torch.randn(
                    (seq_length, mbs, hidden_size), dtype=dtype, device=device
                )

            def _call_model():
                if inp is not None and _set_input_tensor is not None:
                    # Route the synthetic activation around the embedding.
                    _set_input_tensor(inp)
                return model(tokens, position_ids, attention_mask)

            def _zero_state():
                if inp is not None and inp.grad is not None:
                    inp.grad = None
                model.zero_grad(set_to_none=True)

            # Forward-only — for the F slope. The detached sum keeps the
            # output alive long enough that the autograd machinery doesn't
            # short-circuit the forward.
            def fwd_fn():
                _zero_state()
                with torch.enable_grad():
                    out = _call_model()
                if isinstance(out, tuple):
                    out = out[0]
                _ = out.sum().detach()

            t_f = _time_cuda_event(fwd_fn, warmup, measure)
            forward_records[mbs] = t_f

            # Forward + backward — subtract t_f below to recover pure B.
            def fwd_bwd_fn():
                _zero_state()
                with torch.enable_grad():
                    out = _call_model()
                if isinstance(out, tuple):
                    out = out[0]
                if is_last_stage or grad_out is None or out.shape != grad_out.shape:
                    out.float().sum().backward()
                else:
                    out.backward(grad_out)

            t_total = _time_cuda_event(fwd_bwd_fn, warmup, measure)
            # Pure backward = combined minus forward. Clamp to 0 in the
            # rare case where measurement noise dominates a near-zero B.
            t_b = max(0.0, t_total - t_f)
            backward_records[mbs] = t_b

            # Cleanup before the next size.
            model.zero_grad(set_to_none=True)
            del tokens, position_ids, attention_mask
            if inp is not None:
                del inp
            if grad_out is not None:
                del grad_out
            torch.cuda.empty_cache()
    finally:
        if no_sync_ctx is not None:
            try:
                no_sync_ctx.__exit__(None, None, None)
            except Exception:
                pass

    fwd_sizes = sorted(forward_records.keys())
    bwd_sizes = sorted(backward_records.keys())

    return {
        "F": {
            "sizes": fwd_sizes,
            "times": [forward_records[s] for s in fwd_sizes],
            "fit": fit_affine(fwd_sizes, [forward_records[s] for s in fwd_sizes]),
        },
        "B": {
            "sizes": bwd_sizes,
            "times": [backward_records[s] for s in bwd_sizes],
            "fit": fit_affine(bwd_sizes, [backward_records[s] for s in bwd_sizes]),
        },
        # W is not separately profiled. When wgrad_split is on, the
        # weight-grad GEMM is split out of the backward into a separate
        # W task at runtime — but our synthetic backward runs the
        # combined kernel, so B's slope/intercept already includes the
        # wgrad work. The LP gets W = 0 from the fallback in that case.
        "W": None,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _naive_scale_line(
    sizes: List[int], measured_at_one_size: Tuple[int, float]
) -> Tuple[List[int], List[float]]:
    """The line currently used by the LP path: ``t(mbs) = (t_ref / s_ref) * mbs``.
    Plotted alongside the regression for direct comparison."""
    s_ref, t_ref = measured_at_one_size
    if s_ref <= 0:
        return list(sizes), [0.0 for _ in sizes]
    rate = t_ref / s_ref
    return list(sizes), [rate * s for s in sizes]


def plot_affine_fit(
    title: str,
    sizes: List[int],
    times: List[float],
    fit: Dict,
    out_path: str,
    naive_ref: Optional[Tuple[int, float]] = None,
) -> None:
    if not _HAS_MPL or len(sizes) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(sizes, [t * 1000.0 for t in times], color="tab:blue", label="measured")
    xs = np.linspace(0, max(sizes) * 1.05, 50)
    ys = fit["slope"] * xs + fit["intercept"]
    ax.plot(
        xs,
        ys * 1000.0,
        color="tab:orange",
        label=f"affine fit (slope={fit['slope']*1e6:.1f} us/sample, b={fit['intercept']*1e3:.2f} ms, R²={fit['r2']:.3f})",
    )
    if naive_ref is not None:
        nx, ny = _naive_scale_line(sorted(set(sizes)), naive_ref)
        ax.plot(
            nx,
            [y * 1000.0 for y in ny],
            color="tab:red",
            linestyle="--",
            label=f"naive (linear thru mbs={naive_ref[0]})",
        )
    ax.set_xlabel("microbatch size")
    ax.set_ylabel("time (ms)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _provenance_meta() -> Dict:
    """Reproducibility footprint for the affine profile run. Saved alongside
    the fit data so a later run on a different cluster can tell whether the
    cached numbers are still relevant."""
    meta = {"timestamp": time.time()}
    try:
        meta["torch_version"] = torch.__version__
    except Exception:
        pass
    try:
        meta["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    try:
        meta["nccl_version"] = ".".join(str(x) for x in torch.cuda.nccl.version())
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            meta["gpu_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        pass
    return meta


def write_results_and_plots(
    profile_result_path: str,
    pp_rank: int,
    pp_size: int,
    compute: Optional[Dict],
    comm: Optional[Dict],
    naive_compute_refs: Optional[Dict[Tuple[int, str], Tuple[int, float]]] = None,
) -> None:
    """Each rank writes its compute fits to ``affine_profile_rank{N}.json``;
    rank 0 also writes ``affine_profile_comm.json`` (comm fits are
    pp-group-averaged so a single file suffices). Both files include
    provenance metadata for reproducibility."""
    os.makedirs(profile_result_path, exist_ok=True)
    plot_dir = os.path.join(profile_result_path, "affine_plots")
    os.makedirs(plot_dir, exist_ok=True)
    meta = _provenance_meta()
    if compute is not None:
        compute = dict(compute)
        compute["meta"] = meta
    if comm is not None:
        comm = dict(comm)
        comm["meta"] = meta

    # Rank-local compute fits + plots.
    if compute is not None:
        with open(
            os.path.join(profile_result_path, f"affine_profile_rank{pp_rank}.json"), "w"
        ) as f:
            json.dump(compute, f, indent=2)
        for chunk_id, per_op in compute.get("chunks", {}).items():
            for op in ("F", "B"):
                cell = per_op.get(op)
                if cell is None:
                    continue
                naive_ref = None
                if naive_compute_refs is not None:
                    naive_ref = naive_compute_refs.get((int(chunk_id), op))
                plot_affine_fit(
                    title=f"compute {op} chunk={chunk_id} dev={pp_rank}",
                    sizes=cell["sizes"],
                    times=cell["times"],
                    fit=cell["fit"],
                    out_path=os.path.join(
                        plot_dir, f"compute_{op}_chunk{chunk_id}_dev{pp_rank}.png"
                    ),
                    naive_ref=naive_ref,
                )

    # Comm fits + plots — rank 0 only.
    if comm is not None and pp_rank == 0:
        with open(os.path.join(profile_result_path, "affine_profile_comm.json"), "w") as f:
            json.dump(comm, f, indent=2)
        for direction in ("fwd_per_edge", "bwd_per_edge"):
            for edge_idx, payload in comm.get(direction, {}).items():
                plot_affine_fit(
                    title=f"comm {direction.split('_')[0]} edge={edge_idx}->{int(edge_idx)+1 if direction.startswith('fwd') else int(edge_idx)}",
                    sizes=payload["sizes"],
                    times=payload["times"],
                    fit=payload["fit"],
                    out_path=os.path.join(
                        plot_dir, f"comm_{direction.split('_')[0]}_edge{edge_idx}.png"
                    ),
                )


# ---------------------------------------------------------------------------
# Loading affine fits back into the LP cost path
# ---------------------------------------------------------------------------


def load_affine_compute(profile_result_path: str, pp_size: int) -> Optional[Dict]:
    """Read all per-rank affine compute files. Returns
    ``{(chunk_id, dev): {"F": fit, "B": fit, "W": fit_or_None}}`` or None if
    no files are present (e.g. because the flag wasn't on at profile time)."""
    fits: Dict[Tuple[int, int], Dict] = {}
    found = False
    for r in range(pp_size):
        path = os.path.join(profile_result_path, f"affine_profile_rank{r}.json")
        if not os.path.exists(path):
            continue
        found = True
        with open(path, "r") as f:
            payload = json.load(f)
        for chunk_id_str, per_op in payload.get("chunks", {}).items():
            fits[(int(chunk_id_str), r)] = per_op
    return fits if found else None


def load_affine_comm(profile_result_path: str) -> Optional[Dict]:
    path = os.path.join(profile_result_path, "affine_profile_comm.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)
