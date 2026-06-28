# HeteroPipe design notes

This document is the stable reference for the components HeteroPipe adds on top
of CrossPipe. It replaces the scattered in-code rationale so the source comments
can stay short.

## 1. Dynamic microbatch-size MILP

CrossPipe fixes every microbatch to the same size and optimizes only the
operation order. HeteroPipe makes the per-microbatch sizes integer decision
variables and optimizes ordering and sizing jointly on the same
communication-aware schedule graph, for one-chunk zero-bubble schedules.

- Implemented in
  `megatron/core/pipeline_parallel/cdc_scheduler/pp_generator/auto_schedule.py`
  (graph and solve) and driven from `pp_scheduler.py` (`dynamic_mb` path).
- The solver is a general-purpose branch-and-bound MILP. PuLP/CBC is the default;
  `gurobipy` is used automatically when installed and falls back to CBC
  otherwise. The CPLEX-backed CrossPipe schedulers (`auto_cp_schedule.py`) need
  `docplex`.
- The solve runs once, after a profiling/warm-up phase, on the data-parallel-rank
  process that owns the schedule. It is given a wall-clock time limit (300 s by
  default) and a relative optimality gap; in practice the wall-clock limit, not
  the gap, ends the solve.
- Integer sizes are post-processed so that `sum(f_i) == N` (the per-DP global
  batch), regardless of any rounding the solver applies.

### Shape penalty

The cost model is blind to per-shape runtime overhead (cuBLAS kernel selection,
NCCL stream multiplexing, Python dispatch). `--cdc_dynamic_mb_shape_penalty_us`
adds a penalty (in microseconds) per *distinct* microbatch shape to the
objective, regularizing the solver away from fragmenting the batch into many
different sizes. `--cdc_dynamic_mb_max_f_cap` bounds the largest microbatch
(default `uniform_size + 2`).

### Why the schedule is solved at zero assumed latency

`--cdc_dynamic_mb_schedule_lat_ms` decouples the latency the MILP *assumes* when
choosing sizes from the latency the runtime actually injects
(`--cdc_stock_inject_latency_ms`). With a fixed microbatch count the injected
cross-boundary latency is paid on a fixed number of crossings — a near-constant
offset the schedule cannot reduce. Feeding it into the MILP biases the
compute-balance objective toward schedules that are slower at base compute.
Solving on compute alone (`--cdc_dynamic_mb_schedule_lat_ms 0`) yields a
compute-balanced schedule that overlaps the latency naturally. This matches the
thesis finding that folding the link delay into the solve did not help.

## 2. Affine multi-size profiler

Enabled with `--cdc_profile_affine`. For each stage `s` and operation type
(forward, backward, and weight-gradient when separable) and each adjacent
directed link, the profiler times the operation at several microbatch sizes and
fits

    t(mbs) = slope · mbs + intercept

by least squares (intercept clamped to `>= 0`). The fixed `intercept` is what a
proportional, through-the-origin model discards; recovering it is what keeps the
joint optimization honest at small microbatch sizes (otherwise a one-sample
microbatch looks almost free and the solver fragments the batch).

- Each `(size, op)` cell is measured with CUDA events: a few untimed warmups
  (to absorb the one-time cuBLAS shape-cache cost) followed by several timed runs
  whose median is retained.
- Backward is reported as `max(0, t_{F+B} − t_F)`.
- A fit is admitted into the LP cost model only if its `R²` clears
  `--cdc_profile_affine_r2_threshold` (default 0.7); cells that fail keep the
  single-point `T_op / f_profiled` fallback. Enabling the profiler therefore
  never removes information.
- Output: `{profile_result_path}/affine_profile.json` plus scatter/plot images
  under `{profile_result_path}/affine_plots/` (requires `matplotlib`).

Approximations worth stating: the weight-gradient block `W` is not separately
affine-profiled and keeps its canonical cost; the first stage is profiled with
synthetic tokens; the last-stage backward has no loss labels available to the
profiler.

## 3. Variable-size runtime

The runtime replays the solver's fixed per-stage execution order with the fixed
integer size vector. The additions over CrossPipe's fixed-size runtime:

- **Re-slicing data iterator** (`DynamicMicrobatchIterator`): splits the incoming
  equal-size batches into the solver's variable microbatch sizes without padding.
- **True per-event tensor shapes**: each communication event uses the real
  per-microbatch shape, precomputed once per iteration.
- **Deadlock-free first-seen shapes at `pp ≥ 4`**: the receive is deferred to its
  wait point (just-in-time receive) so no receive kernel sits dangling on the
  NCCL stream during cuBLAS first-init's device-wide synchronization. This
  removed the need for the dedicated 2-rank P2P process groups an earlier version
  used; dynamic schedules now reuse the same NCCL communicators as the static
  ones.
- **Size-correct loss weighting**: each microbatch's loss is scaled by
  `f_i / N`, so the summed batch gradient is unbiased regardless of the size
  split. Uniform sizes `[k, k, ...]` reduce to the static loss scaling.

## 4. Stock-PyTorch latency injection

CrossPipe injects cross-datacenter latency on the receiving side, but only
through a custom PyTorch build. HeteroPipe injects it on the **sending** side
using unmodified PyTorch:

- `--cdc_stock_inject_latency_ms` delays `isend` by spinning with
  `torch.cuda._sleep` on a dedicated send-side CUDA stream, so the delay runs
  concurrently with default-stream compute. The receiver sees the data at the
  actual link arrival time.
- Sender-side is chosen deliberately: putting the spin on the receiver's wait
  path would add the latency *on top of* prior compute, overcounting it and
  destroying the overlap the MILP modeled.
- `--cdc_stock_inject_link` selects which sends are delayed
  (`cross_boundary` matches the `--num_dc` / `--pp_stages_per_dc` layout);
  `--cdc_stock_inject_warmup_iters` skips injection for the first few iterations
  so cuBLAS warms its shape cache before latency starts gating sends.

This makes controlled cross-datacenter experiments possible on a single node.
