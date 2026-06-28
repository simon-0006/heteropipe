# HeteroPipe

HeteroPipe is a research fork of [CrossPipe](https://github.com/spcl/crosspipe)
(itself a fork of NVIDIA [Megatron-LM](https://github.com/NVIDIA/Megatron-LM))
that adds **dynamic microbatch sizing** to communication-aware pipeline-parallel
LLM training. It jointly optimizes *what runs when* (operation ordering) and
*how large each microbatch is* (per-microbatch sizing) as a single
mixed-integer linear program over CrossPipe's schedule graph, using cost
estimates from an **affine, multi-size profiler**. The schedule is solved once
before training and executed by a **variable-size pipeline runtime** built
inside Megatron-LM.

This repository is the implementation accompanying the ETH Zürich Bachelor
thesis *"Joint Operation Ordering and Microbatch Sizing for Cross-Datacenter
Pipeline Parallelism"* by Simon Ganter.

> **Status: research prototype.** The code is validated for the thesis
> experiments (4-stage pipeline, single node, `TP=1`); it is not a drop-in
> general-purpose replacement for Megatron-LM. Expect rough edges outside the
> tested configurations.

## Relation to CrossPipe and Megatron-LM

HeteroPipe keeps CrossPipe's communication-aware scheduler and Megatron-Core's
training stack unchanged, and adds a new code path on top of them:

| Layer | Origin | Role |
| --- | --- | --- |
| Megatron-LM / Megatron-Core | NVIDIA | Transformer models, parallelism, training loop |
| CrossPipe | SPCL (ETH Zürich) | Communication-aware pipeline schedule generation and execution |
| **HeteroPipe** | this fork | Dynamic microbatch-size MILP, affine profiler, variable-size runtime, stock-PyTorch latency injection |

The upstream READMEs are preserved verbatim under
[`docs/upstream_crosspipe_readme.md`](docs/upstream_crosspipe_readme.md) and
[`docs/upstream_megatron_readme.md`](docs/upstream_megatron_readme.md). See
[`NOTICE.md`](NOTICE.md) for attribution and license lineage.

## What HeteroPipe adds

- **Dynamic microbatch-size MILP** for one-chunk zero-bubble schedules:
  generalizes CrossPipe's fixed-size formulation by making the per-microbatch
  sizes integer decision variables, optimized jointly with the operation order.
- **Affine compute/communication profiler** (`--cdc_profile_affine`) that
  measures each operation across a range of microbatch sizes and fits a fixed
  overhead plus a per-sample slope (`t = slope·mbs + intercept`), instead of the
  proportional, through-the-origin cost a fixed-size schedule assumes.
- **Variable-size runtime path**: a re-slicing data iterator, true per-event
  tensor shapes, deadlock-free handling of first-seen shapes at `pp ≥ 4`, and
  size-correct loss weighting (`loss_scale = f_i / N`) that keeps the batch
  gradient unbiased.
- **Stock-PyTorch latency injection** (`--cdc_stock_inject_latency_ms`):
  emulates a slow cross-datacenter link on a single node by delaying sends on a
  dedicated CUDA stream, using **unmodified** PyTorch. CrossPipe's original
  injection API required a custom PyTorch build; the thesis experiments do not.
- CrossPipe's existing schedule execution (ZBH1, ZBV, 1F1B, GPipe, ...) is
  retained and used as the baseline.

The design — the MILP, the affine cost model, the shape penalty, and the runtime
support — is documented in [`docs/heteropipe_design.md`](docs/heteropipe_design.md).

## Repository layout

Only the HeteroPipe-relevant paths are listed; the rest of the tree is upstream
Megatron-LM.

```
megatron/core/pipeline_parallel/cdc_scheduler/
├── pp_scheduler.py            # Execution engine: dynamic_mb schedule + variable-size runtime
├── affine_profiler.py         # Affine multi-size compute/comm profiler
├── experiment_manager.py      # Latency/bandwidth sweep driver for experiments
└── pp_generator/
    ├── auto_schedule.py       # MILP schedule + microbatch-size generation (PuLP)
    ├── auto_cp_schedule.py    # CPLEX-backed CrossPipe schedule generation
    └── svg_event.py           # Schedule visualization

megatron/core/pipeline_parallel/schedules.py   # loss_scale for variable microbatch sizes
megatron/core/tensor_parallel/layers.py        # variable-size weight-gradient reshape
megatron/training/arguments.py                 # --cdc_* / --dynamic_schedule flags
pretrain_gpt.py                                 # GPT training entry point
test_crossdc/exp_slurm/                         # CrossPipe Slurm experiment generators
docs/                                           # design notes, reproduction, upstream READMEs
```

## Environment and dependencies

- **Container**: the NGC PyTorch container `24.10-py3` (the version CrossPipe was
  developed against) is the recommended base. Other recent PyTorch builds work
  but were not validated.
- **PyTorch build**: a **stock/unmodified** PyTorch is sufficient for HeteroPipe.
  Use `--cdc_stock_inject_latency_ms` for latency emulation. CrossPipe's original
  `wait_with_lat_delay_in_ms` injection path needs the
  [custom PyTorch branch](https://github.com/C-TC/pytorch/tree/lat_bw_inject_2410)
  and is *not* required to reproduce the thesis results.
- **Python packages**:
  - Required for the dynamic MILP: `pulp` (CBC solver ships with PuLP), `scipy`,
    `numpy`.
  - Optional: `gurobipy` (faster MILP solver, falls back to CBC if absent),
    `docplex` (only for the CPLEX-backed CrossPipe schedulers), `drawsvg`
    (schedule SVGs), `matplotlib` (affine-fit plots), `psutil` (solver thread
    count), `flash-attn` (`--use-flash-attn`).
- Install Megatron-Core itself as usual (see
  [`docs/upstream_megatron_readme.md`](docs/upstream_megatron_readme.md)).

## Quick start

A minimal single-node run that uses the dynamic-microbatch schedule with 10 ms of
emulated cross-boundary latency. Adjust data, tokenizer, and model size for your
setup.

```bash
torchrun --nproc_per_node 4 pretrain_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 4 \
    --num_dc 2 --pp_stages_per_dc 2 2 \
    --transformer-impl local \
    --num-layers 26 --hidden-size 2048 --ffn-hidden-size 8192 \
    --num-attention-heads 16 --seq-length 2048 --max-position-embeddings 2048 \
    --micro-batch-size 8 --global-batch-size 64 \
    --train-iters 200 --bf16 \
    --data-path  <your_data_prefix> \
    --tokenizer-type Llama2Tokenizer --tokenizer-model <your_tokenizer.model> \
    --enable_cdcpp_scheduler \
    --dynamic_schedule dynamic_mb \
    --cdc_profile_affine \
    --cdc_dynamic_mb_schedule_lat_ms 0 \
    --cdc_stock_inject_latency_ms 10 \
    --cdc_stock_inject_link cross_boundary
```

Swap `--dynamic_schedule dynamic_mb` for `--static_schedule ZBH1` to run the
zero-bubble baseline under the same injected latency.

## Key flags

| Flag | Meaning |
| --- | --- |
| `--enable_cdcpp_scheduler` | Enable the CrossPipe/HeteroPipe scheduler. |
| `--dynamic_schedule dynamic_mb` | Use the dynamic microbatch-size schedule (HeteroPipe). |
| `--static_schedule {ZBH1,ZBV,1F1B,GPipe,Interleaved1F1B}` | Use a fixed-size CrossPipe schedule (baselines). |
| `--cdc_profile_affine` | Profile each op at multiple sizes and fit the affine cost model used by the MILP. |
| `--cdc_dynamic_mb_schedule_lat_ms` | Latency (ms) the MILP *assumes* when sizing, decoupled from what is injected. `0` = solve on compute alone; `<0` = use the real injected latency. |
| `--cdc_dynamic_mb_shape_penalty_us` | Penalty (µs) per distinct microbatch shape; regularizes against per-shape runtime overhead. `0` disables. |
| `--cdc_dynamic_mb_max_f_cap` | Upper bound on per-microbatch size. `0` = `uniform_size + 2`; set to the global batch to remove the cap. |
| `--cdc_stock_inject_latency_ms` | Emulated per-link latency (ms) on stock PyTorch. `0` disables. |
| `--cdc_stock_inject_link {none,cross_boundary,all}` | Which sends get the injected delay. |

The full affine-profiler controls (`--cdc_profile_affine_sizes`,
`--cdc_profile_affine_r2_threshold`, ...) are listed in
`megatron/training/arguments.py` and described in
[`docs/heteropipe_design.md`](docs/heteropipe_design.md).

## Reproducing the thesis experiment

The core experiment runs a 4-stage GPT pipeline on a single node and compares the
`dynamic_mb` schedule against the zero-bubble ZBH1 baseline across microbatch
counts and injected latencies. Main configuration:

| Parameter | Value |
| --- | --- |
| GPUs | 4× RTX 3090, single node |
| Pipeline stages | 4 (`PP=4`, `TP=1`, one stage per GPU) |
| Transformer layers | 26 |
| Hidden size / FFN size | 2048 / 8192 |
| Global batch | 64 |
| Microbatch counts | 4, 8, 16 |
| Injected latency | 0, 5, 10 ms |
| Iterations | 200 per run (×3 repetitions) |

The dynamic schedule is solved at **zero assumed latency**
(`--cdc_dynamic_mb_schedule_lat_ms 0`) and executed under the real injected
delay. Step-by-step commands and the sweep harness are in
[`docs/reproduction.md`](docs/reproduction.md). The CrossPipe Slurm experiment
generators under [`test_crossdc/exp_slurm/`](test_crossdc/exp_slurm/) reproduce
the original CrossPipe paper experiments and are a useful template for
multi-node runs.

## Known limitations

- Validated only for `TP=1` (one stage per GPU). The variable-size
  weight-gradient path is exercised without tensor/sequence parallelism.
- The weight-gradient block `W` is not separately affine-profiled; it keeps
  CrossPipe's canonical cost in the MILP.
- The MILP is solved under a wall-clock limit; for larger problems the solver
  typically returns a good incumbent rather than a proved optimum.
- Folding the injected latency directly into the solver was found to *hurt*; the
  recommended setting is `--cdc_dynamic_mb_schedule_lat_ms 0` (see the thesis).

## License and acknowledgements

HeteroPipe inherits the license of Megatron-LM; see [`LICENSE`](LICENSE). It
builds directly on [CrossPipe](https://github.com/spcl/crosspipe) by SPCL
(ETH Zürich) and on NVIDIA [Megatron-LM / Megatron-Core](https://github.com/NVIDIA/Megatron-LM).
Full attribution is in [`NOTICE.md`](NOTICE.md).
