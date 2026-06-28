# Reproducing the thesis experiment

The core thesis result compares the HeteroPipe `dynamic_mb` schedule against the
zero-bubble **ZBH1** baseline on a 4-stage GPT pipeline, on a single node, across
three microbatch counts and three injected latencies.

## Setup

| Parameter | Value |
| --- | --- |
| GPUs | 4× RTX 3090, single node |
| Pipeline / tensor parallel | `PP=4`, `TP=1` (one stage per GPU) |
| Transformer layers | 26 |
| Hidden size / FFN size | 2048 / 8192 |
| Attention heads | 16 |
| Sequence length | 2048 |
| Global batch | 64 |
| Microbatch counts `n_mb` | 4, 8, 16 → `--micro-batch-size` 16, 8, 4 |
| Injected latency | 0, 5, 10 ms |
| Iterations | 200 per run, 3 repetitions |

The latency is emulated on **stock PyTorch** with
`--cdc_stock_inject_latency_ms`; no custom PyTorch build is needed. The schedule
is solved at **zero assumed latency** (`--cdc_dynamic_mb_schedule_lat_ms 0`) and
executed under the real injected delay.

## One cell of the sweep

Each `(schedule, n_mb, latency)` combination is one run. Example: `dynamic_mb`,
`n_mb=8` (`mbs=8`), 10 ms.

```bash
torchrun --nproc_per_node 4 pretrain_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 4 \
    --num_dc 2 --pp_stages_per_dc 2 2 \
    --transformer-impl local \
    --num-layers 26 --hidden-size 2048 --ffn-hidden-size 8192 \
    --num-attention-heads 16 \
    --seq-length 2048 --max-position-embeddings 2048 \
    --micro-batch-size 8 --global-batch-size 64 \
    --train-iters 200 --bf16 \
    --data-path <your_data_prefix> \
    --tokenizer-type Llama2Tokenizer --tokenizer-model <your_tokenizer.model> \
    --enable_cdcpp_scheduler \
    --dynamic_schedule dynamic_mb \
    --cdc_profile_affine \
    --cdc_dynamic_mb_schedule_lat_ms 0 \
    --cdc_stock_inject_latency_ms 10 \
    --cdc_stock_inject_link cross_boundary \
    --cdc_stock_inject_warmup_iters 10
```

The baseline cell is the same command with the schedule flag replaced:

```bash
    --static_schedule ZBH1            # instead of --dynamic_schedule dynamic_mb
```

## Running the sweep

The full study is the cross product:

- schedules: `dynamic_mb`, `ZBH1` (and optionally `1F1B`),
- microbatch counts: `--micro-batch-size` 16, 8, 4 (for `n_mb` = 4, 8, 16 at
  `--global-batch-size 64`),
- injected latencies: `--cdc_stock_inject_latency_ms` 0, 5, 10,
- 3 repetitions each.

Iterate over those values and collect the per-iteration time that the scheduler
logs. Measure on otherwise-idle GPUs; the thesis discards the first iterations of
each window so that profiling, the solver, and the cuBLAS shape cache have
settled before timing.

> **Note.** The exact plotting/aggregation harness used to produce the thesis
> figures is not part of this repository. The commands above reproduce the
> measured quantity (per-iteration time per cell); the figures are then produced
> by aggregating those numbers across repetitions.

## Multi-node / original CrossPipe experiments

The Slurm generators under [`../test_crossdc/exp_slurm/`](../test_crossdc/exp_slurm/)
reproduce the original CrossPipe paper experiments (`lat_bw_delay/`,
`extra_gbs_mem/`, `dc4/`, `pp_dp_tradeoff/`). Those use CrossPipe's
modified-PyTorch injection path (`--cdc_latency_bandwidth_delay_as_F_stage`),
not the stock injection used for the single-node thesis experiments. They are a
good template for adapting HeteroPipe runs to a Slurm cluster — see
[`../docs/upstream_crosspipe_readme.md`](upstream_crosspipe_readme.md) for the
path/CPLEX setup they expect.
