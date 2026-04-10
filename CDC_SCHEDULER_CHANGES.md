# CDC Scheduler – Compatibility Fixes

This document describes every change made to the CDC PP scheduler codebase to get
`pretrain_gpt.py` running with a standard Megatron-LM environment (no custom
PyTorch build, no APEX pre-installed, no latency-injection experiment).

---

## 1. `experiment_manager.py` — missing attribute in `print_expertiment_info`

**File:** `megatron/core/pipeline_parallel/cdc_scheduler/experiment_manager.py`

**Problem:**
`print_expertiment_info()` unconditionally referenced `self.exp_logging_end_iter`,
but that attribute is only assigned in `__init__` when
`cdc_latency_bandwidth_delay_as_F_stage` is non-empty (i.e. a latency-injection
experiment is configured). When running without that flag the attribute never
exists, causing an `AttributeError` at startup.

**Fix:**
Guard the line with `hasattr`:

```python
# before
infos.append(f'iter [{self.exp_logging_end_iter}]: end testing')

# after
if hasattr(self, 'exp_logging_end_iter'):
    infos.append(f'iter [{self.exp_logging_end_iter}]: end testing')
```

**Functional impact:** None. The line is only informational logging; skipping it
when the attribute doesn't exist is correct behaviour.

---

## 2. `svg_event.py` — hard dependency on `drawsvg`

**File:** `megatron/core/pipeline_parallel/cdc_scheduler/pp_generator/svg_event.py`

**Problem:**
`drawsvg` was imported unconditionally at module level. If the package is not
installed the entire import fails, crashing both ranks before any training starts.

**Fix:**
Wrap the import in a try/except and guard `draw_events` with an early return:

```python
# before
import drawsvg as draw

# after
try:
    import drawsvg as draw
except ImportError:
    draw = None

def draw_events(...):
    if draw is None:
        print("Warning: drawsvg is not installed, skipping SVG generation.")
        return None
    ...
```

**Functional impact:** SVG pipeline-schedule visualisations are silently skipped
when `drawsvg` is absent. Install `drawsvg` (`pip install drawsvg`) to restore
them.

---

## 3. `pp_scheduler.py` — five separate issues

**File:** `megatron/core/pipeline_parallel/cdc_scheduler/pp_scheduler.py`

### 3a. `validate_args` — `virtual_pipeline_model_parallel_size` not set

**Problem:**
The assertion required `args.virtual_pipeline_model_parallel_size` to equal
`num_chunks`. For ZBH1 (`num_chunks=1`) this argument defaults to `None` in
Megatron, making the assertion fail. Furthermore, the rest of Megatron (e.g.
`is_pipeline_last_stage`) reads this value directly and crashes with a
`TypeError: 'NoneType' - 1` when it is `None`.

**Fix:**
Accept `None` as equivalent to `1`, then force the argument and the parallel-state
to the correct value so the rest of Megatron sees a consistent setting:

```python
# before
assert (
    args.virtual_pipeline_model_parallel_size
    == self.pp_schedule.sys_config.num_chunks
), "..."

# after
num_chunks = self.pp_schedule.sys_config.num_chunks
vp_size = args.virtual_pipeline_model_parallel_size or 1
assert vp_size == num_chunks, "..."
args.virtual_pipeline_model_parallel_size = num_chunks
mpu.set_virtual_pipeline_model_parallel_world_size(num_chunks)
mpu.set_virtual_pipeline_model_parallel_rank(0)
```

**Functional impact:** `virtual_pipeline_model_parallel_size` is now always set
to `num_chunks` in `args`. Any code that relies on this value being `None` to
detect "no virtual pipeline" will instead see `1`. In practice Megatron treats
`1` and `None` equivalently for single-chunk schedules.

### 3b. `setup_grad_sync` — assumes `no_sync_func` is always set

**Problem:**
`setup_grad_sync` asserted `config.no_sync_func is not None`. Megatron only
sets `no_sync_func` when `overlap_grad_reduce=True` (DDP with async grad
reduction). Without that flag the function crashes on the first training step.

**Fix:**
Fall back to `contextlib.nullcontext` (a no-op context manager) when
`no_sync_func` is `None`:

```python
# before
assert self.config.no_sync_func is not None
if isinstance(self.config.no_sync_func, list): ...

# after
if self.config.no_sync_func is None:
    from contextlib import nullcontext
    self.no_sync_func = [nullcontext] * num_chunks
elif isinstance(self.config.no_sync_func, list):
    self.no_sync_func = self.config.no_sync_func
else:
    self.no_sync_func = [self.config.no_sync_func]
```

**Functional impact:** Without `overlap_grad_reduce`, gradient synchronisation
happens synchronously at the end of the backward pass (standard Megatron
behaviour). The CDC scheduler's `disable_grad_sync` / `enable_grad_sync` calls
become no-ops, which is correct — there is nothing asynchronous to suppress.

### 3c. `forward_backward_func` — spurious `bucket_groups` assertion

**Problem:**
Before the first training step the code asserted:

```python
assert not any(bucket_group.is_last_microbatch
               for bucket_group in model[chunk_id].bucket_groups)
```

`training.py` calls `model_chunk.zero_grad_buffer()` *before*
`forward_backward_func`, and `zero_grad_buffer` explicitly sets
`is_last_microbatch = True` on every bucket (by design). The assertion therefore
always fails on the very first call.

**Fix:**
Remove the assertion and replace it with an explanatory comment:

```python
# is_last_microbatch is set to True by zero_grad_buffer() before this call,
# so no assertion needed
```

**Functional impact:** None. The assertion was a dead check that could never pass
in normal execution.

### 3d. `schedule_comm_event` — latency-injection check fires unconditionally

**Problem:**
In `WAIT_RECV_NEXT` and `WAIT_RECV_PREV` branches the code asserted
`hasattr(handle, 'wait_with_lat_delay_in_ms')` *before* checking whether latency
injection was even enabled (`self.cdc_recv_next / cdc_recv_prev`). Standard
PyTorch handles do not have this method (it requires a custom build), so the
assertion fired on every P2P recv even when no latency injection was configured.

**Fix:**
Move the assertion inside the `if self.cdc_recv_*:` block:

```python
# before
assert hasattr(handle, "wait_with_lat_delay_in_ms"), "..."
if self.cdc_recv_next:
    handle.wait_with_lat_delay_in_ms(...)
else:
    handle.wait()

# after
if self.cdc_recv_next:
    assert hasattr(handle, "wait_with_lat_delay_in_ms"), "..."
    handle.wait_with_lat_delay_in_ms(...)
else:
    handle.wait()
```

**Functional impact:** None when latency injection is disabled (the default).
When `cdc_recv_next` or `cdc_recv_prev` are enabled a custom PyTorch build with
`wait_with_lat_delay_in_ms` is still required — the assertion is preserved there.

---

## 4. Environment setup — APEX CUDA extensions

**Problem:**
The CDC scheduler's W-grad split (`wgrad_split=True` for ZBH1) requires
`gradient_accumulation_fusion=True`, which in turn requires the APEX
`fused_weight_gradient_mlp_cuda` CUDA extension. The package was not installed.

**Fix:**
Build APEX from source, pointing at the system CUDA 12.5 installation (the conda
environment's `nvcc` is CUDA 11.7, which does not match the PyTorch 12.8 ABI):

```bash
cd ~/apex
# comment out check_cuda_torch_binary_vs_bare_metal in setup.py
CUDA_HOME=/usr/local/cuda-12.5 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
    pip install -v --no-cache-dir --no-build-isolation .
```

**Functional impact:** `gradient_accumulation_fusion` now works. Do **not** pass
`--no-gradient-accumulation-fusion` when using the CDC scheduler with ZBH1.

---

## Required run flags

The following flags must be added to any `pretrain_gpt.py` invocation that uses
the CDC PP scheduler:

| Flag | Reason |
|---|---|
| `--head_tail_as_one_layer` | Required by `get_num_layers_in_chunk` |
| `--no-align-grad-reduce` | Not supported by the CDC scheduler |
| `--no-align-param-gather` | Not supported by the CDC scheduler |
| *(omit)* `--no-gradient-accumulation-fusion` | W-grad split requires fusion |

---

## Working run command

```shell
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 pretrain_gpt.py \
  --static_schedule ZBH1 \
  --enable_cdcpp_scheduler \
  --cdc_profile_iter 2 \
  --pipeline-model-parallel-size 2 \
  --num-layers 6 \
  --hidden-size 128 \
  --num-attention-heads 4 \
  --seq-length 512 \
  --max-position-embeddings 512 \
  --micro-batch-size 2 \
  --global-batch-size 16 \
  --train-iters 4 \
  --lr 0.0001 \
  --lr-warmup-iters 1 \
  --eval-interval 1000 \
  --eval-iters 3 \
  --split 949,50,1 \
  --tokenizer-type NullTokenizer \
  --vocab-size 50257 \
  --tensor-model-parallel-size 1 \
  --no-masked-softmax-fusion \
  --no-bias-gelu-fusion \
  --no-bias-dropout-fusion \
  --no-overlap-p2p-communication \
  --use-cpu-initialization \
  --mock-data \
  --bf16 \
  --profile \
  --profile-ranks 0 \
  --profile-step-start 0 \
  --profile-step-end 3 \
  --tensorboard-dir ./tb_logs \
  --head_tail_as_one_layer \
  --no-align-grad-reduce \
  --no-align-param-gather
```

Adjust `CUDA_VISIBLE_DEVICES` to whichever two GPUs are free.
The run exits cleanly at iteration `cdc_profile_iter + 1` (iter 3 for the command
above) — this is the profiling-phase exit, not an error.
