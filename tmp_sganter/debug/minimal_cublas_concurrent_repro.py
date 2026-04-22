"""E3 — minimal 4-rank concurrent first-time-new-shape GEMM repro.

Recreates the concurrency pattern we saw in Megatron at hang time:
  rank 3: first forward with size 3 (new shape on rank 3)
  rank 2: first forward with size 5 (new shape on rank 2, after size-3 fwd)
  rank 1: first backward with size 3 (new backward shape on rank 1)
  rank 0: first backward with size 3 (new backward shape on rank 0)

All four ranks hit their first-time-new-shape GEMM within ~100 ms, aligned via
dist.barrier(). No TransformerEngine, no NCCL P2P, no Megatron code. Just
`nn.Linear` + autograd.

If this HANGS  → bug is in PyTorch/cuBLAS stack itself (rankless of TE).
If this WORKS  → TE is a necessary condition for the hang; continue with E2/E4.

Run:
  torchrun --standalone --nproc_per_node=4 minimal_cublas_concurrent_repro.py
"""
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn


def do_forward(linear, size, seq, hidden, dtype, tag):
    print(f"[r{dist.get_rank()}] {tag} fwd size={size} begin", flush=True)
    t0 = time.perf_counter()
    x = torch.randn(seq, size, hidden, device="cuda", dtype=dtype, requires_grad=True)
    y = linear(x)
    torch.cuda.synchronize()
    print(f"[r{dist.get_rank()}] {tag} fwd size={size} done in {time.perf_counter()-t0:.3f}s", flush=True)
    return y


def do_backward(linear, size, seq, hidden, dtype, tag):
    print(f"[r{dist.get_rank()}] {tag} bwd size={size} begin", flush=True)
    t0 = time.perf_counter()
    x = torch.randn(seq, size, hidden, device="cuda", dtype=dtype, requires_grad=True)
    y = linear(x)
    y.sum().backward()
    torch.cuda.synchronize()
    print(f"[r{dist.get_rank()}] {tag} bwd size={size} done in {time.perf_counter()-t0:.3f}s", flush=True)


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    SEQ = 512
    HIDDEN = 128
    OUT_HIDDEN = 4 * HIDDEN  # mimic an FFN-ish width, as Megatron uses
    DTYPE = torch.bfloat16
    PROFILE_SIZE = 4  # the size all warmup iterations use in Megatron (micro_batch_size)

    # Each rank builds a distinct linear (different weights, same shape)
    torch.manual_seed(rank + 1234)
    linear = nn.Linear(HIDDEN, OUT_HIDDEN, bias=False).to(device="cuda", dtype=DTYPE)

    # -------- Phase 1: profiling-like warmup, all ranks simultaneously --------
    # In Megatron, iterations 1-3 run ZBH1 with size PROFILE_SIZE. Each rank compiles
    # TE/cuBLAS kernels for this shape before the dynamic schedule kicks in.
    if rank == 0:
        print("=== phase 1: warmup all ranks with size=4 ===", flush=True)
    for _ in range(3):
        x = torch.randn(SEQ, PROFILE_SIZE, HIDDEN, device="cuda", dtype=DTYPE, requires_grad=True)
        y = linear(x)
        y.sum().backward()
    torch.cuda.synchronize()
    dist.barrier()

    # -------- Phase 2: concurrent first-time-new-shape GEMMs, aligned --------
    # Ranks do different operations on different never-before-seen shapes,
    # all entering their GEMM within ~100 ms of each other.
    if rank == 0:
        print("=== phase 2: concurrent first-new-shape GEMMs ===", flush=True)
    dist.barrier()

    tag = f"phase2"
    if rank == 0:
        do_backward(linear, size=3, seq=SEQ, hidden=HIDDEN, dtype=DTYPE, tag=tag)
    elif rank == 1:
        do_backward(linear, size=3, seq=SEQ, hidden=HIDDEN, dtype=DTYPE, tag=tag)
    elif rank == 2:
        do_forward(linear, size=5, seq=SEQ, hidden=HIDDEN, dtype=DTYPE, tag=tag)
    elif rank == 3:
        do_forward(linear, size=3, seq=SEQ, hidden=HIDDEN, dtype=DTYPE, tag=tag)

    dist.barrier()
    if rank == 0:
        print("=== phase 2 done ===", flush=True)

    # -------- Phase 3: second round of first-new-shapes, different shapes per rank --------
    if rank == 0:
        print("=== phase 3: each rank a DIFFERENT new shape ===", flush=True)
    dist.barrier()
    # Put different new shapes on each rank to maximize concurrent JIT/cache pressure
    per_rank_sizes = {0: 6, 1: 9, 2: 12, 3: 2}
    size = per_rank_sizes[rank]
    do_forward(linear, size=size, seq=SEQ, hidden=HIDDEN, dtype=DTYPE, tag="phase3")
    do_backward(linear, size=size, seq=SEQ, hidden=HIDDEN, dtype=DTYPE, tag="phase3")
    dist.barrier()
    if rank == 0:
        print("=== phase 3 done ===", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
