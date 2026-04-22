"""Minimal standalone repro for the variable-size P2P hang.

Run as (example, 4 GPUs):
  torchrun --standalone --nproc_per_node=4 minimal_p2p_repro.py \
    --pattern "3,5,4,4" --mode chain

Patterns: comma-separated sizes. Pattern length = number of back-to-back ops.
Modes:
  pair   -- only rank 0 -> rank 1 does back-to-back sends; others idle.
  chain  -- rank i -> rank i+1 for i in 0..world-2 (like a pipeline warmup).

No Megatron, no TransformerEngine, no autograd. Just raw NCCL P2P.
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist


def pair_mode(rank, world_size, sizes, seq, hidden, dtype, timeout_s):
    """Only rank 0 -> rank 1; others idle (but still in world group)."""
    if rank == 0:
        # Post N isends of varying sizes.
        reqs = []
        for mb, s in enumerate(sizes):
            buf = torch.ones(seq, s, hidden, device="cuda", dtype=dtype)
            req = dist.isend(buf, dst=1)
            print(f"[r0] isend mb={mb} size={s} posted", flush=True)
            reqs.append((req, buf, mb))
        t0 = time.perf_counter()
        for req, buf, mb in reqs:
            req.wait()
            torch.cuda.synchronize()
            print(f"[r0] isend mb={mb} done @ {time.perf_counter() - t0:.2f}s", flush=True)
    elif rank == 1:
        reqs = []
        for mb, s in enumerate(sizes):
            buf = torch.empty(seq, s, hidden, device="cuda", dtype=dtype)
            req = dist.irecv(buf, src=0)
            print(f"[r1] irecv mb={mb} size={s} posted", flush=True)
            reqs.append((req, buf, mb))
        t0 = time.perf_counter()
        for req, buf, mb in reqs:
            req.wait()
            torch.cuda.synchronize()
            print(f"[r1] irecv mb={mb} done @ {time.perf_counter() - t0:.2f}s", flush=True)
    else:
        # Idle ranks — just hold a barrier at end.
        pass

    dist.barrier()
    print(f"[r{rank}] barrier done", flush=True)


def chain_mode(rank, world_size, sizes, seq, hidden, dtype, timeout_s):
    """Every rank i sends to rank i+1 (except last) with the same size pattern.

    This mimics a pipeline's forward warmup: every non-last rank sends `N` mbs
    back-to-back to its next; every non-first rank receives `N` from its prev.
    """
    sends, recvs = [], []

    # Post receives first (they're cheap and need to be matched).
    if rank > 0:
        for mb, s in enumerate(sizes):
            buf = torch.empty(seq, s, hidden, device="cuda", dtype=dtype)
            req = dist.irecv(buf, src=rank - 1)
            recvs.append((req, buf, mb))
        print(f"[r{rank}] posted {len(recvs)} irecvs", flush=True)

    if rank < world_size - 1:
        for mb, s in enumerate(sizes):
            buf = torch.ones(seq, s, hidden, device="cuda", dtype=dtype)
            req = dist.isend(buf, dst=rank + 1)
            sends.append((req, buf, mb))
        print(f"[r{rank}] posted {len(sends)} isends", flush=True)

    t0 = time.perf_counter()
    for req, buf, mb in recvs:
        req.wait()
        torch.cuda.synchronize()
        print(f"[r{rank}] recv mb={mb} size={buf.shape[1]} done @ {time.perf_counter() - t0:.2f}s", flush=True)
    for req, buf, mb in sends:
        req.wait()
        torch.cuda.synchronize()
        print(f"[r{rank}] send mb={mb} size={buf.shape[1]} done @ {time.perf_counter() - t0:.2f}s", flush=True)

    dist.barrier()
    print(f"[r{rank}] barrier done", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", required=True, help="comma-separated sizes, e.g. 3,5,4,4")
    p.add_argument("--mode", choices=["pair", "chain"], default="chain")
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dtype", choices=["bf16", "fp32", "fp16"], default="bf16")
    p.add_argument("--timeout", type=int, default=30)
    args = p.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}
    dtype = dtype_map[args.dtype]
    sizes = [int(s) for s in args.pattern.split(",")]

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    if rank == 0:
        print(f"mode={args.mode} world={world_size} pattern={sizes} seq={args.seq} hidden={args.hidden} dtype={args.dtype}", flush=True)

    if args.mode == "pair":
        pair_mode(rank, world_size, sizes, args.seq, args.hidden, dtype, args.timeout)
    else:
        chain_mode(rank, world_size, sizes, args.seq, args.hidden, dtype, args.timeout)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
