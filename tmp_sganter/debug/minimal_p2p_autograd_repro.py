"""Add autograd + compute between recv and send. Closer to pipeline fwd.

For each rank in the chain:
  - receive input from rank-1 (requires_grad buffer)
  - run a small forward: out = relu(linear(input))
  - send out to rank+1
  - optionally run backward at the end

Run:
  torchrun --standalone --nproc_per_node=4 minimal_p2p_autograd_repro.py \
    --pattern 3,5,4,4 [--backward]
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", required=True)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--backward", action="store_true", help="run backward after forward")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    sizes = [int(s) for s in args.pattern.split(",")]

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    if rank == 0:
        print(f"pattern={sizes} world={world} backward={args.backward} dtype={args.dtype}", flush=True)

    # Simple per-rank model: a linear layer (different weights per rank so the forward is non-trivial).
    torch.manual_seed(rank + 1234)
    linear = nn.Linear(args.hidden, args.hidden, bias=False).to(device="cuda", dtype=dtype)

    outputs = []  # keep references so autograd graph isn't freed
    t0 = time.perf_counter()

    # Post all receives up front (to mirror ahead-of-time recv posting)
    recv_reqs = []
    recv_bufs = []
    if rank > 0:
        for mb, s in enumerate(sizes):
            buf = torch.empty(args.seq, s, args.hidden, device="cuda", dtype=dtype, requires_grad=True)
            req = dist.irecv(buf, src=rank - 1)
            recv_reqs.append((req, mb))
            recv_bufs.append(buf)
        print(f"[r{rank}] posted {len(recv_reqs)} irecvs @ {time.perf_counter()-t0:.2f}s", flush=True)

    send_reqs = []
    for mb, s in enumerate(sizes):
        if rank == 0:
            # Rank 0 synthesises input (no recv).
            x = torch.randn(args.seq, s, args.hidden, device="cuda", dtype=dtype, requires_grad=True)
        else:
            # Wait for the corresponding recv, then use the tensor.
            recv_reqs[mb][0].wait()
            x = recv_bufs[mb]
        # Trivial "forward"
        y = torch.relu(linear(x))
        outputs.append(y)
        if rank < world - 1:
            req = dist.isend(y, dst=rank + 1)
            send_reqs.append((req, mb))
            print(f"[r{rank}] mb={mb} size={s} computed+sent @ {time.perf_counter()-t0:.2f}s", flush=True)
        else:
            print(f"[r{rank}] mb={mb} size={s} computed (last stage) @ {time.perf_counter()-t0:.2f}s", flush=True)

    for req, mb in send_reqs:
        req.wait()

    torch.cuda.synchronize()
    print(f"[r{rank}] fwd done @ {time.perf_counter()-t0:.2f}s", flush=True)

    if args.backward:
        # Run backward on each output (no real loss, just to trigger autograd engine)
        for y in outputs:
            loss = y.sum()
            loss.backward(retain_graph=False)
        torch.cuda.synchronize()
        print(f"[r{rank}] bwd done @ {time.perf_counter()-t0:.2f}s", flush=True)

    dist.barrier()
    print(f"[r{rank}] barrier done @ {time.perf_counter()-t0:.2f}s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
