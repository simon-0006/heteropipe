"""Chain mode with a TransformerEngine layer in the middle, variable sizes.

If this hangs with pattern [3,5,4,4] but not with pattern [4,4,4,4], TE is
implicated.

torchrun --standalone --nproc_per_node=4 minimal_p2p_te_repro.py \
  --pattern 3,5,4,4 [--backward] [--layers 3]
"""
import argparse
import os
import time

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te


def build_te_block(hidden, dtype):
    # A TE transformer layer: LayerNorm + MHA + MLP + LayerNorm.
    return te.TransformerLayer(
        hidden_size=hidden,
        ffn_hidden_size=hidden * 4,
        num_attention_heads=4,
        layer_number=1,
        params_dtype=dtype,
        self_attn_mask_type="no_mask",
    ).to(device="cuda", dtype=dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", required=True)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--backward", action="store_true")
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
        print(f"pattern={sizes} world={world} layers={args.layers} backward={args.backward}", flush=True)

    torch.manual_seed(rank + 1234)
    model = torch.nn.Sequential(*[build_te_block(args.hidden, dtype) for _ in range(args.layers)])

    outputs = []
    t0 = time.perf_counter()

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
            x = torch.randn(args.seq, s, args.hidden, device="cuda", dtype=dtype, requires_grad=True)
        else:
            recv_reqs[mb][0].wait()
            x = recv_bufs[mb]

        print(f"[r{rank}] mb={mb} size={s} starting fwd @ {time.perf_counter()-t0:.2f}s", flush=True)
        y = model(x)
        print(f"[r{rank}] mb={mb} size={s} fwd done @ {time.perf_counter()-t0:.2f}s", flush=True)
        outputs.append(y)

        if rank < world - 1:
            req = dist.isend(y, dst=rank + 1)
            send_reqs.append((req, mb))
            print(f"[r{rank}] mb={mb} size={s} sent @ {time.perf_counter()-t0:.2f}s", flush=True)

    for req, mb in send_reqs:
        req.wait()

    torch.cuda.synchronize()
    print(f"[r{rank}] fwd chain done @ {time.perf_counter()-t0:.2f}s", flush=True)

    if args.backward:
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
