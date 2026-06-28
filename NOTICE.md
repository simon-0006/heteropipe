# Attribution and lineage

HeteroPipe is a research fork and stands on two upstream projects.

- **NVIDIA Megatron-LM / Megatron-Core** — the transformer model
  implementations, parallelism, and training stack.
  <https://github.com/NVIDIA/Megatron-LM>
  Licensed under the terms in [`LICENSE`](LICENSE); the original copyright
  notices are retained in the source files.

- **CrossPipe** (SPCL, ETH Zürich) — the communication-aware pipeline schedule
  generation and execution that HeteroPipe extends.
  <https://github.com/spcl/crosspipe>
  See *CrossPipe: Towards Optimal Pipeline Schedules for Cross-Datacenter
  Training*, Chen et al., 2025 (arXiv:2507.00217).

HeteroPipe's own additions (the dynamic microbatch-size MILP, the affine
profiler, the variable-size runtime, and the stock-PyTorch latency injection)
are released under the same license as the inherited code. The upstream READMEs
are preserved unmodified under [`docs/`](docs/) for reference.

This file is attribution, not legal advice. The governing license is
[`LICENSE`](LICENSE).
