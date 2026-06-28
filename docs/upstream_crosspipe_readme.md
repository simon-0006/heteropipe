# Upstream CrossPipe README

> This is the README inherited from the upstream [CrossPipe](https://github.com/spcl/crosspipe) project, on which HeteroPipe is based. It is preserved here for attribution and reference. See the top-level [README.md](../README.md) for HeteroPipe.

# CrossPipe: Towards Optimal Pipeline Schedules for Cross-Datacenter Training (ATC '25).

## Environment Setup

### Prerequisites
- **PyTorch NGC Container**: Version `24.10-py3` used in the paper.
- **Modified PyTorch**: Compile and install the modified PyTorch from [this branch](https://github.com/C-TC/pytorch/tree/lat_bw_inject_2410) for latency and bandwidth delay injection capabilities
  - Current modifications are based on PyTorch commit of `24.10-py3`
  - Adapt modifications to your specific PyTorch version for better compatibility
  - Install extra dependencies. Refer to [Dockerfile](./podman/Dockerfile).

## Code Structure

The core of CrossPipe implementation is organized as follows:

```
megatron/core/pipeline_parallel/cdc_scheduler/
├── pp_generator/           # Pipeline schedule generation algorithms
├── pp_scheduler.py         # CrossPipe execution engine
└── ...                     # Core CrossPipe module implementation

test_crossdc/exp_slurm/     # Slurm scripts for evaluations
```

## Experimental Evaluation

We provide slurm scripts generator in `test_crossdc/exp_slurm/` that reproduce the experiments from our paper:

- **Section 7.1**: `lat_bw_delay/`
- **Section 7.2**: `extra_gbs_mem/`
- **Section 7.3**: `dc4/`
- **Section 7.4**: `pp_dp_tradeoff/`

### Environment Configuration

To adapt these scripts to your environment:

1. **Update Paths**: Modify paths in `test_crossdc/exp_slurm/source_me.sh`
2. **Install CPLEX**: Ensure the `cpoptimizer` executable is available in your PATH, e.g. [dc4](https://github.com/spcl/crosspipe/blob/main/test_crossdc/exp_slurm/dc4/generate.py#L283)
3. **PyTorch Override**: If using the modified PyTorch, ensure it's properly loaded, e.g. [dc4](https://github.com/spcl/crosspipe/blob/main/test_crossdc/exp_slurm/dc4/generate.py#L284-L285). Ignore if it is installed.
4. Other paths in the generator template, e.g. `TRITON_HOME`, `BASE_PATH`.

## PyTorch Modifications

CrossPipe requires specific PyTorch modifications for network delay injection:

### Latency Delay Injection
Use the added API `work_handle.wait_with_lat_delay_in_ms()`:

e.g. to inject latency delay of 100ms:

```python
# Receiver side
work_handle = dist.irecv(tensor, src, group)
# ... other operations ...
work_handle.wait_with_lat_delay_in_ms(timedelta(milliseconds=100))
```

### Bandwidth Delay Injection
Utilize the `tag` argument (unused in original ProcessGroupNCCL implementation):

e.g. to inject bandwidth delay of 50ms:

```python
# Sender side
work_handle = dist.isend(tensor, dst, group, tag=50)

# Receiver side  
work_handle = dist.irecv(tensor, src, group, tag=50)
```

**Note**: Both delay types can be combined. This injection method is specifically designed for CrossPipe, which uses separate GPU streams for each send/receive direction.

### Known Issues and Workarounds

We set `export CUDA_DEVICE_MAX_CONNECTIONS=0` and modified Megatron code (similar to reverting [this commit](https://github.com/NVIDIA/Megatron-LM/commit/bdd973128802031ecf838ec3a8733100077ad455)) to address a PyTorch bug that serializes communication and computation kernels when recording timing-enabled CUDA events on communication streams. This is similar to [PyTorch issue](https://github.com/pytorch/pytorch/issues/143890).

## Citation

```bibtex
@misc{chen2025crosspipeoptimalpipelineschedules,
      title={CrossPipe: Towards Optimal Pipeline Schedules for Cross-Datacenter Training}, 
      author={Tiancheng Chen and Ales Kubicek and Langwen Huang and Torsten Hoefler},
      year={2025},
      eprint={2507.00217},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2507.00217}, 
}
```

*Note: This citation will be updated with the official ATC '25 version once available.*
