import sys, types
sys.path.insert(0, "crosspipe-main")

# Stub missing optional deps before any import triggers them
for mod_name in [
    "docplex", "docplex.cp", "docplex.cp.model", "docplex.cp.config",
    "drawsvg",
]:
    m = types.ModuleType(mod_name)
    m.__path__ = []
    m.context = None
    sys.modules[mod_name] = m

from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.pipeline_config import SystemConfig
from auto_schedule import UnidirectionalZBDependencyGraph, UnidirectionalDynamicBatchSizeZBDependencyGraph

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.expanduser("~/Downloads/temp_BT_Output_Folder")
os.makedirs(OUT_DIR, exist_ok=True)


class PipelineConfig:
    """Single source of truth for all pipeline parameters.

    comp[dev][type]: per-sample compute cost (type: 0=F, 1=B, 2=W)
    actual duration = comp[dev][type] * f[mb] + compBias

    comm[dir]: per-sample transmission time (dir: 0=fwd, 1=bwd)
    actual comm = comm[dir] * f[mb] + commLat[src][dst]
    """

    def __init__(
        self,
        N=32,
        npp=4,
        nbm=4,
        comp=None,
        compBias=1,
        comm=None,
        commLat=None,
        M_F=1, M_B=-1, M_W=0,
        M_Limit=-1,
    ):
        self.N = N
        self.npp = npp
        self.nbm = nbm
        self.comp = comp if comp is not None else [[3, 3, 2] for _ in range(npp)]
        self.compBias = compBias
        self.comm = comm if comm is not None else [0, 0]
        self.commLat = commLat if commLat is not None else [
            [1 if abs(i - j) == 1 else 0 for j in range(npp)] for i in range(npp)
        ]
        self.M_F = M_F
        self.M_B = M_B
        self.M_W = M_W
        self.M_Limit = M_Limit

    @property
    def equal_mb_size(self):
        """Microbatch size assuming equal split."""
        return self.N // self.nbm

    def fixed_durations(self, dev=None):
        """Compute T_F, T_B, T_W assuming equal microbatch sizes.
        If dev is None, returns per-device lists. Otherwise returns for one device."""
        f = self.equal_mb_size
        if dev is not None:
            return {
                "T_F": self.comp[dev][0] * f + self.compBias,
                "T_B": self.comp[dev][1] * f + self.compBias,
                "T_W": self.comp[dev][2] * f + self.compBias,
            }
        return {
            "T_F": [self.comp[d][0] * f + self.compBias for d in range(self.npp)],
            "T_B": [self.comp[d][1] * f + self.compBias for d in range(self.npp)],
            "T_W": [self.comp[d][2] * f + self.compBias for d in range(self.npp)],
        }

    def fixed_comm(self):
        """Compute T_alpha assuming equal microbatch sizes.
        Returns the comm latency matrix (commLat only, since comm*f is baked into T_F/T_B)."""
        return self.commLat

    def to_system_cfg(self):
        """Build a SystemConfig with dummy int durations (for base class assertions)."""
        return SystemConfig(
            num_devices=self.npp,
            num_microbatches=self.nbm,
            T_F=1, T_B=1, T_W=1,
            T_alpha=0,
            M_F=self.M_F, M_B=self.M_B, M_W=self.M_W,
            M_Limit=self.M_Limit,
        )

    def to_crosspipe_system_cfg(self):
        """Build a SystemConfig with fixed durations for crosspipe (equal mb sizes)."""
        dur = self.fixed_durations()
        return SystemConfig(
            num_devices=self.npp,
            num_microbatches=self.nbm,
            T_F=[int(v) for v in dur["T_F"]],
            T_B=[int(v) for v in dur["T_B"]],
            T_W=[int(v) for v in dur["T_W"]],
            T_alpha=self.commLat,
            M_F=self.M_F, M_B=self.M_B, M_W=self.M_W,
            M_Limit=self.M_Limit,
        )

    def print_summary(self):
        f = self.equal_mb_size
        dur = self.fixed_durations()
        print(f"N={self.N}, npp={self.npp}, nbm={self.nbm}, equal_f={f}")
        print(f"comp={self.comp}, compBias={self.compBias}")
        print(f"comm={self.comm}, commLat diag+1={self.commLat[0][1] if self.npp > 1 else 0}")
        print(f"Fixed durations (equal f): T_F={dur['T_F']}, T_B={dur['T_B']}, T_W={dur['T_W']}")


# ========================== Shared config ==========================
CFG = PipelineConfig(
    N=32,
    npp=4,
    nbm=4,
    comp=[[3, 3, 2] for _ in range(4)],
    compBias=1,
    comm=[0, 0],
    commLat=[[1 if abs(i - j) == 1 else 0 for j in range(4)] for i in range(4)],
    M_F=1, M_B=-1, M_W=0,
    M_Limit=-1,
)
# ===================================================================


def solve_dynamic(with_plotting=True):
    """Solve with dynamic microbatch sizes."""
    CFG.print_summary()

    g = UnidirectionalDynamicBatchSizeZBDependencyGraph(
        system_cfg=CFG.to_system_cfg(),
        N=CFG.N,
        comp=CFG.comp,
        compBias=CFG.compBias,
        comm=CFG.comm,
        commLat=CFG.commLat,
    )
    g.build_ilp()
    g.solve_ilp(verbose=False, time_limit=60, relative_gap=0)

    mb_sizes = g.get_microbatch_sizes()
    schedule = g.get_schedule()
    if schedule and with_plotting:
        print(f"Microbatch sizes: {mb_sizes} (sum={sum(mb_sizes)})")
        plot_dynamic_schedule(schedule, g)
    
    return schedule, g.get_objective_value()


def solve_crosspipe(with_plotting=True):
    """Solve with crosspipe fixed durations (equal microbatch sizes)."""
    cfg = CFG.to_crosspipe_system_cfg()
    g = UnidirectionalZBDependencyGraph(cfg)
    g.build_ilp()
    g.solve_ilp(verbose=False, time_limit=60, relative_gap=0)
    schedule = g.get_schedule()
    if schedule and with_plotting:
        plot_schedule(schedule, cfg)

    return schedule, g.get_objective_value()


def plot_dynamic_schedule(schedule, graph):
    """Gantt chart for dynamic-batch-size schedules where block widths vary per microbatch."""
    colors = {"F": "#7239DC", "B": "#CD5C5C", "W": "#A0DAB8"}
    num_dev = len(schedule)
    mb_sizes = graph.get_microbatch_sizes()
    comp = graph.comp
    compBias = graph.compBias

    fig, ax = plt.subplots(figsize=(14, 1.2 * num_dev + 0.8))

    for dev, blocks in enumerate(schedule):
        y = num_dev - 1 - dev
        for b in blocks:
            tt_idx = {"F": 0, "B": 1, "W": 2}[b.task_type]
            duration = comp[dev][tt_idx] * mb_sizes[b.mb_id] + compBias
            start = b.end_time - duration
            ax.barh(
                y, duration, left=start, height=0.7,
                color=colors.get(b.task_type, "gray"),
                edgecolor="black", linewidth=0.5,
            )
            cx = start + duration / 2
            if duration > 2:
                ax.text(cx, y, f"{b.task_type}{b.mb_id}", ha="center", va="center", fontsize=7)

    ax.set_yticks(range(num_dev))
    ax.set_yticklabels([f"P{num_dev - 1 - i}" for i in range(num_dev)])
    ax.set_xlabel("Time")
    ax.set_title(f"Dynamic Batch Schedule  (f = {mb_sizes})")

    legend_patches = [mpatches.Patch(color=c, label=t) for t, c in colors.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "schedule_dynamic.png"), dpi=150)


def plot_schedule(schedule, cfg):
    """Gantt-chart style visualization of a pipeline schedule.
    schedule: List[List[PipelineBlockDesc]] — one list per device, sorted by end_time.
    """
    colors = {"F": "#7239DC", "B": "#CD5C5C", "W": "#A0DAB8"}
    num_dev = len(schedule)

    fig, ax = plt.subplots(figsize=(14, 1.2 * num_dev + 0.8))

    for dev, blocks in enumerate(schedule):
        y = num_dev - 1 - dev  # device 0 on top
        for b in blocks:
            task_type = b.task_type
            duration = {
                "F": int(cfg.T_F[0][dev]),
                "B": int(cfg.T_B[0][dev]),
                "W": int(cfg.T_W[0][dev]),
            }[task_type]
            start = b.end_time - duration
            ax.barh(
                y, duration, left=start, height=0.7,
                color=colors.get(task_type, "gray"),
                edgecolor="black", linewidth=0.5,
            )
            # label with microbatch id
            cx = start + duration / 2
            if duration > 2:
                ax.text(cx, y, f"{task_type}{b.mb_id}", ha="center", va="center", fontsize=7)

    ax.set_yticks(range(num_dev))
    ax.set_yticklabels([f"P{num_dev - 1 - i}" for i in range(num_dev)])
    ax.set_xlabel("Time")
    ax.set_title("Pipeline Schedule")

    legend_patches = [mpatches.Patch(color=c, label=t) for t, c in colors.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "schedule.png"), dpi=150)

def plot_improvement_over_different_microbatches(list_mb):
    list_mb = sorted(list_mb)
    y = []
    for nbm in list_mb:
        CFG.nbm = nbm
        schedule_CrossPi, finish_time_CrossPi = solve_crosspipe(with_plotting=True)
        schedule_Dynamic,  finish_time_Dynamic  = solve_dynamic(with_plotting=True)

        improvement = (finish_time_CrossPi - finish_time_Dynamic) / finish_time_CrossPi
        y.append(improvement)
        print(f"nbm={nbm}: CrossPipe={finish_time_CrossPi}, Dynamic={finish_time_Dynamic}, improvement={improvement:.3%}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list_mb, [v * 100 for v in y], marker="o", linewidth=2, color="#7239DC")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Number of microbatches (nbm)")
    ax.set_ylabel("Makespan improvement over CrossPipe (%)")
    ax.set_title(f"Dynamic vs CrossPipe schedule improvement\n(N={CFG.N}, npp={CFG.npp})")
    ax.set_xticks(list_mb)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "improvement_over_microbatches.png"), dpi=150)



if __name__ == "__main__":
    # solve_dynamic()
    # solve_crosspipe()
    make_improvement = True 
    if make_improvement:
        nmb = 2
        list_mb = []
        while nmb < CFG.N:
            list_mb.append(nmb)
            nmb *= 2
    
        plot_improvement_over_different_microbatches(list_mb)
    plt.show()
