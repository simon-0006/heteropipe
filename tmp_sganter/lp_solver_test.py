import os, sys, types
# Add the heteropipe repo root (parent of `megatron/`) and this script's dir
# (for the local `auto_schedule` import) so imports work regardless of cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _THIS_DIR)

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


# ========================== Shared config (2-GPU benchmark, profiled) =====
# Source: tb_logs/total.json, f_profile = --micro-batch-size = 2.
# Per-sample cost in microseconds = (T_x[dev] / f_profile) * 1e6.
# Microsecond scale keeps values integer-friendly for the MILP.
CFG = PipelineConfig(
    N=32,
    npp=2,
    nbm=16,
    comp=[
        [3661, 4057,  12],   # dev 0: F, B, W  (T_W ~0: grad_accum_fusion absorbs dW into B)
        [3569, 2206,  94],   # dev 1
    ],
    compBias=0,              # baseline: pure-linear MILP ceiling (no fixed overhead)
    comm=[0, 0],             # matches CrossPipe's MILP (ignores T_bw)
    commLat=[
        [ 0, 95],            # 95 us on-node NVLink (from T_alpha)
        [95,  0],
    ],
    M_F=1, M_B=-1, M_W=0,
    M_Limit=-1,
)
# ===========================================================================


def solve_dynamic(with_plotting=True, time_limit=600, relative_gap=0.001,
                  out_name="schedule_dynamic.png"):
    """Solve with dynamic microbatch sizes (MILP picks the sizes)."""
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
    g.solve_ilp(verbose=True, time_limit=time_limit, relative_gap=relative_gap)

    mb_sizes = g.get_microbatch_sizes()
    schedule = g.get_schedule()
    if schedule and with_plotting:
        print(f"Microbatch sizes: {mb_sizes} (sum={sum(mb_sizes)})")
        plot_dynamic_schedule(schedule, g, out_name=out_name)

    return schedule, g.get_objective_value()


def solve_crosspipe(with_plotting=True, time_limit=600, relative_gap=0.001,
                    out_name="schedule.png"):
    """Solve with crosspipe fixed durations (equal microbatch sizes)."""
    cfg = CFG.to_crosspipe_system_cfg()
    g = UnidirectionalZBDependencyGraph(cfg)
    g.build_ilp()
    g.solve_ilp(verbose=False, time_limit=time_limit, relative_gap=relative_gap)
    schedule = g.get_schedule()
    if schedule and with_plotting:
        plot_schedule(schedule, cfg, out_name=out_name)

    return schedule, g.get_objective_value()


def solve_dynamic_fixed(fixed_sizes, with_plotting=True, time_limit=600,
                        relative_gap=0.001, out_name="schedule_dynamic_fixed.png"):
    """Solve dynamic-MB MILP with microbatch sizes locked to `fixed_sizes`.
    Only the task ordering is optimised. Useful for evaluating a specific
    size assignment (e.g. the one actually used in the 2-GPU benchmark) under
    the MILP's own cost model."""
    assert len(fixed_sizes) == CFG.nbm, (
        f"expected {CFG.nbm} sizes, got {len(fixed_sizes)}"
    )
    assert sum(fixed_sizes) == CFG.N, (
        f"sizes must sum to N={CFG.N}, got {sum(fixed_sizes)}"
    )

    g = UnidirectionalDynamicBatchSizeZBDependencyGraph(
        system_cfg=CFG.to_system_cfg(),
        N=CFG.N,
        comp=CFG.comp,
        compBias=CFG.compBias,
        comm=CFG.comm,
        commLat=CFG.commLat,
    )
    g.build_ilp()
    # Lock each f_vars[i] to the given size.
    for i, f_val in enumerate(fixed_sizes):
        g.prob += g.prob_f[i] == f_val
    g.solve_ilp(verbose=False, time_limit=time_limit, relative_gap=relative_gap)

    schedule = g.get_schedule()
    if schedule and with_plotting:
        print(f"Microbatch sizes (locked): {fixed_sizes} (sum={sum(fixed_sizes)})")
        plot_dynamic_schedule(schedule, g, out_name=out_name)

    return schedule, g.get_objective_value()


def plot_dynamic_schedule(schedule, graph, out_name="schedule_dynamic.png"):
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
    plt.savefig(os.path.join(OUT_DIR, out_name), dpi=150)


def plot_schedule(schedule, cfg, out_name="schedule.png"):
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
    plt.savefig(os.path.join(OUT_DIR, out_name), dpi=150)

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



def run_baseline():
    """Single-point comparison at CFG's default parameters.

    Prints makespans for three scenarios:
      [1] CrossPipe (equal mb sizes, ZBH1-style baseline)
      [2] Dynamic optimal — MILP jointly picks sizes and ordering
      [3] Dynamic fixed — MILP orders, with sizes locked to the set actually
          used in the 2-GPU benchmark run

    [3] vs [1] tells you the MILP's predicted speedup for the sizes you
    actually ran. [2] vs [3] tells you how much better the MILP thinks it
    could have done if given more solver time / a different gap target.
    """
    CFG.print_summary()
    print()

    actual_sizes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 10, 5, 2, 1]

    print("=" * 72)
    print("[1/3] CrossPipe (equal mb sizes)")
    print("=" * 72)
    _, t_cp = solve_crosspipe(with_plotting=True, out_name="baseline_crosspipe.png")

    print()
    print("=" * 72)
    print("[2/3] Dynamic (MILP picks sizes)")
    print("=" * 72)
    _, t_dyn = solve_dynamic(with_plotting=True, out_name="baseline_dynamic_optimal.png")

    print()
    print("=" * 72)
    print(f"[3/3] Dynamic with sizes locked to {actual_sizes}")
    print("=" * 72)
    _, t_actual = solve_dynamic_fixed(
        actual_sizes,
        with_plotting=True,
        out_name="baseline_dynamic_benchmark_sizes.png",
    )

    def pct(a, b):
        return (b - a) / b * 100 if b else 0.0

    print()
    print("=" * 72)
    print("SUMMARY (makespan in microseconds; MILP objective value)")
    print("=" * 72)
    print(f"  [1] CrossPipe  (equal sizes):         {t_cp:12.2f}")
    print(f"  [2] Dynamic    (optimal sizes):       {t_dyn:12.2f}   "
          f"speedup vs [1] = {pct(t_dyn, t_cp):+.2f}%")
    print(f"  [3] Dynamic    (benchmark sizes):     {t_actual:12.2f}   "
          f"speedup vs [1] = {pct(t_actual, t_cp):+.2f}%")
    print()
    print(f"  Gap between optimal [2] and benchmark sizes [3]: "
          f"{pct(t_dyn, t_actual):+.2f}% (how much [2] beats [3])")
    print()
    print("Interpretation:")
    print("  - [3] vs [1] is the theoretical speedup the MILP expected for the")
    print("    sizes actually used. The measured benchmark was -23%, so the gap")
    print("    between expected and measured is the cost-model error.")
    print("  - [2] vs [3] tells you if the chosen sizes were near-optimal under")
    print("    the MILP's own assumptions (small gap = solver was fine, sizes")
    print("    were good for the model).")


if __name__ == "__main__":
    mode = "baseline"  # "baseline" | "sweep_nbm"

    if mode == "baseline":
        run_baseline()
    elif mode == "sweep_nbm":
        nmb = 2
        list_mb = []
        while nmb < CFG.N:
            list_mb.append(nmb)
            nmb *= 2
        plot_improvement_over_different_microbatches(list_mb)

    plt.show()
