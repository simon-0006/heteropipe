# adapted from https://github.com/sail-sg/zero-bubble-pipeline-parallelism

import numpy as np
import psutil
from typing import Dict, List, Optional, Tuple
from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.pipeline_config import PipelineBlockDesc, SystemConfig
from pulp import LpVariable, LpProblem, LpMinimize, LpStatus, lpSum, value
import pulp
import gurobipy as gp
import scipy.sparse as sp

gurobi_options = {
    "WLSACCESSID": "5dfd5cdf-fbc3-470a-9c8f-1ba62e6e1ff3",
    "WLSSECRET": "af68676f-f7a0-4a53-a5da-94d907ad77f2",
    "LICENSEID": "2790412",
    "THREADS": psutil.cpu_count(logical=False),
}


class DependencyGraph:
    def __init__(self, system_cfg: SystemConfig) -> None:
        self.system_cfg = system_cfg
        self.num_dev = system_cfg.num_devices
        self.num_mb = system_cfg.num_microbatches
        self.num_chunk = system_cfg.num_chunks
        self.nnodes = self.num_dev * self.num_mb * self.num_chunk * 3

        self.inherent_direct_dep: Optional[np.array] = None
        self.inherent_dep: Optional[sp.csr_matrix] = None
        self.prob: Optional[LpProblem] = None
        self.prob_F: Optional[Dict[int, LpVariable]] = None

        for chunk in range(self.num_chunk):
            assert all([isinstance(Tf, (int, np.integer)) for Tf in self.system_cfg.T_F[chunk]])
            assert all([isinstance(Tb, (int, np.integer)) for Tb in self.system_cfg.T_B[chunk]])
            assert all([isinstance(Tw, (int, np.integer)) for Tw in self.system_cfg.T_W[chunk]])
            assert all(
                [
                    f + b + w == 0
                    for f, b, w in zip(
                        self.system_cfg.M_F[chunk], self.system_cfg.M_B[chunk], self.system_cfg.M_W[chunk]
                    )
                ]
            )

    # ID: [dev][mb][task_type]
    def _get_id(self, dev: int, mb: int, task_type: int) -> int:
        raise NotImplementedError

    def _get_dev(self, id: int) -> int:
        raise NotImplementedError

    def _get_mb(self, id: int) -> int:
        raise NotImplementedError

    def _get_task_type(self, id: int) -> int:
        raise NotImplementedError

    def _get_task_time_cost(self, id: int) -> int:
        raise NotImplementedError

    def _get_comm_cost(self, src_dev_id, dst_dev_id) -> int:
        raise NotImplementedError

    def _get_mem_cost(self, id: int) -> int:
        raise NotImplementedError

    def _init_direct_inherent_dependency(self) -> None:
        raise NotImplementedError

    def _init_inherent_dependency(self) -> None:
        # propagated from the direct inherent dependency
        # e.g. the F block of mb 1 mush be finished before the F block of mb 2,3,.. on the same device
        assert self.inherent_dep is None
        assert self.inherent_direct_dep is not None

        adj = sp.lil_matrix((self.nnodes, self.nnodes), dtype=int)
        for i in range(self.nnodes):
            for j in self.inherent_direct_dep[i]:
                adj[j, i] = 1
        adj = adj.tocsr()
        while True:
            new_adj = adj.dot(adj) + adj
            # set nnz to 1
            new_adj.data[:] = 1
            if (adj != new_adj).nnz == 0:
                break
            adj = new_adj
        self.inherent_dep = adj

    def _schedulable_on_dev(self, id_i: int, id_j: int) -> bool:
        # only effective for the same device
        return (
            id_i != id_j
            and self._get_dev(id_i) == self._get_dev(id_j)
            and not self.inherent_dep[id_i, id_j]
            and not self.inherent_dep[id_j, id_i]
        )

    def build_ilp(self) -> None:
        raise NotImplementedError

    def solve_ilp(self, verbose=True, warm_start=False, time_limit=200, relative_gap=0.01) -> None:
        solver = None
        try:
            with gp.Env() as env:
                solver = pulp.GUROBI(
                    mip=True,
                    msg=verbose,
                    warmStart=warm_start,
                    gapRel=relative_gap,
                    MIPGapAbs=1e-6,
                    timeLimit=time_limit,
                    env=env,
                )
                status = self.prob.solve(solver)
                print(f"Status: {LpStatus[status]}")
        except Exception as e:
            print(f"Gurobi failed ({e}), falling back to CBC")
            solver = None
            cbc = pulp.PULP_CBC_CMD(
                msg=verbose,
                timeLimit=time_limit,
                gapRel=relative_gap,
            )
            status = self.prob.solve(cbc)
            print(f"Status: {LpStatus[status]}")
        finally:
            if solver is not None:
                solver.close()

    def get_schedule(self) -> List[List[PipelineBlockDesc]]:
        raise NotImplementedError

    def get_lp_status(self) -> int:
        # if prob is None, return -4
        if self.prob is None:
            return -4
        return self.prob.status

    def get_objective_value(self) -> int:
        if self.prob is None:
            return None
        return pulp.value(self.prob.objective)


class UnidirectionalZBDependencyGraph(DependencyGraph):
    def __init__(self, system_cfg: SystemConfig) -> None:
        super().__init__(system_cfg)

        self._init_direct_inherent_dependency()
        self._init_inherent_dependency()

    # ID: [dev][mb][task_type]
    def _get_id(self, dev: int, mb: int, task_type: int) -> int:
        return dev * self.num_mb * 3 + mb * 3 + task_type

    def _get_dev(self, id: int) -> int:
        return id // (self.num_mb * 3)

    def _get_mb(self, id: int) -> int:
        return (id // 3) % self.num_mb

    def _get_task_type(self, id: int) -> int:
        return id % 3

    def _get_task_time_cost(self, id: int) -> int:
        task_type = self._get_task_type(id)
        dev = self._get_dev(id)
        return [
            self.system_cfg.T_F[0][dev],
            self.system_cfg.T_B[0][dev],
            self.system_cfg.T_W[0][dev],
        ][task_type]

    def _get_comm_cost(self, src_dev_id, dst_dev_id) -> int:
        return self.system_cfg.T_alpha[src_dev_id][dst_dev_id]

    def _get_mem_cost(self, id: int) -> int:
        task_type = self._get_task_type(id)
        dev = self._get_dev(id)
        return [
            self.system_cfg.M_F[0][dev],
            self.system_cfg.M_B[0][dev],
            self.system_cfg.M_W[0][dev],
        ][task_type]

    def _init_direct_inherent_dependency(self) -> None:
        # inherent dependency is the type of dependency that is not affected by the scheduling
        assert self.inherent_direct_dep is None

        parents = []  # parents[id] is a set of direct dependencies of id
        for dev in range(self.num_dev):
            for mb in range(self.num_mb):
                for task_type in range(3):
                    p = set()
                    if task_type == 0:
                        # F block
                        if mb > 0:
                            # prev mb from same device
                            p.add(self._get_id(dev, mb - 1, 0))
                        if dev > 0:
                            # same mb from prev device
                            p.add(self._get_id(dev - 1, mb, 0))
                    elif task_type == 1:
                        # B block
                        if dev == self.num_dev - 1:
                            # last device from corresponding F block
                            p.add(self._get_id(dev, mb, 0))
                        else:
                            # same mb for next device
                            p.add(self._get_id(dev + 1, mb, 1))
                        if mb > 0:
                            # prev mb from same device
                            p.add(self._get_id(dev, mb - 1, 1))
                    elif task_type == 2:
                        # W block
                        # corresponding B block
                        p.add(self._get_id(dev, mb, 1))
                        if mb > 0:
                            # prev mb from same device
                            # not necessary, but shrink the search space
                            p.add(self._get_id(dev, mb - 1, 2))
                    else:
                        raise ValueError("Invalid task type")
                    parents.append(p)
        self.inherent_direct_dep = parents

    def build_ilp(self) -> None:
        prob = LpProblem("AutoSchedule", LpMinimize)

        # dependency order graph
        # P[i][j] = 1 if i is scheduled before j
        # i and j are on the same device
        # schedulable dep as lp variables
        P: Dict[Tuple, LpVariable] = {}
        for i in range(self.nnodes):
            for j in range(i):
                if self._schedulable_on_dev(i, j):
                    P[(i, j)] = LpVariable(f"P_{i}_{j}", 0, 1, cat="Binary")
                    P[(j, i)] = 1 - P[(i, j)]

        # completion time
        F: Dict[int, LpVariable] = LpVariable.dicts(
            "F", (range(self.nnodes),), None, None, cat="Continuous"
        )

        inf = (
            (
                max(self.system_cfg.T_F[0])
                + max(self.system_cfg.T_B[0])
                + max(self.system_cfg.T_W[0])
                + np.max(self.system_cfg.T_alpha) * 3
            )
            * self.num_dev
            * self.num_mb
        )

        # anchor the first task of the 0th device
        first_task = self._get_id(0, 0, 0)
        prob += F[first_task] >= self._get_task_time_cost(first_task)

        M_limits = []

        for i in range(self.nnodes):
            mem_cost = []
            for prev in range(self.nnodes):
                if i == prev:
                    continue
                if prev in self.inherent_direct_dep[i]:
                    # direct dependency, cross device or same device
                    prob += F[i] >= F[prev] + self._get_task_time_cost(i) + (
                        self._get_comm_cost(self._get_dev(prev), self._get_dev(i))
                    )

                if self._get_dev(i) == self._get_dev(prev):
                    if self.inherent_dep[i, prev]:
                        pass
                    elif self.inherent_dep[prev, i]:
                        mem_cost.append(self._get_mem_cost(prev))
                    else:
                        # schedulable dependency
                        prob += (
                            F[i]
                            >= F[prev]
                            + self._get_task_time_cost(i)
                            - inf * P[(i, prev)]
                        )
                        mem_cost.append(self._get_mem_cost(prev) * P[(prev, i)])

            mem_i = lpSum(mem_cost) + self._get_mem_cost(i)
            M_limits.append(mem_i)
            if self.system_cfg.M_Limit[self._get_dev(i)] > 0:
                prob += mem_i <= self.system_cfg.M_Limit[self._get_dev(i)]

        res = LpVariable("res")
        # minimize the maximum completion time
        for i in range(self.nnodes):
            cost_sum = []
            for after in range(self.nnodes):
                if i == after or self._get_dev(i) != self._get_dev(after):
                    continue
                if self.inherent_dep[after, i]:
                    continue
                elif self.inherent_dep[i, after]:
                    cost_sum.append(self._get_task_time_cost(after))
                else:
                    cost_sum.append(self._get_task_time_cost(after) * P[(i, after)])
            dev = self._get_dev(i)
            prob += res >= F[i] + lpSum(cost_sum) - F[
                self._get_id(dev, 0, 0)
            ] + self._get_task_time_cost(self._get_id(dev, 0, 0))

        # for dev in range(self.num_dev):
        #     # Notice: different from paper, we minimize the maximum completion time of the whole pipeline
        #     # instead of the max time range of arbitrary device
        #     # (0,0,0) was anchored
        #     prob += res >= F[self._get_id(dev, self.num_mb - 1, 2)] - F[
        #         self._get_id(0, 0, 0)
        #     ] + self._get_task_time_cost(self._get_id(0, 0, 0))

        for i in range(self.num_dev):
            prob += res >= F[self._get_id(i, self.num_mb - 1, 2)] - F[
                self._get_id(i, 0, 0)
            ] + self._get_task_time_cost(self._get_id(i, 0, 0))

        # Tiebreaker: among equally-optimal makespans, prefer scheduling W ops as early as possible.
        # W has no successors so the main objective is indifferent to their placement.
        eps = 1e-4
        w_penalty = lpSum(
            F[self._get_id(dev, mb, 2)]
            for dev in range(self.num_dev)
            for mb in range(self.num_mb)
        )
        prob.setObjective(res + eps * w_penalty)

        self.prob = prob
        self.prob_F = F

    def get_schedule(self) -> List[List[PipelineBlockDesc]]:
        assert self.prob is not None
        assert self.prob_F is not None

        type_id_to_task = ["F", "B", "W"]
        schedule = [[] for _ in range(self.num_dev)]

        try:
            for dev in range(self.num_dev):
                for mb in range(self.num_mb):
                    for task_type in range(3):
                        task_id = self._get_id(dev, mb, task_type)
                        schedule[dev].append(                               
                            PipelineBlockDesc(
                                device_id=dev,
                                mb_id=mb,
                                task_type=type_id_to_task[task_type],
                                end_time=int(value(self.prob_F[task_id])),
                            )
                        )

            # sort by completion time
            for dev in range(self.num_dev):
                schedule[dev].sort(key=lambda x: x.end_time)
        except Exception as e:
            return None

        return schedule


class UnidirectionalDynamicBatchSizeZBDependencyGraph(UnidirectionalZBDependencyGraph):
    """ZB dependency graph where microbatch sizes are decision variables.

    Task durations: comp[dev][type] * f[mb] + compBias
    Comm durations: comm[dir]      * f[mb] + commLat[src][dst]
    """

    def __init__(
        self,
        system_cfg: SystemConfig,
        N: int,
        comp,       # comp[dev][type]: per-sample compute cost
        compBias,   # fixed overhead per operation
        comm,       # comm[dir]: per-sample comm cost (0=fwd, 1=bwd)
        commLat,    # commLat[dev][dev']: fixed latency between devices
    ):
        # system_cfg is passed with dummy T_F/T_B/T_W to satisfy base assertions;
        # actual durations come from comp/compBias/f[mb].
        self.N = N
        self.comp = comp
        self.compBias = compBias
        self.comm = comm
        self.commLat = commLat
        self.prob_f = None  # will hold solved microbatch size variables

        super().__init__(system_cfg)

    def _task_time_expr(self, node_id, f_vars):
        """Return LP expression for duration of node: comp[dev][type] * f[mb] + compBias."""
        dev = self._get_dev(node_id)
        tt = self._get_task_type(node_id)
        mb = self._get_mb(node_id)
        return self.comp[dev][tt] * f_vars[mb] + self.compBias

    def _comm_cost_expr(self, prev_id, cur_id, f_vars):
        """Return LP expression for comm cost on a cross-device dependency edge.
        Returns 0 for same-device edges."""
        src = self._get_dev(prev_id)
        dst = self._get_dev(cur_id)
        if src == dst:
            return 0

        cur_type = self._get_task_type(cur_id)
        mb = self._get_mb(cur_id)

        # Determine direction: F receives from previous device (fwd), B from next device (bwd)
        if cur_type == 0 and dst == src + 1:
            d = 0  # forward
        elif cur_type == 1 and dst == src - 1:
            d = 1  # backward
        else:
            # fallback: use latency only
            return self.commLat[src][dst]

        return self.comm[d] * f_vars[mb] + self.commLat[src][dst]

    def build_ilp(self) -> None:
        prob = LpProblem("DynamicBatchSchedule", LpMinimize)

        # --- Microbatch size variables ---
        f_vars = [
            LpVariable(f"f_{mb}", lowBound=1, upBound=self.N, cat="Integer")
            for mb in range(self.num_mb)
        ]
        prob += lpSum(f_vars) == self.N

        # --- Ordering variables P[(i,j)] for schedulable pairs ---
        P: Dict[Tuple, LpVariable] = {}
        for i in range(self.nnodes):
            for j in range(i):
                if self._schedulable_on_dev(i, j):
                    P[(i, j)] = LpVariable(f"P_{i}_{j}", 0, 1, cat="Binary")
                    P[(j, i)] = 1 - P[(i, j)]

        # --- Completion time variables ---
        F: Dict[int, LpVariable] = LpVariable.dicts(
            "F", (range(self.nnodes),), None, None, cat="Continuous"
        )

        # Big-M: needs to exceed the max possible difference F[i] - F[prev] - task_time(i).
        # Tighter M = sum of all task durations + comm on a single device (upper bound on span).
        max_comp = max(max(row) for row in self.comp)
        max_comm = max(self.comm)
        max_lat = max(max(row) for row in self.commLat)
        # Each device runs at most 3*num_mb ops; cross-device comm at most num_mb times
        bigM = 3 * self.num_mb * (max_comp * self.N + self.compBias) + self.num_mb * (max_comm * self.N + max_lat)

        # Anchor first task
        first_task = self._get_id(0, 0, 0)
        prob += F[first_task] >= self._task_time_expr(first_task, f_vars)

        # --- Dependency & ordering constraints ---
        for i in range(self.nnodes):
            mem_cost = []
            for prev in range(self.nnodes):
                if i == prev:
                    continue

                if prev in self.inherent_direct_dep[i]:
                    # Direct dependency (same or cross device)
                    prob += (
                        F[i] >= F[prev]
                        + self._task_time_expr(i, f_vars)
                        + self._comm_cost_expr(prev, i, f_vars)
                    )

                if self._get_dev(i) == self._get_dev(prev):
                    if self.inherent_dep[i, prev]:
                        pass
                    elif self.inherent_dep[prev, i]:
                        mem_cost.append(self._get_mem_cost(prev))
                    else:
                        # Schedulable: big-M disjunction (linear — no f*P product)
                        prob += (
                            F[i] >= F[prev]
                            + self._task_time_expr(i, f_vars)
                            - bigM * P[(i, prev)]
                        )
                        mem_cost.append(self._get_mem_cost(prev) * P[(prev, i)])

            mem_i = lpSum(mem_cost) + self._get_mem_cost(i)
            if self.system_cfg.M_Limit[self._get_dev(i)] > 0:
                prob += mem_i <= self.system_cfg.M_Limit[self._get_dev(i)]

        # --- Makespan objective (simple formulation — no bilinear terms) ---
        res = LpVariable("res")
        for dev in range(self.num_dev):
            last_w = self._get_id(dev, self.num_mb - 1, 2)
            first_f = self._get_id(dev, 0, 0)
            prob += (
                res >= F[last_w]
                - F[first_f]
                + self._task_time_expr(first_f, f_vars)
            )

        # Tiebreaker: prefer early W completion
        eps = 1e-4
        w_penalty = lpSum(
            F[self._get_id(dev, mb, 2)]
            for dev in range(self.num_dev)
            for mb in range(self.num_mb)
        )
        prob.setObjective(res + eps * w_penalty)

        self.prob = prob
        self.prob_F = F
        self.prob_f = f_vars

    def get_microbatch_sizes(self) -> List[int]:
        """Return solved microbatch sizes."""
        if self.prob_f is None:
            return None
        return [int(value(fv)) for fv in self.prob_f]

    def get_schedule(self) -> List[List[PipelineBlockDesc]]:
        schedule = super().get_schedule()
        if schedule is None:
            return None
        # Attach solved microbatch sizes as an attribute for plotting
        self._solved_mb_sizes = self.get_microbatch_sizes()
        return schedule
