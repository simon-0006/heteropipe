from collections import defaultdict
from typing import Dict, List, Optional, Tuple
try:
    import docplex.cp.model as cp_model_cplex
    from docplex.cp.config import context as cp_context
except ImportError:
    cp_model_cplex = None
    cp_context = None
from .pipeline_config import PipelineBlockDesc, SystemConfig


class ZBUDCPLEXScheduler:
    def __init__(self, sys_cfg: SystemConfig, warm_start=False) -> None:
        if cp_model_cplex is None:
            raise ImportError("docplex is required for CPLEX schedulers. Install with: pip install docplex")
        self.sys_cfg = sys_cfg
        self.num_dev = sys_cfg.num_devices
        assert sys_cfg.num_chunks == 1
        self.num_chunks = sys_cfg.num_chunks
        self.num_mb = sys_cfg.num_microbatches
        self.zero_1_dp_modeling = sys_cfg.zero_1_dp_modeling
        self.solved = False
        self.cp_context = cp_context
        self.solution: Optional[cp_model_cplex.CpoSolveResult] = None

        # Compute horizon (max time)
        comm_round_trip = sum(
            [
                sys_cfg.T_alpha[i][i + 1] + sys_cfg.T_alpha[i + 1][i]
                for i in range(self.num_dev - 1)
            ]
        ) + sum(
            [
                sys_cfg.T_beta[i][i + 1] + sys_cfg.T_beta[i + 1][i]
                for i in range(self.num_dev - 1)
            ]
        )
        compute_round_trip = sum(
            [
                sys_cfg.T_F[0][i] + sys_cfg.T_B[0][i] + sys_cfg.T_W[0][i]
                for i in range(self.num_dev)
            ]
        )
        dp_comm = sum(
            [
                sys_cfg.T_DP[0][i][0] + sys_cfg.T_DP[0][i][1]
                for i in range(self.num_dev)
            ]
        ) if self.zero_1_dp_modeling else 0
        
        self.horizon = int((comm_round_trip + compute_round_trip + dp_comm) * self.num_mb)

        # Create the model
        self.model = cp_model_cplex.CpoModel()

        # Create variables (mb, task_type, dev) -> interval
        self.task_interval: Dict[
            Tuple[int, int, int], cp_model_cplex.CpoIntervalVar
        ] = {}
        self.memory_usage = [[] for _ in range(self.num_dev)]
        # (mb, src, dst) -> interval
        self.communication_intervals: Dict[
            Tuple[int, int, int], cp_model_cplex.CpoIntervalVar
        ] = {}
        # (dev, chunk, 2) -> interval
        self.dp_comm_interval: Dict[
            Tuple[int, int, int], cp_model_cplex.CpoIntervalVar
        ] = {}
        
        self.task_to_send_comm: Dict[cp_model_cplex.CpoIntervalVar, cp_model_cplex.CpoIntervalVar] = {}

        # Create interval variables
        for dev in range(self.num_dev):
            dev_memory_usage = []
            for mb in range(self.num_mb):
                for task_type in range(3):
                    t_interval = [sys_cfg.T_F[0][dev], sys_cfg.T_B[0][dev], sys_cfg.T_W[0][dev]][
                        task_type
                    ]
                    interval_var = cp_model_cplex.interval_var(
                        size=int(t_interval),
                        name=f"interval_{mb}_{task_type}_{dev}",
                        start=(0, self.horizon),
                    )
                    self.task_interval[(mb, task_type, dev)] = interval_var

                    memory_increase = int(
                        [sys_cfg.M_F[0][dev], sys_cfg.M_B[0][dev], sys_cfg.M_W[0][dev]][
                            task_type
                        ]
                    )
                    if memory_increase > 0:
                        dev_memory_usage.append(
                            cp_model_cplex.step_at_start(interval_var, memory_increase)
                        )
                    elif memory_increase < 0:
                        dev_memory_usage.append(
                            -cp_model_cplex.step_at_end(interval_var, -memory_increase)
                        )

            self.memory_usage[dev] = dev_memory_usage
            self.model.add(sum(dev_memory_usage) <= int(sys_cfg.M_Limit[dev]))

        # communication intervals
        for src in range(self.num_dev - 1):
            dst = src + 1
            if self.sys_cfg.T_beta[src][dst] == 0:
                continue
            for mb in range(self.num_mb):
                interval_var = cp_model_cplex.interval_var(
                    size=int(sys_cfg.T_beta[src][dst]),
                    name=f"comm_interval_{mb}_{src}_{dst}",
                    start=(0, self.horizon),
                )
                self.communication_intervals[(mb, src, dst)] = interval_var
            self.model.add(
                cp_model_cplex.no_overlap(
                    [
                        self.communication_intervals[(mb, src, dst)]
                        for mb in range(self.num_mb)
                    ]
                )
            )
        for src in range(1, self.num_dev):
            dst = src - 1
            if self.sys_cfg.T_beta[src][dst] == 0:
                continue
            for mb in range(self.num_mb):
                interval_var = cp_model_cplex.interval_var(
                    size=int(sys_cfg.T_beta[src][dst]),
                    name=f"comm_interval_{mb}_{src}_{dst}",
                    start=(0, self.horizon),
                )
                self.communication_intervals[(mb, src, dst)] = interval_var
            self.model.add(
                cp_model_cplex.no_overlap(
                    [
                        self.communication_intervals[(mb, src, dst)]
                        for mb in range(self.num_mb)
                    ]
                )
            )
        
        if self.zero_1_dp_modeling:
            for chunk in range(self.num_chunks):
                for dev in range(self.num_dev):
                    interval_var = cp_model_cplex.interval_var(
                        size=int(sys_cfg.T_DP[chunk][dev][0]),
                        name=f"dp_ag_interval_{dev}_{chunk}",
                        start=(0, self.horizon),
                    )
                    self.dp_comm_interval[(dev, chunk, 0)] = interval_var
                    interval_var = cp_model_cplex.interval_var(
                        size=int(sys_cfg.T_DP[chunk][dev][1]),
                        name=f"dp_rs_interval_{dev}_{chunk}",
                        start=(0, self.horizon),
                    )
                    self.dp_comm_interval[(dev, chunk, 1)] = interval_var
            
            for dev in range(self.num_dev):
                cp_model_cplex.no_overlap(
                    [
                        self.dp_comm_interval[(dev, chunk, i)]
                        for chunk in range(self.num_chunks)
                        for i in range(2)
                    ]
                )

        # No overlap constraints
        for dev in range(self.num_dev):
            self.model.add(
                cp_model_cplex.no_overlap(
                    [
                        self.task_interval[(mb, task_type, dev)]
                        for mb in range(self.num_mb)
                        for task_type in range(3)
                    ]
                )
            )

        # On-device Dependencies
        for mb in range(self.num_mb):
            for dev in range(self.num_dev):
                # F -> B -> W
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.task_interval[(mb, 0, dev)],
                        self.task_interval[(mb, 1, dev)],
                    )
                )
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.task_interval[(mb, 1, dev)],
                        self.task_interval[(mb, 2, dev)],
                    )
                )

        # DP comm constraints
        if self.zero_1_dp_modeling:
            for dev in range(self.num_dev):
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.dp_comm_interval[(dev, 0, 0)],
                        self.task_interval[(0, 0, dev)],
                    )
                )
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.task_interval[(self.num_mb - 1, 2, dev)],
                        self.dp_comm_interval[(dev, 0, 1)],
                    )
                )
        
        # Microbatch ordering constraints
        for mb in range(1, self.num_mb):
            for dev in range(self.num_dev):
                for task_type in range(3):
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb - 1, task_type, dev)],
                            self.task_interval[(mb, task_type, dev)],
                        )
                    )

        # Cross-device data dependencies
        for mb in range(self.num_mb):
            for dev in range(self.num_dev - 1):
                # Forward pass
                forward_delay = int(sys_cfg.T_alpha[dev][dev + 1])
                if self.sys_cfg.T_beta[dev][dev + 1] == 0:
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 0, dev)],
                            self.task_interval[(mb, 0, dev + 1)],
                            delay=forward_delay if forward_delay > 0 else None,
                        )
                    )
                else:
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 0, dev)],
                            self.communication_intervals[(mb, dev, dev + 1)],
                        )
                    )
                    self.task_to_send_comm[self.task_interval[(mb, 0, dev)]] = self.communication_intervals[(mb, dev, dev + 1)]
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.communication_intervals[(mb, dev, dev + 1)],
                            self.task_interval[(mb, 0, dev + 1)],
                            delay=forward_delay if forward_delay > 0 else None,
                        )
                    )
                # Backward pass
                backward_delay = int(sys_cfg.T_alpha[dev + 1][dev])
                if self.sys_cfg.T_beta[dev + 1][dev] == 0:
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 1, dev + 1)],
                            self.task_interval[(mb, 1, dev)],
                            delay=backward_delay if backward_delay > 0 else None,
                        )
                    )
                else:
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 1, dev + 1)],
                            self.communication_intervals[(mb, dev + 1, dev)],
                        )
                    )
                    self.task_to_send_comm[self.task_interval[(mb, 1, dev + 1)]] = self.communication_intervals[(mb, dev + 1, dev)]
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.communication_intervals[(mb, dev + 1, dev)],
                            self.task_interval[(mb, 1, dev)],
                            delay=backward_delay if backward_delay > 0 else None,
                        )
                    )

        if self.zero_1_dp_modeling:
            self.model.add(cp_model_cplex.start_of(self.dp_comm_interval[(0, 0, 0)]) == 0)
        else:
            # First task starts at time 0
            self.model.add(cp_model_cplex.start_of(self.task_interval[(0, 0, 0)]) == 0)

        # Objective variables
        self.dev_span = cp_model_cplex.integer_var(0, self.horizon, name="dev_span")
        self.bubble = cp_model_cplex.integer_var(0, self.horizon, name="bubble")

        if self.zero_1_dp_modeling:
            self.model.add(cp_model_cplex.max(
                [
                    cp_model_cplex.end_of(self.dp_comm_interval[(dev, 0, 1)])
                    - cp_model_cplex.start_of(self.dp_comm_interval[(dev, 0, 0)])
                    for dev in range(self.num_dev)
                ]
            ) == self.dev_span)
        else:
            # Maximum completion time across all devices
            self.model.add(
                cp_model_cplex.max(
                    [
                        cp_model_cplex.end_of(self.task_interval[(self.num_mb - 1, 2, dev)])
                        - cp_model_cplex.start_of(self.task_interval[(0, 0, dev)])
                        for dev in range(self.num_dev)
                    ]
                )
                == self.dev_span
            )

        # Bubble time calculation
        bubble_terms = [
            cp_model_cplex.end_of(self.task_interval[(self.num_mb - 1, 2, dev)])
            - cp_model_cplex.start_of(self.task_interval[(0, 0, dev)])
            - self.num_mb * int(sys_cfg.T_F[0][dev] + sys_cfg.T_B[0][dev] + sys_cfg.T_W[0][dev])
            for dev in range(self.num_dev)
        ]
        self.model.add(cp_model_cplex.max(bubble_terms) == self.bubble)

        # Set the objective
        self.model.minimize(self.dev_span)
        # self.model.add(cp_model_cplex.minimize_static_lex([self.dev_span, self.bubble]))

        if warm_start:
            self.set_hint_schedule()

    def set_hint_schedule(self):
        """Set warm start solution using heuristic schedule"""
        from pipeline import HeuristicZBUDPipeline

        # Generate heuristic schedule
        hint_pp = HeuristicZBUDPipeline(self.sys_cfg)
        hint_pp.schedule()
        hint_pp.solve_dependencies()
        # hint_pp.print_schedule(save=True, name="hint_schedule")
        dev_span_hint = hint_pp.get_schedule_time(device_wise=True)
        dev_tasks = hint_pp.device_scheduled_tasks

        # Create a warm start solution
        warm_start_solution = self.model.create_empty_solution()

        # Add interval variable assignments
        for dev in range(self.num_dev):
            for task in dev_tasks[dev]:
                str_to_type = {"F": 0, "B": 1, "W": 2}
                task_type = str_to_type[task.task_type]
                mb_id = task.microbatch_id

                interval_var = self.task_interval[(mb_id, task_type, dev)]
                warm_start_solution.add_interval_var_solution(
                    interval_var,
                    presence=True,
                    start=int(task.start_time),
                    end=int(task.completion_time),
                )

        # Add objective variable hints
        warm_start_solution.add_integer_var_solution(self.dev_span, int(dev_span_hint))

        # Set the warm start solution
        self.model.set_starting_point(warm_start_solution)

    def solve(self, logging=False, time_limit_sec=600, relative_gap=0.0001, workers=64):
        # Configure solver parameters
        self.cp_context.params.TimeLimit = time_limit_sec
        self.cp_context.params.Workers = workers
        self.cp_context.params.LogVerbosity = "Terse" if logging else "Quiet"
        self.cp_context.params.LogPeriod = 10000 if logging else 1000
        self.cp_context.params.RelativeOptimalityTolerance = relative_gap

        # Solve the model
        solution: cp_model_cplex.CpoSolveResult = self.model.solve()
        if solution:
            self.solved = True
            self.solution = solution
            print(f"Solution status: {solution.get_solve_status()}")
        else:
            print("No solution found")

    def get_schedule(self) -> List[List[PipelineBlockDesc]]:
        if not self.solved:
            raise ValueError("Model not solved yet")

        schedule = [[] for _ in range(self.num_dev)]
        for dev in range(self.num_dev):
            for mb in range(self.num_mb):
                for task_type in range(3):
                    end_time = self.solution.get_value(
                        self.task_interval[(mb, task_type, dev)]
                    )[1]
                    post_send_time = -1
                    if self.task_interval[(mb, task_type, dev)] in self.task_to_send_comm:
                        post_send_time = self.solution.get_value(
                            self.task_to_send_comm[self.task_interval[(mb, task_type, dev)]]
                        )[0]
                    schedule[dev].append(
                        PipelineBlockDesc(
                            device_id=dev,
                            mb_id=mb,
                            task_type=["F", "B", "W"][task_type],
                            end_time=end_time,
                            post_send_time=post_send_time,
                        )
                    )
            schedule[dev].sort(key=lambda x: x.end_time)
        return schedule



class ZBWaveCPLEXScheduler:
    def __init__(self, sys_cfg: SystemConfig, warm_start=False) -> None:
        if cp_model_cplex is None:
            raise ImportError("docplex is required for CPLEX schedulers. Install with: pip install docplex")
        self.sys_cfg = sys_cfg
        self.num_dev = sys_cfg.num_devices
        assert sys_cfg.num_chunks == 2
        self.num_chunks = sys_cfg.num_chunks
        self.num_mb = sys_cfg.num_microbatches
        self.zero_1_dp_modeling = sys_cfg.zero_1_dp_modeling
        self.solved = False
        self.cp_context = cp_context
        self.solution: Optional[cp_model_cplex.CpoSolveResult] = None

        # Compute horizon (max time)
        comm_round_trip = sum(
            [
                sys_cfg.T_alpha[i][i + 1] + sys_cfg.T_alpha[i + 1][i]
                for i in range(self.num_dev - 1)
            ]
        ) + sum(
            [
                sys_cfg.T_beta[i][i + 1] + sys_cfg.T_beta[i + 1][i]
                for i in range(self.num_dev - 1)
            ]
        )
        compute_round_trip = sum(
            [
                sys_cfg.T_F[0][i] + sys_cfg.T_B[0][i] + sys_cfg.T_W[0][i]
                for i in range(self.num_dev)
            ]
        )
        dp_comm = sum(
            [
                sys_cfg.T_DP[i][j][0] + sys_cfg.T_DP[i][j][1]
                for i in range(self.num_chunks)
                for j in range(self.num_dev)
            ]
        ) if self.zero_1_dp_modeling else 0
        
        self.horizon = 2 * int((comm_round_trip + compute_round_trip + dp_comm) * self.num_mb)

        # Create the model
        self.model = cp_model_cplex.CpoModel()

        # Create variables (similar structure but using docplex syntax)
        # (mb, task_type, chunk, dev) -> interval
        self.task_interval: Dict[
            Tuple[int, int, int, int], cp_model_cplex.CpoIntervalVar
        ] = {}
        self.memory_usage = [[] for _ in range(self.num_dev)]
        # (mb, chunk, src, dst) -> interval
        self.communication_intervals: Dict[
            Tuple[int, int, int, int], cp_model_cplex.CpoIntervalVar
        ] = {}
        # (dev, chunk, 2) -> interval
        self.dp_comm_interval: Dict[
            Tuple[int, int, int], cp_model_cplex.CpoIntervalVar
        ] = {}
        self.task_to_send_comm: Dict[cp_model_cplex.CpoIntervalVar, cp_model_cplex.CpoIntervalVar] = {}

        # Create interval variables
        for dev in range(self.num_dev):
            dev_memory_usage = []
            for mb in range(self.num_mb):
                for task_type in range(3):
                    for chunk in range(2):
                        t_interval = [
                            sys_cfg.T_F[chunk][dev],
                            sys_cfg.T_B[chunk][dev],
                            sys_cfg.T_W[chunk][dev],
                        ][task_type]
                        interval_var = cp_model_cplex.interval_var(
                            size=int(t_interval),
                            name=f"interval_{mb}_{task_type}_{chunk}_{dev}",
                            start=(0, self.horizon),
                        )
                        self.task_interval[(mb, task_type, chunk, dev)] = interval_var

                        memory_increase = int(
                            [sys_cfg.M_F[chunk][dev], sys_cfg.M_B[chunk][dev], sys_cfg.M_W[chunk][dev]][
                                task_type
                            ]
                        )
                        if memory_increase > 0:
                            dev_memory_usage.append(
                                cp_model_cplex.step_at_start(
                                    interval_var, memory_increase
                                )
                            )
                        elif memory_increase < 0:
                            dev_memory_usage.append(
                                -cp_model_cplex.step_at_end(
                                    interval_var, -memory_increase
                                )
                            )

            self.memory_usage[dev] = dev_memory_usage
            self.model.add(sum(dev_memory_usage) <= int(sys_cfg.M_Limit[dev]))

        # communication intervals
        for src in range(self.num_dev - 1):
            dst = src + 1
            if self.sys_cfg.T_beta[src][dst] == 0:
                continue
            for chunk in range(2):
                for mb in range(self.num_mb):
                    interval_var = cp_model_cplex.interval_var(
                        size=int(sys_cfg.T_beta[src][dst]),
                        name=f"comm_interval_{mb}_{chunk}_{src}_{dst}",
                        start=(0, self.horizon),
                    )
                    self.communication_intervals[(mb, chunk, src, dst)] = interval_var
            self.model.add(
                cp_model_cplex.no_overlap(
                    [
                        self.communication_intervals[(mb, chunk, src, dst)]
                        for mb in range(self.num_mb)
                        for chunk in range(2)
                    ]
                )
            )
        for src in range(1, self.num_dev):
            dst = src - 1
            if self.sys_cfg.T_beta[src][dst] == 0:
                continue
            for chunk in range(2):
                for mb in range(self.num_mb):
                    interval_var = cp_model_cplex.interval_var(
                        size=int(sys_cfg.T_beta[src][dst]),
                        name=f"comm_interval_{mb}_{chunk}_{src}_{dst}",
                        start=(0, self.horizon),
                    )
                    self.communication_intervals[(mb, chunk, src, dst)] = interval_var
            self.model.add(
                cp_model_cplex.no_overlap(
                    [
                        self.communication_intervals[(mb, chunk, src, dst)]
                        for mb in range(self.num_mb)
                        for chunk in range(2)
                    ]
                )
            )

        if self.zero_1_dp_modeling:
            for chunk in range(2):
                for dev in range(self.num_dev):
                    interval_var = cp_model_cplex.interval_var(
                        size=int(sys_cfg.T_DP[chunk][dev][0]),
                        name=f"dp_ag_interval_{dev}_{chunk}",
                        start=(0, self.horizon),
                    )
                    self.dp_comm_interval[(dev, chunk, 0)] = interval_var
                    interval_var = cp_model_cplex.interval_var(
                        size=int(sys_cfg.T_DP[chunk][dev][1]),
                        name=f"dp_rs_interval_{dev}_{chunk}",
                        start=(0, self.horizon),
                    )
                    self.dp_comm_interval[(dev, chunk, 1)] = interval_var
            
            for dev in range(self.num_dev):
                cp_model_cplex.no_overlap(
                    [
                        self.dp_comm_interval[(dev, chunk, i)]
                        for chunk in range(2)
                        for i in range(2)
                    ]
                )

        # No overlap constraints
        for dev in range(self.num_dev):
            self.model.add(
                cp_model_cplex.no_overlap(
                    [
                        self.task_interval[(mb, task_type, chunk, dev)]
                        for mb in range(self.num_mb)
                        for task_type in range(3)
                        for chunk in range(2)
                    ]
                )
            )

        # On-device Dependencies
        for mb in range(self.num_mb):
            for dev in range(self.num_dev):
                # F0 -> F1 -> B1 -> B0 -> W
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.task_interval[(mb, 0, 0, dev)],
                        self.task_interval[(mb, 0, 1, dev)],
                    )
                )
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.task_interval[(mb, 0, 1, dev)],
                        self.task_interval[(mb, 1, 1, dev)],
                    )
                )
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.task_interval[(mb, 1, 1, dev)],
                        self.task_interval[(mb, 1, 0, dev)],
                    )
                )
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.task_interval[(mb, 1, 1, dev)],
                        self.task_interval[(mb, 2, 1, dev)],
                    )
                )
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.task_interval[(mb, 1, 0, dev)],
                        self.task_interval[(mb, 2, 0, dev)],
                    )
                )

        # DP comm constraints
        if self.zero_1_dp_modeling:
            for dev in range(self.num_dev):
                for chunk in range(2):
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.dp_comm_interval[(dev, chunk, 0)],
                            self.task_interval[(0, 0, chunk, dev)],
                        )
                    )
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(self.num_mb - 1, 2, chunk, dev)],
                            self.dp_comm_interval[(dev, chunk, 1)],
                        )
                    )

        # Microbatch ordering constraints
        for mb in range(1, self.num_mb):
            for dev in range(self.num_dev):
                for task_type in range(3):
                    for chunk in range(2):
                        self.model.add(
                            cp_model_cplex.end_before_start(
                                self.task_interval[(mb - 1, task_type, chunk, dev)],
                                self.task_interval[(mb, task_type, chunk, dev)],
                            )
                        )

        # Cross-device data dependencies
        for mb in range(self.num_mb):
            for dev in range(self.num_dev - 1):
                # F0 & B1
                forward_delay = int(sys_cfg.T_alpha[dev][dev + 1])
                if self.sys_cfg.T_beta[dev][dev + 1] == 0:
                    # F0
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 0, 0, dev)],
                            self.task_interval[(mb, 0, 0, dev + 1)],
                            delay=forward_delay if forward_delay > 0 else None,
                        )
                    )
                    # B1
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 1, 1, dev)],
                            self.task_interval[(mb, 1, 1, dev + 1)],
                            delay=forward_delay if forward_delay > 0 else None,
                        )
                    )
                else:
                    # F0
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 0, 0, dev)],
                            self.communication_intervals[(mb, 0, dev, dev + 1)],
                        )
                    )
                    self.task_to_send_comm[self.task_interval[(mb, 0, 0, dev)]] = self.communication_intervals[(mb, 0, dev, dev + 1)]
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.communication_intervals[(mb, 0, dev, dev + 1)],
                            self.task_interval[(mb, 0, 0, dev + 1)],
                            delay=forward_delay if forward_delay > 0 else None,
                        )
                    )
                    # B1
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 1, 1, dev)],
                            self.communication_intervals[(mb, 1, dev, dev + 1)],
                        )
                    )
                    self.task_to_send_comm[self.task_interval[(mb, 1, 1, dev)]] = self.communication_intervals[(mb, 1, dev, dev + 1)]
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.communication_intervals[(mb, 1, dev, dev + 1)],
                            self.task_interval[(mb, 1, 1, dev + 1)],
                            delay=forward_delay if forward_delay > 0 else None,
                        )
                    )
                # F1 & B0
                backward_delay = int(sys_cfg.T_alpha[dev + 1][dev])
                if self.sys_cfg.T_beta[dev + 1][dev] == 0:
                    # F1
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 0, 1, dev + 1)],
                            self.task_interval[(mb, 0, 1, dev)],
                            delay=backward_delay if backward_delay > 0 else None,
                        )
                    )
                    # B0
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 1, 0, dev + 1)],
                            self.task_interval[(mb, 1, 0, dev)],
                            delay=backward_delay if backward_delay > 0 else None,
                        )
                    )
                else:
                    # F1
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 0, 1, dev + 1)],
                            self.communication_intervals[(mb, 1, dev + 1, dev)],
                        )
                    )
                    self.task_to_send_comm[self.task_interval[(mb, 0, 1, dev + 1)]] = self.communication_intervals[(mb, 1, dev + 1, dev)]
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.communication_intervals[(mb, 1, dev + 1, dev)],
                            self.task_interval[(mb, 0, 1, dev)],
                            delay=backward_delay if backward_delay > 0 else None,
                        )
                    )
                    # B0
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 1, 0, dev + 1)],
                            self.communication_intervals[(mb, 0, dev + 1, dev)],
                        )
                    )
                    self.task_to_send_comm[self.task_interval[(mb, 1, 0, dev + 1)]] = self.communication_intervals[(mb, 0, dev + 1, dev)]
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.communication_intervals[(mb, 0, dev + 1, dev)],
                            self.task_interval[(mb, 1, 0, dev)],
                            delay=backward_delay if backward_delay > 0 else None,
                        )
                    )

        if self.zero_1_dp_modeling:
            self.model.add(cp_model_cplex.start_of(self.dp_comm_interval[(0, 0, 0)]) == 0)
        else:
            # First task starts at time 0
            self.model.add(cp_model_cplex.start_of(self.task_interval[(0, 0, 0, 0)]) == 0)

        # Objective variables
        self.dev_span = cp_model_cplex.integer_var(0, self.horizon, name="dev_span")
        self.bubble = cp_model_cplex.integer_var(0, self.horizon, name="bubble")

        if self.zero_1_dp_modeling:
            self.model.add(cp_model_cplex.max(
                [
                    cp_model_cplex.end_of(self.dp_comm_interval[(dev, chunk, 1)])
                    - cp_model_cplex.start_of(self.dp_comm_interval[(dev, 0, 0)])
                    for dev in range(self.num_dev)
                    for chunk in range(2)
                ]
            ) == self.dev_span)
        else:
            # Maximum completion time across all devices
            self.model.add(
                cp_model_cplex.max(
                    [
                        cp_model_cplex.end_of(
                            self.task_interval[(self.num_mb - 1, 2, 0, dev)]
                        )
                        - cp_model_cplex.start_of(self.task_interval[(0, 0, 0, dev)])
                        for dev in range(self.num_dev)
                    ]
                )
                == self.dev_span
            )

        # Bubble time calculation
        bubble_terms = [
            cp_model_cplex.end_of(self.task_interval[(self.num_mb - 1, 2, 0, dev)])
            - cp_model_cplex.start_of(self.task_interval[(0, 0, 0, dev)])
            - self.num_mb
            * int(sum([sys_cfg.T_F[chunk][dev] + sys_cfg.T_B[chunk][dev] + sys_cfg.T_W[chunk][dev] for chunk in range(2)]))
            for dev in range(self.num_dev)
        ]
        self.model.add(cp_model_cplex.max(bubble_terms) == self.bubble)

        # Set the objective
        self.model.minimize(self.dev_span)
        # self.model.add(cp_model_cplex.minimize_static_lex([self.dev_span, self.bubble]))

        if warm_start:
            raise NotImplementedError(
                "Warm start not implemented for ZBWaveCPLEXScheduler"
            )

    def solve(self, logging=False, time_limit_sec=600, relative_gap=0.0001, workers=64):
        # Configure solver parameters
        self.cp_context.params.TimeLimit = time_limit_sec
        self.cp_context.params.Workers = workers
        self.cp_context.params.LogVerbosity = "Terse" if logging else "Quiet"
        self.cp_context.params.RelativeOptimalityTolerance = relative_gap

        # Solve the model
        solution: cp_model_cplex.CpoSolveResult = self.model.solve()
        if solution:
            self.solved = True
            self.solution = solution
            print(f"Solution status: {solution.get_solve_status()}")
        else:
            print("No solution found")

    def get_schedule(self) -> List[List[PipelineBlockDesc]]:
        if not self.solved:
            raise ValueError("Model not solved yet")

        schedule = [[] for _ in range(self.num_dev)]
        for dev in range(self.num_dev):
            for mb in range(self.num_mb):
                for task_type in range(3):
                    for chunk in range(2):
                        start_time = self.solution.get_value(
                            self.task_interval[(mb, task_type, chunk, dev)]
                        )[0]
                        end_time = self.solution.get_value(
                            self.task_interval[(mb, task_type, chunk, dev)]
                        )[1]
                        post_send_time = -1
                        if self.task_interval[(mb, task_type, chunk, dev)] in self.task_to_send_comm:
                            post_send_time = self.solution.get_value(
                                self.task_to_send_comm[self.task_interval[(mb, task_type, chunk, dev)]]
                            )[0]
                        schedule[dev].append(
                            PipelineBlockDesc(
                                device_id=dev,
                                mb_id=mb,
                                task_type=["F", "B", "W"][task_type],
                                chunk_id=chunk,
                                start_time=start_time,
                                end_time=end_time,
                                post_send_time=post_send_time,
                            )
                        )
            schedule[dev].sort(key=lambda x: x.end_time)
        return schedule


class ZBLoopCPLEXScheduler:
    def __init__(self, sys_cfg: SystemConfig, warm_start=False) -> None:
        if cp_model_cplex is None:
            raise ImportError("docplex is required for CPLEX schedulers. Install with: pip install docplex")
        self.sys_cfg = sys_cfg
        self.num_dev = sys_cfg.num_devices
        assert self.num_dev > 2, "currently only support more than 2 devices"
        assert sys_cfg.num_chunks == 2
        self.num_chunks = sys_cfg.num_chunks
        self.num_mb = sys_cfg.num_microbatches
        self.zero_1_dp_modeling = sys_cfg.zero_1_dp_modeling
        self.solved = False
        self.cp_context = cp_context
        self.solution: Optional[cp_model_cplex.CpoSolveResult] = None
        
        # Compute horizon (max time)
        comm_round_trip = sum(
            [
                sys_cfg.T_alpha[i][i + 1] + sys_cfg.T_alpha[i + 1][i]
                for i in range(self.num_dev - 1)
            ]
        ) + sum(
            [
                sys_cfg.T_beta[i][i + 1] + sys_cfg.T_beta[i + 1][i]
                for i in range(self.num_dev - 1)
            ]
        )
        compute_round_trip = sum(
            [
                sys_cfg.T_F[chunk][i] + sys_cfg.T_B[chunk][i] + sys_cfg.T_W[chunk][i]
                for chunk in range(self.num_chunks)
                for i in range(self.num_dev)
            ]
        )
        dp_comm = sum(
            [
                sys_cfg.T_DP[0][i][0] + sys_cfg.T_DP[0][i][1]
                for i in range(self.num_dev)
            ]
        ) if self.zero_1_dp_modeling else 0
        
        self.horizon = 4 * int((comm_round_trip + compute_round_trip + dp_comm) * self.num_mb)

        # Create the model
        self.model = cp_model_cplex.CpoModel()

        # Create variables (similar structure but using docplex syntax)
        # (mb, task_type, chunk, dev) -> interval
        self.task_interval: Dict[
            Tuple[int, int, int, int], cp_model_cplex.CpoIntervalVar
        ] = {}
        self.memory_usage = [[] for _ in range(self.num_dev)]
        # (mb, chunk(sender), src, dst) -> interval
        self.communication_intervals: Dict[
            Tuple[int, int, int, int], cp_model_cplex.CpoIntervalVar
        ] = {}
        # (dev, chunk, 2) -> interval
        self.dp_comm_interval: Dict[
            Tuple[int, int, int], cp_model_cplex.CpoIntervalVar
        ] = {}
        self.task_to_send_comm: Dict[cp_model_cplex.CpoIntervalVar, cp_model_cplex.CpoIntervalVar] = {}

        # Create interval variables
        for dev in range(self.num_dev):
            dev_memory_usage = []
            for mb in range(self.num_mb):
                for task_type in range(3):
                    for chunk in range(2):
                        t_interval = [
                            sys_cfg.T_F[chunk][dev],
                            sys_cfg.T_B[chunk][dev],
                            sys_cfg.T_W[chunk][dev],
                        ][task_type]
                        interval_var = cp_model_cplex.interval_var(
                            size=int(t_interval),
                            name=f"interval_{mb}_{task_type}_{chunk}_{dev}",
                            start=(0, self.horizon),
                        )
                        self.task_interval[(mb, task_type, chunk, dev)] = interval_var

                        memory_increase = int(
                            [sys_cfg.M_F[chunk][dev], sys_cfg.M_B[chunk][dev], sys_cfg.M_W[chunk][dev]][
                                task_type
                            ]
                        )
                        if memory_increase > 0:
                            dev_memory_usage.append(
                                cp_model_cplex.step_at_start(
                                    interval_var, memory_increase
                                )
                            )
                        elif memory_increase < 0:
                            dev_memory_usage.append(
                                -cp_model_cplex.step_at_end(
                                    interval_var, -memory_increase
                                )
                            )

            self.memory_usage[dev] = dev_memory_usage
            self.model.add(sum(dev_memory_usage) <= int(sys_cfg.M_Limit[dev]))

        # communication intervals
        # forward
        non_overlap_intervals_prev_to_next = defaultdict(list) # prev -> list
        non_overlap_intervals_next_to_prev = defaultdict(list) # next -> list
        for src in range(self.num_dev):
            dst = (src + 1) % self.num_dev
            if self.sys_cfg.T_beta[src][dst] == 0:
                continue
            for chunk in range(2):
                if chunk == 1 and src == self.num_dev - 1 and dst == 0:
                    continue
                for mb in range(self.num_mb):
                    interval_var = cp_model_cplex.interval_var(
                        size=int(sys_cfg.T_beta[src][dst]),
                        name=f"comm_interval_{mb}_{chunk}_{src}_{dst}",
                        start=(0, self.horizon),
                    )
                    self.communication_intervals[(mb, chunk, src, dst)] = (
                        interval_var
                    )
                    non_overlap_intervals_prev_to_next[src].append(interval_var)
        
        for src in range(self.num_dev):
            dst = (src - 1 + self.num_dev) % self.num_dev
            if self.sys_cfg.T_beta[src][dst] == 0:
                continue
            for chunk in range(2):
                if chunk == 0 and src == 0 and dst == self.num_dev - 1:
                    continue
                for mb in range(self.num_mb):
                    interval_var = cp_model_cplex.interval_var(
                        size=int(sys_cfg.T_beta[src][dst]),
                        name=f"comm_interval_{mb}_{chunk}_{src}_{dst}",
                        start=(0, self.horizon),
                    )
                    self.communication_intervals[(mb, chunk, src, dst)] = interval_var
                    non_overlap_intervals_next_to_prev[dst].append(interval_var)
        
        if self.sys_cfg.two_dc:
            # multiplex communication: 0<-> num_dev - 1 and num_dev/2 <-> num_dev/2 + 1
            non_overlap_intervals_next_to_prev[self.num_dev // 2].extend(non_overlap_intervals_prev_to_next[self.num_dev - 1])
            del non_overlap_intervals_prev_to_next[self.num_dev - 1]
            non_overlap_intervals_prev_to_next[self.num_dev // 2 - 1].extend(non_overlap_intervals_next_to_prev[0])
            del non_overlap_intervals_next_to_prev[0]
        
        for _, intervals in non_overlap_intervals_prev_to_next.items():
            if len(intervals) < 2:
                continue                
            self.model.add(cp_model_cplex.no_overlap(intervals))
        for _, intervals in non_overlap_intervals_next_to_prev.items():
            if len(intervals) < 2:
                continue                
            self.model.add(cp_model_cplex.no_overlap(intervals))

        if self.zero_1_dp_modeling:
            for chunk in range(2):
                for dev in range(self.num_dev):
                    interval_var = cp_model_cplex.interval_var(
                        size=int(sys_cfg.T_DP[chunk][dev][0]),
                        name=f"dp_ag_interval_{dev}_{chunk}",
                        start=(0, self.horizon),
                    )
                    self.dp_comm_interval[(dev, chunk, 0)] = interval_var
                    interval_var = cp_model_cplex.interval_var(
                        size=int(sys_cfg.T_DP[chunk][dev][1]),
                        name=f"dp_rs_interval_{dev}_{chunk}",
                        start=(0, self.horizon),
                    )
                    self.dp_comm_interval[(dev, chunk, 1)] = interval_var
            
            for dev in range(self.num_dev):
                cp_model_cplex.no_overlap(
                    [
                        self.dp_comm_interval[(dev, chunk, i)]
                        for chunk in range(2)
                        for i in range(2)
                    ]
                )

        # No overlap constraints
        for dev in range(self.num_dev):
            self.model.add(
                cp_model_cplex.no_overlap(
                    [
                        self.task_interval[(mb, task_type, chunk, dev)]
                        for mb in range(self.num_mb)
                        for task_type in range(3)
                        for chunk in range(2)
                    ]
                )
            )

        # On-device Dependencies
        for mb in range(self.num_mb):
            for dev in range(self.num_dev):
                # F0 -> F1 -> B1 -> B0 -> W
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.task_interval[(mb, 0, 0, dev)],
                        self.task_interval[(mb, 0, 1, dev)],
                    )
                )
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.task_interval[(mb, 0, 1, dev)],
                        self.task_interval[(mb, 1, 1, dev)],
                    )
                )
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.task_interval[(mb, 1, 1, dev)],
                        self.task_interval[(mb, 1, 0, dev)],
                    )
                )
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.task_interval[(mb, 1, 1, dev)],
                        self.task_interval[(mb, 2, 1, dev)],
                    )
                )
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.task_interval[(mb, 1, 0, dev)],
                        self.task_interval[(mb, 2, 0, dev)],
                    )
                )

        # DP comm constraints
        if self.zero_1_dp_modeling:
            for dev in range(self.num_dev):
                for chunk in range(2):
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.dp_comm_interval[(dev, chunk, 0)],
                            self.task_interval[(0, 0, chunk, dev)],
                        )
                    )
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(self.num_mb - 1, 2, chunk, dev)],
                            self.dp_comm_interval[(dev, chunk, 1)],
                        )
                    )

        # Microbatch ordering constraints
        for mb in range(1, self.num_mb):
            for dev in range(self.num_dev):
                for task_type in range(3):
                    for chunk in range(2):
                        self.model.add(
                            cp_model_cplex.end_before_start(
                                self.task_interval[(mb - 1, task_type, chunk, dev)],
                                self.task_interval[(mb, task_type, chunk, dev)],
                            )
                        )

        # Cross-device data dependencies
        for mb in range(self.num_mb):
            for dev in range(self.num_dev - 1):
                # F0 & F1
                forward_delay = int(sys_cfg.T_alpha[dev][dev + 1])
                if self.sys_cfg.T_beta[dev][dev + 1] == 0:
                    # F0
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 0, 0, dev)],
                            self.task_interval[(mb, 0, 0, dev + 1)],
                            delay=forward_delay if forward_delay > 0 else None,
                        )
                    )
                    # F1
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 0, 1, dev)],
                            self.task_interval[(mb, 0, 1, dev + 1)],
                            delay=forward_delay if forward_delay > 0 else None,
                        )
                    )
                else:
                    # F0
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 0, 0, dev)],
                            self.communication_intervals[(mb, 0, dev, dev + 1)],
                        )
                    )
                    self.task_to_send_comm[self.task_interval[(mb, 0, 0, dev)]] = self.communication_intervals[(mb, 0, dev, dev + 1)]
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.communication_intervals[(mb, 0, dev, dev + 1)],
                            self.task_interval[(mb, 0, 0, dev + 1)],
                            delay=forward_delay if forward_delay > 0 else None,
                        )
                    )
                    # F1
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 0, 1, dev)],
                            self.communication_intervals[(mb, 1, dev, dev + 1)],
                        )
                    )
                    self.task_to_send_comm[self.task_interval[(mb, 0, 1, dev)]] = self.communication_intervals[(mb, 1, dev, dev + 1)]
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.communication_intervals[(mb, 1, dev, dev + 1)],
                            self.task_interval[(mb, 0, 1, dev + 1)],
                            delay=forward_delay if forward_delay > 0 else None,
                        )
                    )
                # B1 & B0
                backward_delay = int(sys_cfg.T_alpha[dev + 1][dev])
                if self.sys_cfg.T_beta[dev + 1][dev] == 0:
                    # B1
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 1, 1, dev + 1)],
                            self.task_interval[(mb, 1, 1, dev)],
                            delay=backward_delay if backward_delay > 0 else None,
                        )
                    )
                    # B0
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 1, 0, dev + 1)],
                            self.task_interval[(mb, 1, 0, dev)],
                            delay=backward_delay if backward_delay > 0 else None,
                        )
                    )
                else:
                    # B1
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 1, 1, dev + 1)],
                            self.communication_intervals[(mb, 1, dev + 1, dev)],
                        )
                    )
                    self.task_to_send_comm[self.task_interval[(mb, 1, 1, dev + 1)]] = self.communication_intervals[(mb, 1, dev + 1, dev)]
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.communication_intervals[(mb, 1, dev + 1, dev)],
                            self.task_interval[(mb, 1, 1, dev)],
                            delay=backward_delay if backward_delay > 0 else None,
                        )
                    )
                    # B0
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.task_interval[(mb, 1, 0, dev + 1)],
                            self.communication_intervals[(mb, 0, dev + 1, dev)],
                        )
                    )
                    self.task_to_send_comm[self.task_interval[(mb, 1, 0, dev + 1)]] = self.communication_intervals[(mb, 0, dev + 1, dev)]
                    self.model.add(
                        cp_model_cplex.end_before_start(
                            self.communication_intervals[(mb, 0, dev + 1, dev)],
                            self.task_interval[(mb, 1, 0, dev)],
                            delay=backward_delay if backward_delay > 0 else None,
                        )
                    )
        # loop back link
        for mb in range(self.num_mb):
            first = 0
            last = self.num_dev - 1
            forward_delay = int(sys_cfg.T_alpha[last][first])
            if self.sys_cfg.T_beta[last][first] == 0:
                # F0
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.task_interval[(mb, 0, 0, last)], # chunk0
                        self.task_interval[(mb, 0, 1, first)], # chunk1
                        delay=forward_delay if forward_delay > 0 else None,
                    )
                )
            else:
                # F0
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.task_interval[(mb, 0, 0, last)],
                        self.communication_intervals[(mb, 0, last, first)],
                    )
                )
                self.task_to_send_comm[self.task_interval[(mb, 0, 0, last)]] = self.communication_intervals[(mb, 0, last, first)]
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.communication_intervals[(mb, 0, last, first)],
                        self.task_interval[(mb, 0, 1, first)],
                        delay=forward_delay if forward_delay > 0 else None,
                    )
                )
            backward_delay = int(sys_cfg.T_alpha[first][last])
            if self.sys_cfg.T_beta[first][last] == 0:
                # B1
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.task_interval[(mb, 1, 1, first)], # chunk1
                        self.task_interval[(mb, 1, 0, last)], # chunk0
                        delay=backward_delay if backward_delay > 0 else None,
                    )
                )
            else:
                # B1
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.task_interval[(mb, 1, 1, first)],
                        self.communication_intervals[(mb, 1, first, last)],
                    )
                )
                self.task_to_send_comm[self.task_interval[(mb, 1, 1, first)]] = self.communication_intervals[(mb, 1, first, last)]
                self.model.add(
                    cp_model_cplex.end_before_start(
                        self.communication_intervals[(mb, 1, first, last)],
                        self.task_interval[(mb, 1, 0, last)],
                        delay=backward_delay if backward_delay > 0 else None,
                    )
                )

        if self.zero_1_dp_modeling:
            self.model.add(cp_model_cplex.start_of(self.dp_comm_interval[(0, 0, 0)]) == 0)
        else:
            # First task starts at time 0
            self.model.add(cp_model_cplex.start_of(self.task_interval[(0, 0, 0, 0)]) == 0)

        # Objective variables
        self.dev_span = cp_model_cplex.integer_var(0, self.horizon, name="dev_span")
        self.bubble = cp_model_cplex.integer_var(0, self.horizon, name="bubble")

        if self.zero_1_dp_modeling:
            self.model.add(cp_model_cplex.max(
                [
                    cp_model_cplex.end_of(self.dp_comm_interval[(dev, chunk, 1)])
                    - cp_model_cplex.start_of(self.dp_comm_interval[(dev, 0, 0)])
                    for dev in range(self.num_dev)
                    for chunk in range(2)
                ]
            ) == self.dev_span)
        else:
            # Maximum completion time across all devices
            self.model.add(
                cp_model_cplex.max(
                    [
                        cp_model_cplex.end_of(
                            self.task_interval[(self.num_mb - 1, 2, 0, dev)]
                        )
                        - cp_model_cplex.start_of(self.task_interval[(0, 0, 0, dev)])
                        for dev in range(self.num_dev)
                    ]
                )
                == self.dev_span
            )

        # Bubble time calculation
        bubble_terms = [
            cp_model_cplex.end_of(self.task_interval[(self.num_mb - 1, 2, 0, dev)])
            - cp_model_cplex.start_of(self.task_interval[(0, 0, 0, dev)])
            - self.num_mb
            * int(sum([sys_cfg.T_F[chunk][dev] + sys_cfg.T_B[chunk][dev] + sys_cfg.T_W[chunk][dev] for chunk in range(2)]))
            for dev in range(self.num_dev)
        ]
        self.model.add(cp_model_cplex.max(bubble_terms) == self.bubble)

        # Set the objective
        self.model.minimize(self.dev_span)
        # self.model.add(cp_model_cplex.minimize_static_lex([self.dev_span, self.bubble]))

        if warm_start:
            raise NotImplementedError(
                "Warm start not implemented for ZBLoopCPLEXScheduler"
            )

    def solve(self, logging=False, time_limit_sec=600, relative_gap=0.0001, workers=64):
        # Configure solver parameters
        self.cp_context.params.TimeLimit = time_limit_sec
        self.cp_context.params.Workers = workers
        self.cp_context.params.LogVerbosity = "Terse" if logging else "Quiet"
        self.cp_context.params.RelativeOptimalityTolerance = relative_gap

        # Solve the model
        solution: cp_model_cplex.CpoSolveResult = self.model.solve()
        if solution:
            self.solved = True
            self.solution = solution
            print(f"Solution status: {solution.get_solve_status()}")
        else:
            print("No solution found")

    def get_schedule(self) -> List[List[PipelineBlockDesc]]:
        if not self.solved:
            raise ValueError("Model not solved yet")

        schedule = [[] for _ in range(self.num_dev)]
        for dev in range(self.num_dev):
            for mb in range(self.num_mb):
                for task_type in range(3):
                    for chunk in range(2):
                        start_time = self.solution.get_value(
                            self.task_interval[(mb, task_type, chunk, dev)]
                        )[0]
                        end_time = self.solution.get_value(
                            self.task_interval[(mb, task_type, chunk, dev)]
                        )[1]
                        post_send_time = -1
                        if self.task_interval[(mb, task_type, chunk, dev)] in self.task_to_send_comm:
                            post_send_time = self.solution.get_value(
                                self.task_to_send_comm[self.task_interval[(mb, task_type, chunk, dev)]]
                            )[0]
                        schedule[dev].append(
                            PipelineBlockDesc(
                                device_id=dev,
                                mb_id=mb,
                                task_type=["F", "B", "W"][task_type],
                                chunk_id=chunk,
                                start_time=start_time,
                                end_time=end_time,
                                post_send_time=post_send_time,
                            )
                        )
            schedule[dev].sort(key=lambda x: x.end_time)
        return schedule
