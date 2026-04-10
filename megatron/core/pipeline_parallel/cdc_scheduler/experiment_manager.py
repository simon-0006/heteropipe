from collections import defaultdict
import json
import os
import time
from typing import Dict, Tuple

import numpy as np


class ExperimentManager:
    def __init__(self, args, tp_rank, dp_rank, pp_rank, num_chunks, exp_dir):
        self.args = args
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.pp_rank = pp_rank
        self.num_chunks = num_chunks

        self.cdc_log_profile = (
            True if self.tp_rank == 0 and self.dp_rank == 0 else False
        )

        self.cdc_profile_iter = args.cdc_profile_iter

        self.profile_result_path = exp_dir
        self.profile_result_rank_file = os.path.join(
            self.profile_result_path, f"{pp_rank}.json"
        )

        # (mb, chunk, type) -> [time, memory before, memory after]
        self.cdc_compute_profile_dict = {}
        # four lists of [alpha, beta] x [to_next, to_prev]
        self.cdc_comm_profiles = None
        # DP comm: 3D list of (chunk, device, 2) -> time
        self.cdc_dp_comm_profiles = None
        # basic memory: parameter, grad, optimizer state
        self.cdc_base_memory = -1
        # parameters in chunk
        self.cdc_chunk_parameters: Dict[int, int] = {}
        # other info to log: chunk -> (vocabembedding, lmhead, numlayers)
        self.cdc_layer_info = {}
        # profile_result_path should exist
        assert os.path.exists(
            self.profile_result_path
        ), f"Profile result path {self.profile_result_path} does not exist"

        self.exp_logging = args.cdc_exp_logging
        if self.tp_rank != 0 or self.dp_rank != 0 or self.pp_rank != 0:
            self.exp_logging_on_this_rank = False
        else:
            self.exp_logging_on_this_rank = True
        self.exp_logging_path = exp_dir
        self.exp_logging_iter_time = defaultdict(list)
        self.exp_logging_max_allocated_mem = defaultdict(list)
        self.exp_logging_perf_model_iter_time = {}
        self.cdc_exp_dump_execution_plan = args.cdc_exp_dump_execution_plan
        self.cdc_latency_bandwidth_delay_as_F_stage = (
            args.cdc_latency_bandwidth_delay_as_F_stage
        )
        self.cdc_exp_test_iters = args.cdc_exp_per_cfg_test_iters
        self.cdc_exp_test_start_iter = max(
            args.cdc_exp_test_start_iter, self.cdc_profile_iter + 1
        )

        self.exp_logging_first_mb = False  # used in benchmarking

        if len(self.cdc_latency_bandwidth_delay_as_F_stage) > 0:
            # if iter = <> then change latency.
            self.cdc_exp_override_iter_map = {
                self.cdc_exp_test_start_iter
                + i
                * self.cdc_exp_test_iters: self.cdc_latency_bandwidth_delay_as_F_stage[
                    i
                ]
                for i in range(len(self.cdc_latency_bandwidth_delay_as_F_stage))
            }
            self.exp_logging_end_iter = (
                self.cdc_exp_test_start_iter + len(self.cdc_latency_bandwidth_delay_as_F_stage) * self.cdc_exp_test_iters
            )
            args.exit_interval = self.exp_logging_end_iter
        else:
            args.exit_interval = self.cdc_exp_test_start_iter
            
        self.profile_result = None
    
    def print_expertiment_info(self) -> str:
        infos = []
        infos.append('CDC Experiment Info:')
        infos.append(f'iter [{self.cdc_profile_iter}]: profile')
        infos.append(f'iter [{self.cdc_exp_test_start_iter}]: start testing')
        for i, (latency, bandwidth) in enumerate(self.cdc_latency_bandwidth_delay_as_F_stage):
            infos.append(f'iter [{self.cdc_exp_test_start_iter + i * self.cdc_exp_test_iters}-{self.cdc_exp_test_start_iter + (i+1) * self.cdc_exp_test_iters - 1}]: latency {latency}, bandwidth {bandwidth}')
        if hasattr(self, 'exp_logging_end_iter'):
            infos.append(f'iter [{self.exp_logging_end_iter}]: end testing')
        return '\n'.join(infos)
    
    def read_profile_result(self):
        while not os.path.exists(os.path.join(self.profile_result_path, "total.json")):
            time.sleep(2)
        
        with open(os.path.join(self.profile_result_path, "total.json"), "r") as f:
            self.profile_result = json.load(f)
        
        self.T_F_stage = np.max(self.profile_result["T_F"]) * self.num_chunks

    def dump_execution_plan_on_this_rank(self):
        return self.cdc_exp_dump_execution_plan and self.exp_logging_on_this_rank

    def F_stage_factor_to_seconds(self, factor) -> float:
        return factor * self.T_F_stage
    
    def seconds_to_F_stage_factor(self, seconds) -> float:
        return seconds / self.T_F_stage

    def get_injected_latency_bandwidth_delay_seconds(self) -> Tuple[float, float]:
        latency, bandwidth = self.get_injected_latency_bandwidth_delay_as_F_stage()
        return self.F_stage_factor_to_seconds(latency), self.F_stage_factor_to_seconds(bandwidth)
        
    def get_injected_latency_bandwidth_delay_as_F_stage(self) -> Tuple[float, float]:
        cur_iter = self.args.curr_iteration
        assert cur_iter >= self.cdc_exp_test_start_iter
        idx = (cur_iter - self.cdc_exp_test_start_iter) // self.cdc_exp_test_iters
        idx = min(idx, len(self.cdc_latency_bandwidth_delay_as_F_stage) - 1)
        return self.cdc_exp_override_iter_map[self.cdc_exp_test_start_iter + self.cdc_exp_test_iters * idx]

    def profile_in_current_iter(self):
        return self.args.curr_iteration == self.cdc_profile_iter
        # return False
    
    def profiling_done(self):
        return self.args.curr_iteration > self.cdc_profile_iter

    def record_schedule_start_in_current_iter(self):
        return (
            self.args.curr_iteration >= self.cdc_exp_test_start_iter
            and self.exp_logging_first_mb
        )

    def record_schedule_end_in_current_iter(self):
        return self.args.curr_iteration >= self.cdc_exp_test_start_iter

    def write_to_json_in_current_iter(self):
        # TODO: write before end/changing config
        if not self.exp_logging_on_this_rank:
            return False
        iters_list = [self.cdc_exp_test_start_iter + (i+1) * self.cdc_exp_test_iters - 1 for i in range(len(self.cdc_latency_bandwidth_delay_as_F_stage))]
        return self.args.curr_iteration in iters_list

    def need_schedule_update_in_current_iter(self):
        update_iter = [self.cdc_exp_test_start_iter + i * self.cdc_exp_test_iters for i in range(len(self.cdc_latency_bandwidth_delay_as_F_stage))]
        return self.args.curr_iteration in update_iter
