import torch
import simpy
import copy
import numpy as np
import pandas as pd
from tensorflow.python.keras.engine.compile_utils import create_pseudo_input_names

from torch_geometric.data import HeteroData
from Environment.data import DataGenerator
from Environment.simulation import *


class State:
    def __init__(self):
        self.data = None
        self.mask = None
        self.current_ops = None

    def update(self, data, mask, current_ops=None):
        self.data = data
        self.mask = mask
        self.current_ops = current_ops if current_ops is not None else self.current_ops


class Factory:
    def __init__(self, data_src, safety_margin=2, device='cpu', algorithm=('RL', 'RL'), record_events=False):
        self.data_src = data_src
        self.safety_margin = safety_margin
        self.device = device
        self.algorithm = algorithm
        self.record_events = record_events

        if type(data_src) is DataGenerator:
            self.df_operations, self.df_locations, self.df_resources = data_src.generate()
        else:
            self.df_operations = pd.read_excel(data_src, sheet_name="operations", engine='openpyxl')
            self.df_locations = pd.read_excel(data_src, sheet_name="locations", engine='openpyxl')
            self.df_resources = pd.read_excel(data_src, sheet_name="resources", engine='openpyxl')

        self.num_jobs = len(self.df_operations["Job_Name"].unique())
        self.num_operations = len(self.df_operations)
        self.num_inputpoints = len([category for category in self.df_locations["Category"] if int(category) == 0])
        self.num_machines = len([category for category in self.df_locations["Category"] if int(category) == 1])
        self.num_buffers = len([category for category in self.df_locations["Category"] if int(category) == 2])
        self.num_outputpoints = len([category for category in self.df_locations["Category"] if int(category) == 3])
        self.num_cranes = len(self.df_resources)

        self.num_rows = len(self.df_locations["Y_Coordinate"].unique())
        self.num_bays = len(self.df_locations["X_Coordinate"].unique())

        self.x_max = int(self.df_locations["X_Coordinate"].max())
        self.y_max = int(self.df_locations["Y_Coordinate"].max())

        self.proctimes = self.df_operations.filter(like='Machine').to_numpy()
        self.proctime_max = np.max(self.proctimes)
        self.proctime_min = np.min(self.proctimes[self.proctimes != 0])

        self.location_id_to_name = {}
        for i, row in self.df_locations.iterrows():
            self.location_id_to_name[int(row["Global_Index"])] = row["Name"]

        self.resource_id_to_name = {}
        for i, row in self.df_resources.iterrows():
            self.resource_id_to_name[int(row["Index"])] = row["Name"]

        self.input_dim_crane = 4
        self.input_dim_operation = 9
        self.input_dim_machine = 8
        self.input_dim_buffer = 4
        self.input_dim_output = 4
        self.input_dim_pair = 6

        self.meta_data_ms = (["operation", "machine", "buffer", "output"],
                             [("operation", "predecessor", "operation"),
                              ("operation", "successor", "operation"),
                              ("machine", "machine_to_operation", "operation"),
                              ("operation", "operation_to_machine", "machine"),
                              ("buffer", "buffer_to_operation", "operation"),
                              ("operation", "operation_to_buffer", "buffer"),
                              ("operation", "operation_to_output", "output"),
                              ("output", "output_to_operation", "operation"),])

        self.state_size_ms = {"operation": self.input_dim_operation,
                              "machine": self.input_dim_machine,
                              "buffer": self.input_dim_buffer,
                              "output": self.input_dim_output}

        self.num_nodes_ms = {"operation": self.num_operations,
                             "machine": self.num_machines,
                             "buffer": self.num_buffers,
                             "output": self.num_outputpoints}

        self.state = None
        self.mask = None

    def step(self, action):
        if self.scheduling_mode == "machine":
            location_id = action // self.num_jobs + self.num_inputpoints
            job_id = action % self.num_jobs

            job = self.monitor.remove_from_queue(job_id, scheduling_mode=self.scheduling_mode)
            current_location = job.current_location
            next_location = self.location_id_to_name[location_id]

            if self.monitor.record_events:
                self.monitor.record(self.sim_env.now, location=current_location, job=job.name,
                                    next_location=next_location, event="Machine_Allocated")

            if self.locations[next_location].category == 2:
                self.monitor.add_to_queue(job, scheduling_mode="machine")

            if self.locations[current_location].call_for_machine_scheduling.get(job.id) is not None:
                self.locations[current_location].call_for_machine_scheduling[job.id].succeed(next_location)
            else:
                self.locations[current_location].call_for_transporting[job_id].succeed(next_location)

            self.scheduling_mode = "crane"
        else:
            crane_id = action

            job = self.monitor.remove_from_queue(scheduling_mode=self.scheduling_mode)
            current_location = job.current_location
            crane = self.resource_id_to_name.get(crane_id)

            self.locations[current_location].call_for_crane_scheduling[job.id].succeed(crane)
            self.scheduling_mode = "machine"

            mask = self._get_ms_mask()
            if mask.any():
                self.monitor.set_scheduling_flag(scheduling_mode="machine")

        done = False

        while True:
            if self.monitor.machine_scheduling or self.monitor.crane_scheduling:
                while self.sim_env.now in [event[0] for event in self.sim_env._queue]:
                    self.sim_env.step()

                if self.scheduling_mode == "machine":
                    mask = self._get_ms_mask()
                else:
                    job = self.monitor.queue_for_crane_scheduling
                    mask = self._get_cs_mask(job, job.next_location)

                if mask.any():
                    self.mask = mask
                    break
                else:
                    self.monitor.machine_scheduling = False

            if self.sink.num_jobs_degenerated == self.num_jobs:
                done = True
                self.monitor.get_logs("./temp.xlsx")
                break

            self.sim_env.step()

        next_state = self._get_state()
        reward = self._calculate_reward()

        self.estimated_completion_time = copy.copy(self.estimated_completion_time_updated)
        if self.decision_time != self.sim_env.now:
            self.decision_time = self.sim_env.now

        return next_state, reward, done

    def reset(self):
        self.sim_env, self.jobs, self.source, self.sink, self.locations, self.resources, self.monitor \
            = self._build_model()

        self.scheduling_mode = "machine"
        self.decision_time = 0.0
        self.estimated_completion_time = np.zeros(self.num_jobs)
        self.estimated_completion_time_updated = np.zeros(self.num_jobs)

        while True:
            if self.monitor.machine_scheduling or self.monitor.crane_scheduling:
                while self.sim_env.now in [event[0] for event in self.sim_env._queue]:
                    self.sim_env.step()
                break
            self.sim_env.step()

        if self.decision_time != self.sim_env.now:
            self.decision_time = self.sim_env.now

        state = self._get_state()

        self.decision_time = self.sim_env.now
        self.estimated_completion_time = copy.copy(self.estimated_completion_time_updated)

        return state

    def _get_ms_mask(self):
        num_rows = self.num_machines + self.num_buffers + self.num_outputpoints
        num_columns = self.num_jobs

        mask_machine = np.zeros((num_rows, num_columns), dtype=bool)
        mask_buffer = np.zeros((num_rows, num_columns), dtype=bool)
        mask_output = np.zeros((num_rows, num_columns), dtype=bool)

        for job in self.monitor.queue_for_machine_scheduling.values():
            if job.in_transportation:
                continue

            operation = job.get_current_operation()
            current_coord = self.locations[job.current_location].coord

            for name, location in self.locations.items():
                local_id = location.local_id
                global_id = location.global_id
                target_coord = location.coord
                category = location.category

                if category == 0:
                    continue
                else:
                    flag_availability = (not location.check_status()) or (job.current_location == name)
                    flag_accessibility = not ((current_coord[0] < self.safety_margin
                                               and target_coord[0] > self.x_max - self.safety_margin) or
                                              (current_coord[0] > self.x_max - self.safety_margin
                                               and target_coord[0] < self.safety_margin))
                    flag_crane_availability = self._get_cs_mask(job, name).any()

                    if category == 1:
                        if operation is not None:
                            flag_eligibility = int(operation.get_processing_time(local_id)) != 0
                            mask_machine[global_id - self.num_inputpoints, job.id] \
                                = (flag_eligibility & flag_availability & flag_accessibility & flag_crane_availability)
                        else:
                            continue
                    elif category == 2:
                        if job.next_location is not None:
                            continue
                        else:
                            if (operation is None) or (not operation.id in self.monitor.operations_waiting.keys()):
                                mask_buffer[global_id - self.num_inputpoints, job.id] \
                                    = flag_availability & flag_accessibility
                            else:
                                continue
                    elif category == 3:
                        if operation is None:
                            mask_output[global_id - self.num_inputpoints, job.id] \
                                = flag_availability & flag_accessibility
                        else:
                            continue
                    else:
                        continue

        if (mask_machine | mask_output).any():
            mask = mask_machine | mask_output
        else:
            mask = mask_buffer

        mask = torch.tensor(mask, dtype=torch.bool).to(self.device)

        return mask


    def _get_cs_mask(self, job, next_location):
        mask = np.zeros(self.num_cranes + 1, dtype=bool)

        if job.current_location == next_location:
            mask[self.num_cranes] = 1
        else:
            current_location_coord = self.locations[job.current_location].coord
            next_location_coord = self.locations[next_location].coord

            for crane in self.resources.values():
                flag_accessibility = False
                if ((crane.id == 0) and (current_location_coord[0] <= self.x_max - self.safety_margin)
                    and (next_location_coord[0] <= self.x_max - self.safety_margin)) or \
                        ((crane.id == 1) and (current_location_coord[0] >= self.safety_margin)
                         and (next_location_coord[0] >= self.safety_margin)):
                    flag_accessibility = True

                # ex) current working order: (3, M1, M3) -> new working order: (2, M3, M1)
                flag_not_reversed = True
                if crane.current_working_order is not None:
                    if crane.current_working_order[2] == job.current_location:
                        flag_not_reversed = False

                # ex) queue: [(3, M1, M3), (4, M4, M1)] -> new working order: (2, M3, M4)
                flag_not_cycled = True
                if job.current_location != next_location:
                    start = job.current_location
                    queue_from = []
                    queue_to = []

                    for temp in crane.queue:
                        if self.locations[temp[1]].category == 1 and self.locations[temp[2]].category == 1:
                            queue_from.append(temp[1])
                            queue_to.append(temp[2])

                    new_start = next_location
                    while new_start in queue_from:
                        index = queue_from.index(new_start)
                        new_start = queue_to[index]

                        if new_start == start:
                            flag_not_cycled = False
                            break

                # ex) [Crane 0] current working order: (6, M2, M0) / queue: [(4, M1, M2), (3, M3, B0)]
                #     [Crane 1] current working order: (2, M0, B1) / queue: [(9, I0, M4)] -> new working order: (5, B0, M3)
                flag_not_blocked = True
                if job.current_location != next_location:
                    if crane.current_working_order is not None:
                        queue = ([crane.current_working_order] + crane.queue[:]
                                 + [(job.id, job.current_location, next_location)])
                        exclude_first = True if crane.status == "unloading" else False
                    else:
                        queue = crane.queue[:] + [(job.id, job.current_location, next_location)]
                        exclude_first = False

                    if crane.opposite.current_working_order is not None:
                        queue_opposite = [crane.opposite.current_working_order] + crane.opposite.queue
                        exclude_first_opposite = True if crane.opposite.status == "unloading" else False
                    else:
                        queue_opposite = crane.opposite.queue[:]
                        exclude_first_opposite = False

                    if exclude_first:
                        queue_from = set([temp[1] for temp in queue[1:]
                                          if self.locations[temp[1]].category == 1])
                    else:
                        queue_from = set([temp[1] for temp in queue
                                          if self.locations[temp[1]].category == 1])

                    if exclude_first_opposite:
                        queue_from_opposite = set([temp[1] for temp in queue_opposite[1:]
                                                   if self.locations[temp[1]].category == 1])
                    else:
                        queue_from_opposite = set([temp[1] for temp in queue_opposite
                                                   if self.locations[temp[1]].category == 1])

                    queue_to = set([temp[2] for temp in queue if self.locations[temp[2]].category == 1])
                    queue_to_opposite = set([temp[2] for temp in queue_opposite
                                             if self.locations[temp[2]].category == 1])

                    intersection1 = len(set.intersection(queue_to, queue_from_opposite)) > 0
                    intersection2 = len(set.intersection(queue_from, queue_to_opposite)) > 0

                    if intersection1 and intersection2:
                        flag_not_blocked = False
                        break

                mask[crane.id] = flag_accessibility & flag_not_reversed & flag_not_cycled & flag_not_blocked

        mask = torch.tensor(mask, dtype=torch.bool).to(self.device)

        return mask

    def _get_state(self):
        if self.scheduling_mode == "machine":
            machine_scheduling_algorithm = self.algorithm[0]

            if machine_scheduling_algorithm == "RL":
                fea_operation = np.zeros((self.num_operations, self.input_dim_operation))
                fea_machine = np.zeros((self.num_machines, self.input_dim_machine))
                fea_buffer = np.zeros((self.num_buffers, self.input_dim_buffer))
                fea_output = np.zeros((self.num_outputpoints, self.input_dim_output))
                fea_pair = np.zeros((self.num_jobs, self.num_machines + self.num_buffers + self.num_outputpoints, self.input_dim_pair))
                current_operations = np.zeros(self.num_jobs)

                edge_predecessor, edge_successor = [[], []], [[], []]
                edge_machine_to_operation, edge_operation_to_machine = [[], []], [[], []]
                edge_buffer_to_operation, edge_operation_to_buffer = [[], []], [[], []]
                edge_output_to_operation, edge_operation_to_output = [[], []], [[], []]

                proctime_remaining = np.zeros((self.num_operations, self.num_machines))
                proctime_current = np.zeros((self.num_jobs, self.num_machines))

                proctime_remaining_mask = np.zeros(self.num_operations, dtype=bool)
                proctime_current_mask = np.zeros(self.num_jobs, dtype=bool)

                self.estimated_completion_time_updated = copy.copy(self.estimated_completion_time)

                # Operation Feature
                for j in self.df_operations["Job_Index"].unique():
                    if j in self.monitor.jobs_before_system.keys():
                        job = self.monitor.jobs_before_system[j]
                    elif j in self.monitor.jobs_in_system.keys():
                        job = self.monitor.jobs_in_system[j]
                    else:
                        job = self.monitor.jobs_after_system[j]

                    if job.step < len(job.operations):
                        current_operations[job.id] = job.operations[job.step].id
                    else:
                        current_operations[job.id] = job.operations[-1].id

                    # 작업되지 않은 operation들의 작업시간 정보
                    for operation in job.operations[job.step:]:
                        proctime_remaining[operation.id, :] \
                            = (operation.options - self.proctime_min) / (self.proctime_max - self.proctime_min)
                        proctime_remaining_mask[operation.id] = True

                    # 의사결정이 필요한 job에 대하여, 해당 job의 다음 operation 작업시간 정보
                    if j in self.monitor.queue_for_machine_scheduling.keys():
                        if job.step < len(job.operations):
                            proctime_current[job.id, :] \
                                = (job.operations[job.step].options - self.proctime_min) / (self.proctime_max - self.proctime_min)
                            proctime_current_mask[job.id] = True

                    # job에 수행되는 각 operation의 평균 작업시간
                    job_proctime = [(np.mean(operation.options[operation.options != 0] - self.proctime_min)
                                     / (self.proctime_max - self.proctime_min))
                                    for operation in job.operations]
                    # job에 수행되는 모든 operation의 평균 작업시간 총합
                    job_proctime_sum = np.sum(job_proctime)

                    if job.step < len(job.operations):
                        job_remaining_proctime_sum = np.sum([job_proctime[job.step:]])
                    else:
                        job_remaining_proctime_sum = 0

                    for k, operation in enumerate(job.operations):
                        eligible_options = ((operation.options[operation.options != 0] - self.proctime_min)
                                            / (self.proctime_max - self.proctime_min))

                        if ((k < job.step)
                            or (k == job.step and operation.id in self.monitor.operations_working.keys())):
                            machine = self.locations[operation.allocated_machine]
                            proctime = operation.get_processing_time(machine.local_id)
                            earliest_finish_time = operation.start_time
                        else:
                            proctime = np.min(eligible_options)
                            earliest_finish_time = max(self.sim_env.now, job.arrival_time)

                        earliest_finish_time = earliest_finish_time + proctime

                        # Operation Feature
                        if (operation.id in self.monitor.operations_working.keys()
                                or operation.id in self.monitor.operations_waiting.keys()):
                            f0 = [0, 1, 0]
                        elif operation.id in self.monitor.operations_done:
                            f0 = [0, 0, 1]
                        else:
                            f0 = [1, 0, 0]

                        f1 = np.min(eligible_options)
                        f2 = np.mean(eligible_options)
                        f3 = np.max(eligible_options)
                        f4 = np.sum(job_proctime[k:]) # / (len(job.operations) - k)
                        f5 = len(eligible_options) / self.num_machines
                        f6 = earliest_finish_time # / (k + 1)

                        fea_operation[operation.id, :3] = f0
                        fea_operation[operation.id, 3:] = [f1, f2, f3, f4, f5, f6]

                    self.estimated_completion_time_updated[job.id] = earliest_finish_time

                # Location Feature
                proctime_remaining = proctime_remaining[proctime_remaining_mask]
                proctime_current = proctime_current[proctime_current_mask]

                proctime_remaining_mean = np.array([np.mean(temp[temp >= 0]) for temp in proctime_remaining])
                proctime_current_mean = np.array([np.mean(temp[temp >= 0]) for temp in proctime_current])

                proctime_remaining_sum = np.sum(proctime_remaining_mean)  # 작업이 미완료된 operation의 평균 작업시간 합
                proctime_current_sum = np.sum(proctime_current_mean)  # 스케줄링 대상 operation의 평균 작업시간 합

                proctime_compatible = np.copy(proctime_current)

                available_time_list = []
                for location in self.locations.values():
                    if location.category == 0:
                        continue
                    else:
                        fully_occupied = location.check_status()
                        if not fully_occupied:
                            f0 = [1, 0]
                        else:
                            f0 = [0, 1]

                        xcoord, ycoord = location.coord
                        f1 = [xcoord / self.x_max if self.x_max !=0 else 0,
                              ycoord / self.y_max if self.y_max != 0 else 0]

                        if location.category == 1:
                            if fully_occupied:
                                proctime_compatible[:, location.local_id] = -1

                            eligible_proctime_remaining = proctime_remaining[:, location.local_id][
                                proctime_remaining[:, location.local_id] >= 0]
                            eligible_proctime_current = proctime_current[:, location.local_id][
                                proctime_current[:, location.local_id] >= 0]

                            available_time = location.get_available_time()
                            available_time_list.append(available_time)

                            f2 = np.sum(eligible_proctime_current) / proctime_current_sum
                            f3 = len(eligible_proctime_current) / len(proctime_current)
                            f4 = available_time - self.sim_env.now
                            f5 = (self.sim_env.now - location.completion_time) if not fully_occupied else 0

                            fea_machine[location.local_id, :2] = f0
                            fea_machine[location.local_id, 2:4] = f1
                            fea_machine[location.local_id, 4:] = [f2, f3, f4, f5]

                        elif location.category == 2:
                            fea_buffer[location.local_id, :2] = f0
                            fea_buffer[location.local_id, 2:4] = f1

                        else:
                            fea_output[location.local_id, :2] = f0
                            fea_output[location.local_id, 2:4] = f1

                if int(np.max(available_time_list) - self.sim_env.now) != 0:
                    fea_machine[:, 6] = fea_machine[:, 6] / (np.max(available_time_list) - self.sim_env.now)
                fea_machine[:, 7] = fea_machine[:, 7] / np.max(fea_machine[:, 7]) \
                    if np.max(fea_machine[:, 7]) > 0.0 else 0.0

                # Pair Feature
                tag = np.array([(temp >= 0).any() for temp in proctime_compatible])
                proctime_compatible = proctime_compatible[tag]

                for j, job in enumerate(self.monitor.queue_for_machine_scheduling.values()):

                    if job.step < len(job.operations):
                        current_operation = job.operations[job.step]
                    else:
                        current_operation = job.operations[-1]

                    for i, location in enumerate(self.locations.values()):
                        if location.category == 0:
                            continue
                        else:
                            job_coord = self.locations[job.current_location].coord
                            f1 = (location.coord[0] - job_coord[0]) / self.x_max if self.x_max != 0 else 0
                            f2 = (location.coord[1] - job_coord[1]) / self.y_max if self.y_max != 0 else 0

                            if location.category == 1:
                                fully_occupied = location.check_status()

                                if not fully_occupied:
                                    options = current_operation.options
                                    options = (options - self.proctime_min) / (self.proctime_max - self.proctime_min)
                                    proctime = current_operation.get_processing_time(location.local_id)
                                    proctime = (proctime - self.proctime_min) / (self.proctime_max - self.proctime_min)

                                    if proctime >= 0:
                                        proctime_compatible_copy = copy.copy(proctime_compatible)
                                        proctime_compatible_copy[:, location.local_id] = -1
                                        # num_compatible_pairs = len(proctime_compatible[proctime_compatible >= 0])
                                        # num_compatible_pairs_updated \
                                        #     = len(proctime_compatible_copy[proctime_compatible_copy >= 0])
                                        # min_proctime_compatible \
                                        #     = np.array([np.min(temp[temp >= 0]) for temp in proctime_compatible])
                                        # min_proctime_compatible_updated \
                                        #     = np.array([np.min(temp[temp >= 0]) for temp in proctime_compatible_copy])

                                        f3 = proctime
                                        f4 = proctime / np.max(options)
                                        f5 = proctime / np.max(proctime_compatible[:, location.local_id]) \
                                            if np.max(proctime_compatible[:, location.local_id]) > 0 else 0
                                        f6 = proctime / np.max(proctime_compatible)

                                        fea_pair[job.id, location.global_id - self.num_inputpoints, :] = [f1, f2, f3, f4, f5, f6]
                                    else:
                                        fea_pair[job.id, location.global_id - self.num_inputpoints, :] = [f1, f2, 0, 0, 0, 0]
                            else:
                                fea_pair[job.id, location.global_id - self.num_inputpoints, :] = [f1, f2, 0, 0, 0, 0]

                # Edge Construction
                for j in self.df_operations["Job_Index"].unique():
                    if j in self.monitor.jobs_before_system.keys():
                        job = self.monitor.jobs_before_system[j]
                    elif j in self.monitor.jobs_in_system.keys():
                        job = self.monitor.jobs_in_system[j]
                    else:
                        job = self.monitor.jobs_after_system[j]

                    for k, operation in enumerate(job.operations):
                        if k > 0:
                            edge_predecessor[0].append(operation.id - 1)
                            edge_predecessor[1].append(operation.id)
                            edge_successor[0].append(operation.id)
                            edge_successor[1].append(operation.id - 1)

                        for location in self.locations.values():
                            if location.category == 0:
                                continue

                            elif location.category == 1:
                                if k >= job.step:
                                    proctime = operation.get_processing_time(location.local_id)
                                    if proctime != 0:
                                        edge_operation_to_machine[0].append(operation.id)
                                        edge_operation_to_machine[1].append(location.local_id)
                                        edge_machine_to_operation[0].append(location.local_id)
                                        edge_machine_to_operation[1].append(operation.id)
                                else:
                                    if location.name == operation.allocated_machine:
                                        edge_operation_to_machine[0].append(operation.id)
                                        edge_operation_to_machine[1].append(location.local_id)
                                        edge_machine_to_operation[0].append(location.local_id)
                                        edge_machine_to_operation[1].append(operation.id)

                            elif location.category == 2:
                                if k >= job.step:
                                    edge_operation_to_buffer[0].append(operation.id)
                                    edge_operation_to_buffer[1].append(location.local_id)
                                    edge_buffer_to_operation[0].append(location.local_id)
                                    edge_buffer_to_operation[1].append(operation.id)

                            else:
                                if k == len(job.operations) - 1:
                                    edge_operation_to_output[0].append(operation.id)
                                    edge_operation_to_output[1].append(location.local_id)
                                    edge_output_to_operation[0].append(location.local_id)
                                    edge_output_to_operation[1].append(operation.id)

                fea_operation = torch.from_numpy(fea_operation).type(torch.float32).to(self.device)
                fea_machine = torch.from_numpy(fea_machine).type(torch.float32).to(self.device)
                fea_buffer = torch.from_numpy(fea_buffer).type(torch.float32).to(self.device)
                fea_output = torch.from_numpy(fea_output).type(torch.float32).to(self.device)
                edge_predecessor = torch.from_numpy(np.array(edge_predecessor)).type(torch.long).to(self.device)
                edge_successor = torch.from_numpy(np.array(edge_successor)).type(torch.long).to(self.device)
                edge_operation_to_machine = torch.from_numpy(np.array(edge_operation_to_machine)).type(torch.long).to(self.device)
                edge_machine_to_operation = torch.from_numpy(np.array(edge_machine_to_operation)).type(torch.long).to(self.device)
                edge_operation_to_buffer = torch.from_numpy(np.array(edge_operation_to_buffer)).type(torch.long).to(self.device)
                edge_buffer_to_operation = torch.from_numpy(np.array(edge_buffer_to_operation)).type(torch.long).to(self.device)
                edge_operation_to_output = torch.from_numpy(np.array(edge_operation_to_output)).type(torch.long).to(self.device)
                edge_output_to_operation = torch.from_numpy(np.array(edge_output_to_operation)).type(torch.long).to(self.device)

                data = HeteroData()
                data["operation"].x = fea_operation
                data["machine"].x = fea_machine
                data["buffer"].x = fea_buffer
                data["output"].x = fea_output
                data["operation", "predecessor", "operation"].edge_index = edge_predecessor
                data["operation", "successor", "operation"].edge_index = edge_successor
                data["operation", "operation_to_machine", "machine"].edge_index = edge_operation_to_machine
                data["machine", "machine_to_operation", "operation"].edge_index = edge_machine_to_operation
                data["operation", "operation_to_buffer", "buffer"].edge_index = edge_operation_to_buffer
                data["buffer", "buffer_to_operation", "operation"].edge_index = edge_buffer_to_operation
                data["operation", "operation_to_output", "output"].edge_index = edge_operation_to_output
                data["output", "output_to_operation", "operation"].edge_index = edge_output_to_operation

            else:
                num_rows = self.num_machines + self.num_buffers + self.num_outputpoints
                num_columns = self.num_jobs

                data = np.zeros((num_rows, num_columns))

                if machine_scheduling_algorithm == "SPT":
                    for job in self.monitor.queue_for_machine_scheduling.values():
                        operation = job.get_current_operation()
                        for location in self.locations.values():
                            if location.category == 1:
                                if operation is not None:
                                    proctime = operation.get_processing_time(location.local_id)
                                    data[location.global_id - self.num_inputpoints, job.id] \
                                        = 1 / proctime if proctime > 0 else 1
                            elif location.category == 2:
                                data[location.global_id - self.num_inputpoints, job.id] = 1
                            elif location.category == 3:
                                if operation is None:
                                    data[location.global_id - self.num_inputpoints, job.id] = 1

                elif machine_scheduling_algorithm == "MOR":
                    for job in self.monitor.queue_for_machine_scheduling.values():
                        remaining_operations = len(job.operations) - job.step
                        for location in self.locations.values():
                            if location.category == 1:
                                if remaining_operations > 0:
                                    data[location.global_id - self.num_inputpoints, job.id] = remaining_operations
                            elif location.category == 2:
                                data[location.global_id - self.num_inputpoints, job.id] = 1
                            elif location.category == 3:
                                if remaining_operations == 0:
                                    data[location.global_id - self.num_inputpoints, job.id] = 1

                elif machine_scheduling_algorithm == "MWKR":
                    for job in self.monitor.queue_for_machine_scheduling.values():
                        if job.step < len(job.operations):
                            remaining_work = np.sum([np.mean(operation.options[operation.options != 0])
                                                     for operation in job.operations[job.step:]])
                        else:
                            remaining_work = 0.0
                        for location in self.locations.values():
                            if location.category == 1:
                                if remaining_work > 0.0:
                                    data[location.global_id - self.num_inputpoints, job.id] = remaining_work
                            elif location.category == 2:
                                data[location.global_id - self.num_inputpoints, job.id] = 1
                            elif location.category == 3:
                                if remaining_work == 0.0:
                                    data[location.global_id - self.num_inputpoints, job.id] = 1

                elif machine_scheduling_algorithm == "RAND":
                    data[:, :] = 1.0
        else:
            crane_scheduling_algorithm = self.algorithm[1]

            if crane_scheduling_algorithm == "RL":
                pass
            else:
                data = np.zeros(self.num_cranes + 1)
                data[self.num_cranes] = 1

                job = self.monitor.queue_for_crane_scheduling
                location_coord = self.locations[job.current_location].coord

                if crane_scheduling_algorithm == "SETT":
                    for crane in self.resources.values():
                        if len(crane.queue) > 0:
                            crane_coord = self.locations[crane.queue[-1][2]].coord
                        else:
                            if crane.current_working_order is not None:
                                crane_coord = self.locations[crane.current_working_order[2]].coord
                            else:
                                crane_coord = crane.current_coord

                        x_travel_time = abs(location_coord[0] - crane_coord[0]) / crane.x_velocity
                        y_travel_time = abs(location_coord[1] - crane_coord[1]) / crane.y_velocity
                        empty_travel_time = max(x_travel_time, y_travel_time)

                        data[crane.id] = 1 / empty_travel_time if empty_travel_time > 0 else 1

                elif crane_scheduling_algorithm == "NCR":
                    pass

                elif crane_scheduling_algorithm == "LOR":
                    for crane in self.resources.values():
                        remaining_jobs = 0
                        if crane.current_working_order is not None:
                            remaining_jobs += 1
                        remaining_jobs += len(crane.queue)

                        data[crane.id] = 1 / remaining_jobs if remaining_jobs > 0 else 1

                elif crane_scheduling_algorithm == "LWKR":
                    for crane in self.resources.values():
                        sequence = []
                        if crane.current_working_order is not None:
                            if crane.to_location == crane.current_working_order[1]:
                                sequence.append(crane.current_working_order[1])
                                sequence.append(crane.current_working_order[2])
                            else:
                                sequence.append(crane.current_working_order[2])
                        for working_order in crane.queue:
                            sequence.append(working_order[1])
                            sequence.append(working_order[2])

                        remaining_work = 0
                        current_coord = crane.current_coord
                        for location_name in sequence:
                            location_coord = self.locations[location_name].coord

                            x_travel_time = abs(location_coord[0] - current_coord[0]) / crane.x_velocity
                            y_travel_time = abs(location_coord[1] - current_coord[1]) / crane.y_velocity
                            travel_time = max(x_travel_time, y_travel_time)
                            remaining_work += travel_time

                            current_coord = location_coord

                        data[crane.id] = 1 / remaining_work if remaining_work > 0 else 1

                elif crane_scheduling_algorithm == "RAND":
                    data[:] = 1.0

        if self.scheduling_mode == "machine":
            mask = self._get_ms_mask()
        else:
            job = self.monitor.queue_for_crane_scheduling
            mask = self._get_cs_mask(job, job.next_location)

        state = State()
        if self.scheduling_mode == "machine" and self.algorithm[0] == "RL":
            state.update(data, mask, current_operation)
        else:
            state.update(data, mask)

        self.state = state

        return state

    def _calculate_reward(self):
        return 0.0

    def _build_model(self):
        sim_env = simpy.Environment()
        monitor = Monitor(self.record_events)

        jobs = []
        df_operations_group = self.df_operations.groupby(by=["Job_Name", "Job_Index", "Arrival_Date"])

        for idx, df_temp in df_operations_group:
            job_name, job_index, arrival_date = idx

            operations = []
            for _, row in df_temp.iterrows():
                operation_name = row["Operation_Name"]
                operation_index = row["Operation_Index"]
                options = np.array(row.iloc[-self.num_machines:].to_list())
                operation = Operation(operation_name, operation_index, options=options)
                operations.append(operation)

            job = Job(job_name, job_index, arrival_date, operations)
            jobs.append(job)

        jobs = sorted(jobs, key=lambda x: x.arrival_time)
        input_points = self.df_locations["Name"][self.df_locations["Category"] == 0].to_list()

        locations = {}
        resources = {}

        source = Source(sim_env, "Source", jobs, locations, input_points, monitor)
        sink = Sink(sim_env, "Sink", monitor)

        for _, row in self.df_locations.iterrows():
            name = row["Name"]
            global_index = int(row["Global_Index"])
            local_index = int(row["Local_Index"])
            category = int(row["Category"])
            coord = (int(row["X_Coordinate"]), int(row["Y_Coordinate"]))

            if category == 0:
                location = InputPoint(sim_env, name, global_index, local_index, category, coord, locations, resources, monitor)
            elif category == 1:
                location = Machine(sim_env, name, global_index, local_index, category, coord, locations, resources, monitor)
            elif category == 2:
                location = Buffer(sim_env, name, global_index, local_index, category, coord, locations, resources, monitor)
            elif category == 3:
                location = OutputPoint(sim_env, name, global_index, local_index, category, coord, sink, monitor)

            locations[name] = location

        for _, row in self.df_resources.iterrows():
            name = row["Name"]
            index = int(row["Index"])
            x_velocity = float(row["X_Velocity"])
            y_velocity = float(row["Y_Velocity"])
            initial_coord = (int(row["Initial_X_Coordinate"]), int(row["Initial_Y_Coordinate"]))

            crane = Crane(sim_env, name, index, self.safety_margin, x_velocity, y_velocity, initial_coord, locations, monitor)
            resources[name] = crane

        for crane in resources.values():
            if crane.id == 0:
                opposite_crane = resources["C-1"]
            else:
                opposite_crane = resources["C-0"]
            crane.set_opposite_crane(opposite_crane)

        return sim_env, jobs, source, sink, locations, resources, monitor


if __name__ == "__main__":
    import random
    from Agent.FlexibleJobShop.heuristic import MachineSchedulingHeuristic
    from Agent.CraneTransportation.heuristic import CraneSchedulingHeuristic

    agent_ms = MachineSchedulingHeuristic()
    agent_cs = CraneSchedulingHeuristic()

    # data_src = DataGenerator()
    data_src = "../input/validation/10-5/instance-1.xlsx"
    env = Factory(data_src, algorithm=("RL","RAND"), record_events=True)

    step = 0
    random.seed(42)
    state = env.reset()

    while True:
        if env.scheduling_mode == "machine":
            # action = agent_ms.act(state)
            mask = state.mask.flatten()
            candidates = np.where(mask == True)[0]
            action = np.random.choice(candidates)
        else:
            action = agent_cs.act(state)

        next_state, reward, done = env.step(action)

        state = next_state
        step += 1

        print(step)

        if done:
            break