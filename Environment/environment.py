import torch
import simpy
import copy
import numpy as np
import pandas as pd

from collections import OrderedDict
from torch_geometric.data import HeteroData
from Environment.data import DataGenerator
from Environment.simulation import *


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

        self.num_rows = self.df_locations["Y_Coordinate"].max()
        self.num_bays = self.df_locations["X_Coordinate"].max()

        self.input_dim_crane = 4
        self.input_dim_operation = 9
        self.input_dim_machine = 6
        self.input_dim_buffer = None
        self.input_dim_pair = 6

        self.meta_data = (["operation", "inputpoint", "machine", "buffer", "outputpoint", "crane"],
                          [("operation", "predecessor", "operation"),
                           ("machine", "machine_to_operation", "operation"),
                           ("operation", "operation_to_machine", "machine"),
                           ("buffer", "buffer_to_operation", "operation"),
                           ("operation", "operation_to_buffer", "buffer"),
                           ("crane", "crane_to_machine", "machine"),
                           ("machine", "machine_to_crane", "crane"),
                           ("crane", "crane_to_buffer", "buffer"),
                           ("buffer", "buffer_to_crane", "crane")])

        self.state_size = {"operation": self.input_dim_operation,
                           "machine": self.input_dim_machine,
                           "crane": self.input_dim_crane}

        self.num_nodes = {"operation": self.num_operations,
                          "machine": self.num_machines,
                          "crane": self.num_cranes}


    def step(self, action):
        if self.scheduling_mode == "machine":
            machine_id = action % self.num_machines
            job_id = action // self.num_machines
            done = False
        else:
            crane_id = action

        machine_flag = np.zeros(self.num_machines, dtype=bool)
        for id, name in self.machine_ids.items():
            machine = self.model[name]
            idle, available_time = machine.check_status()
            if id == machine_id:
                machine_flag[id] = False
            else:
                machine_flag[id] = idle

        job = self.monitor.remove_queue(job_id, machine_flag)
        operation = job.get_current_operation()
        current_machine = job.current_machine
        next_machine = self.machine_ids[machine_id] if self.machine_ids.get(machine_id) is not None else "Buffer"

        self.completion_time_updated[job.id] = self.sim_env.now + operation.get_processing_time(machine_id)

        if current_machine is None:
            self.model["Source"].calling_event[job.name].succeed(next_machine)
        else:
            self.model[current_machine].calling_event[job.name].succeed(next_machine)

        self.actions_done.append(machine_id)

        while True:
            while True:
                if self.monitor.scheduling:
                    while self.sim_env.now in [event[0] for event in self.sim_env._queue]:
                        self.sim_env.step()
                    break
                if len(self.monitor.operations_done) == len(self.df_scenario):
                    done = True
                    break

                if len(self.sim_env._queue) == 0:
                    log = self.monitor.get_logs()
                    print(0)

                self.sim_env.step()

            if self.decision_time != self.sim_env.now:
                self.actions_done = []

            if self.algorithm == "RL":
                next_state, current_ops, added_info = self._get_state_for_RL()
                mask = self._get_mask()
            else:
                next_state = self._get_state_for_heuristics()

            if done:
                break

            jobs_to_buffer = []
            for job in self.monitor.jobs_in_queue.values():
                if not mask[:, job.id].any():
                    operation = job.get_current_operation()
                    if not operation.id in self.monitor.operations_in_buffer.keys():
                        jobs_to_buffer.append(job.id)

            for job_id in jobs_to_buffer:
                machine_flag = np.zeros(self.num_machines, dtype=bool)
                job = self.monitor.remove_queue(job_id, machine_flag)
                current_machine = job.current_machine
                next_machine = "Buffer"

                if current_machine is None:
                    self.model["Source"].calling_event[job.name].succeed(next_machine)
                else:
                    self.model[current_machine].calling_event[job.name].succeed(next_machine)

            if not mask.any():
                if self.monitor.scheduling:
                    self.monitor.scheduling = False
            else:
                break

        reward = self._calculate_reward()

        self.estimated_completion_time = copy.copy(self.estimated_completion_time_updated)
        self.completion_time = copy.copy(self.completion_time_updated)
        self.total_tardiness = self.model["Sink"].total_tardiness
        if self.decision_time != self.sim_env.now:
            self.decision_time = self.sim_env.now

        return next_state, reward, done, mask, current_ops, added_info

    def reset(self):
        self.sim_env, self.jobs, self.locations, self.resources, self.monitor = self._modeling()

        self.scheduling_mode = "machine"
        self.decision_time = 0.0
        self.estimated_completion_time = np.zeros(self.num_jobs)
        self.estimated_completion_time_updated = np.zeros(self.num_jobs)

        while True:
            while True:
                if self.monitor.machine_scheduling or self.monitor.crane_scheduling:
                    while self.sim_env.now in [event[0] for event in self.sim_env._queue]:
                        self.sim_env.step()
                    break
                self.sim_env.step()

            if self.decision_time != self.sim_env.now:
                self.decision_time = self.sim_env.now

            state, current_ops, added_info = self._get_state()
            mask = self._get_mask()

            self.decision_time = self.sim_env.now

        self.estimated_completion_time = copy.copy(self.estimated_completion_time_updated)

        return state, mask, current_ops, added_info

    def _set_static_parameters(self, df_operations, df_locations, df_resources):
        mapping_for_machine_scheduling = OrderedDict()
        mapping_for_crane_scheduling = OrderedDict()

        index_for_machine_scheduling = 0
        index_for_crane_scheduling = 0
        for i, row in df_locations.iterrows():
            if int(row["Location_Type"]) in [1, 2, 3]:
                mapping_for_machine_scheduling[row["Location_Index"]] = index
                index_for_machine_scheduling += 1

        job_ids = OrderedDict()
        for i, name in zip(df_scenario["Job_Index"].unique(), df_scenario["Job_Name"].unique()):
            job_ids[int(i)] = name

        machine_ids = OrderedDict()
        for i, name in enumerate(df_scenario.columns[9:]):
            machine_ids[int(i)] = name


    def _get_mask(self):
        if self.scheduling_mode == "machine":
            num_rows = self.num_machines + self.num_buffers + self.num_outputpoints
            num_columns = self.num_jobs

            mask_machine = np.zeros((num_rows, num_columns), dtype=bool)
            mask_buffer = np.zeros((num_rows, num_columns), dtype=bool)
            mask_output = np.zeros((num_rows, num_columns), dtype=bool)

            for job_id, job in self.monitor.queue_for_machine_scheduling.values():
                operation = job.get_current_operation()
                current_coord = self.locations[job.current_location]

                for name, location in self.locations.items():
                    local_id = location.local_id
                    global_id = location.global_id
                    target_coord = location.coord
                    category = location.category

                    flag_availability = location.check_status()
                    flag_accesibility = (current_coord[0] < self.safety_margin
                                         and target_coord[0] < self.num_bays - self.safety_margin) or \
                                        (current_coord[0] > self.num_bays - self.safety_margin - 1
                                         and target_coord[0] > self.safety_margin - 1)

                    if category == 1:
                        if operation is not None:
                            flag_eligibility = int(operation.get_processing_time(local_id)) != 0
                            mask_machine[global_id - self.num_inputpoints, job.id] \
                                = flag_eligibility & flag_availability & flag_accesibility
                        else:
                            continue
                    elif category == 2:
                        mask_buffer[global_id - self.num_inputpoints, job.id] = flag_availability & flag_accesibility
                    elif category == 3:
                        mask_output[global_id - self.num_inputpoints, job.id] = flag_availability & flag_accesibility
                    else:
                        continue

            if (mask_machine | mask_output).any():
                mask = mask_machine | mask_output
            else:
                mask = mask_buffer

        else:
            mask = np.zeros(self.num_cranes, dtype=bool)

            for job in self.monitor.queue_for_crane_scheduling.values():
                location_name = job.current_location
                location_coord = self.locations[location_name].coord

                for crane_id, crane_name in self.crane_ids.items():
                    if ((crane_id == 0) and (location_coord[0] < self.num_bays - self.safety_margin)) or \
                            ((crane_id == 1) and (location_coord[0] > self.safety_margin - 1)):
                        mask[crane_id] = 1

        mask = torch.tensor(mask, dtype=torch.bool).to(self.device)

        return mask

    def _get_state(self):
        pass

    def _calculate_reward(self):
        pass

    def _modeling(self):
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

        locations = {}
        resources = {}

        source = Source(sim_env, "Source", jobs, locations, monitor)
        sink = Sink(sim_env, "Sink", monitor)

        for _, row in self.df_locations.iterrows():
            name = row["Name"]
            global_index = row["Global_Index"]
            local_index = row["Local_Index"]
            category = row["Category"]
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
            index = row["Index"]
            x_velocity = row["X_Velocity"]
            y_velocity = row["Y_Velocity"]
            initial_coord = (int(row["Initial_X_Coordinate"]), int(row["Initial_Y_Coordinate"]))

            crane = Crane(sim_env, name, index, self.safety_margin, x_velocity, y_velocity, initial_coord, locations, monitor)
            resources[name] = crane

        return sim_env, jobs, locations, resources, monitor


if __name__ == "__main__":
    data_src = DataGenerator()
    env = Factory(data_src)
    state = env.reset()