import torch
import simpy
import copy
import numpy as np
import pandas as pd

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

        self.location_id_to_name = {}
        for i, row in self.df_locations.iterrows():
            self.location_id_to_name[int(row["Global_Index"])] = row["Name"]

        self.resource_id_to_name = {}
        for i, row in self.df_resources.iterrows():
            self.resource_id_to_name[int(row["Index"])] = row["Name"]

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

        self.state = None

    def step(self, action):
        if self.scheduling_mode == "machine":
            location_id = action // self.num_jobs + self.num_inputpoints
            job_id = action % self.num_jobs

            job = self.monitor.remove_from_queue(job_id, scheduling_mode=self.scheduling_mode)
            current_location = job.current_location
            next_location = self.location_id_to_name[location_id]

            self.locations[current_location].call_for_machine_scheduling[job.id].succeed(next_location)
            self.scheduling_mode = "crane"
        else:
            crane_id = action

            job = self.monitor.remove_from_queue(scheduling_mode=self.scheduling_mode)
            current_location = job.current_location
            crane = self.resource_id_to_name.get(crane_id)

            self.locations[current_location].call_for_crane_scheduling[job.id].succeed(crane)
            self.scheduling_mode = "machine"

            mask = self._get_mask()
            if mask.any():
                self.monitor.set_scheduling_flag(scheduling_mode="machine")

        done = False

        while True:
            if self.monitor.machine_scheduling or self.monitor.crane_scheduling:
                while self.sim_env.now in [event[0] for event in self.sim_env._queue]:
                    self.sim_env.step()

                mask = self._get_mask()
                if mask.any():
                    break
                else:
                    self.monitor.machine_scheduling = False

            if len(self.monitor.jobs_after_system) == self.num_jobs:
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

    def _get_mask(self):
        if self.scheduling_mode == "machine":
            num_rows = self.num_machines + self.num_buffers + self.num_outputpoints
            num_columns = self.num_jobs

            mask_machine = np.zeros((num_rows, num_columns), dtype=bool)
            mask_buffer = np.zeros((num_rows, num_columns), dtype=bool)
            mask_output = np.zeros((num_rows, num_columns), dtype=bool)

            for job in self.monitor.queue_for_machine_scheduling.values():
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
                        flag_availability = ~location.check_status() or job.current_location == name
                        flag_accessibility = ~((current_coord[0] < self.safety_margin
                                                and target_coord[0] > self.x_max - self.safety_margin) or \
                                               (current_coord[0] > self.x_max - self.safety_margin
                                                and target_coord[0] < self.safety_margin))

                        if category == 1:
                            if operation is not None:
                                flag_eligibility = int(operation.get_processing_time(local_id)) != 0
                                mask_machine[global_id - self.num_inputpoints, job.id] \
                                    = flag_eligibility & flag_availability & flag_accessibility
                            else:
                                continue
                        elif category == 2:
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

        else:
            mask = np.zeros(self.num_cranes + 1, dtype=bool)

            job = self.monitor.queue_for_crane_scheduling

            if job.current_location == job.next_location:
                mask[self.num_cranes] = 1
            else:
                current_location_coord = self.locations[job.current_location].coord
                next_location_coord = self.locations[job.next_location].coord

                for crane in self.resources.values():
                    if ((crane.id == 0) and (current_location_coord[0] <= self.x_max - self.safety_margin)
                        and (next_location_coord[0] <= self.x_max - self.safety_margin)) or \
                            ((crane.id == 1) and (current_location_coord[0] >= self.safety_margin)
                             and (current_location_coord[0] >= self.safety_margin)):
                        mask[crane.id] = 1

        mask = torch.tensor(mask, dtype=torch.bool).to(self.device)

        return mask

    def _get_state(self):
        if self.scheduling_mode == "machine":
            machine_scheduling_algorithm = self.algorithm[0]

            if machine_scheduling_algorithm == "RL":
                pass
            else:
                num_rows = self.num_machines + self.num_buffers + self.num_outputpoints
                num_columns = self.num_jobs

                data = np.zeros((num_rows, num_columns))

                if machine_scheduling_algorithm == "SPT":
                    pass
                elif machine_scheduling_algorithm == "MOR":
                    pass
                elif machine_scheduling_algorithm == "MWKR":
                    pass
                elif machine_scheduling_algorithm == "RAND":
                    data[:, :] = 1.0
        else:
            crane_scheduling_algorithm = self.algorithm[1]

            if crane_scheduling_algorithm == "RL":
                pass
            else:
                data = np.zeros(self.num_cranes + 1)

                if crane_scheduling_algorithm == "SETT":
                    pass
                elif crane_scheduling_algorithm == "NCR":
                    pass
                elif crane_scheduling_algorithm == "LWKR":
                    pass
                elif crane_scheduling_algorithm == "RAND":
                    data[:] = 1.0

        mask = self._get_mask()

        state = State()
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
                opposite_crane = resources["Crane-1"]
            else:
                opposite_crane = resources["Crane-0"]
            crane.set_opposite_crane(opposite_crane)

        return sim_env, jobs, source, sink, locations, resources, monitor


if __name__ == "__main__":
    from Agent.FlexibleJobShop.heuristic import MachineSchedulingHeuristic
    from Agent.CraneTransportation.heuristic import CraneSchedulingHeuristic

    agent_ms = MachineSchedulingHeuristic()
    agent_cs = CraneSchedulingHeuristic()

    data_src = DataGenerator()
    env = Factory(data_src, algorithm=("RAND","RAND"), record_events=True)

    step = 0

    state = env.reset()

    while True:
        if env.scheduling_mode == "machine":
            action = agent_ms.act(state)
        else:
            action = agent_cs.act(state)

        next_state, reward, done = env.step(action)

        state = next_state
        step += 1

        print(step)

        if done:
            break