import numpy as np
import pandas as pd


class Crane:
    def __init__(self, env, name, id, safety_margin, x_velocity, y_velocity, initial_coord, locations, monitor):
        self.env = env
        self.name = name
        self.id = id
        self.locations = locations
        self.monitor = monitor
        self.safety_margin = safety_margin
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity
        # self.max_x = max_x
        # self.max_y = max_y

        self.opposite = None
        self.queue = []
        self.location_mapping = {}

        self.current_coord = initial_coord
        self.target_coord = (-1.0, -1.0)
        self.safety_coord = (-1.0, -1.0)

        self.idle = True
        self.status = "waiting" # "loading", "unloading"
        self.job = None
        self.from_location = None
        self.to_location = None
        self.working_start = None

        self.update_time = 0.0

        self.idle_time = 0.0
        self.empty_travel_time = 0.0
        self.avoiding_time = 0.0

        self.waiting_event = None
        self.process = env.process(self._run())

    def add_to_queue(self, location_name):
        self.queue.append(location_name)

    def set_opposite_crane(self, crane):
        self.opposite = crane

    def get_travel_time(self, target_coord):
        x_travel_time = abs(target_coord[0] - self.current_coord[0]) / self.x_velocity
        y_travel_time = abs(target_coord[1] - self.current_coord[1]) / self.y_velocity
        travel_time = max(x_travel_time, y_travel_time)
        return travel_time

    def update_location(self, time, on_location=False):
        xcoord = self.current_coord[0]
        ycoord = self.current_coord[1]
        time_elapsed = time - self.update_time

        if time_elapsed > 0.0 and self.target_coord[0] != -1.0:
            if self.safety_coord[0] != -1.0:
                x_direction = np.sign(self.safety_coord[0] - xcoord)
                # x_limit = self.safety_coord[0]
            else:
                x_direction = np.sign(self.target_coord[0] - xcoord)
                # x_limit = self.target_coord[0]

            y_direction = np.sign(self.target_coord[1] - ycoord)
            # y_limit = self.target_coord[1]

            xcoord = xcoord + time_elapsed * self.x_velocity * x_direction
            ycoord = ycoord + time_elapsed * self.y_velocity * y_direction

            # if x_direction == 1:
            #     x_coord = np.clip(xcoord, a_min=1, a_max=x_limit)
            # else:
            #     x_coord = np.clip(xcoord, a_min=x_limit, a_max=self.max_x)
            #
            # if y_direction == 1:
            #     y_coord = np.clip(ycoord, a_min=1, a_max=y_limit)
            # else:
            #     y_coord = np.clip(ycoord, a_min=y_limit, a_max=self.max_y)

        self.update_time = time
        self.current_coord = (xcoord, ycoord)

        if on_location:
            self.current_coord = (int(xcoord), int(ycoord))

    def _initialize(self):
        for name, location in self.locations.items():
            self.location_mapping[location.coord] = location

    def _run(self):
        self._initialize()

        while True:
            if len(self.queue) == 0:
                self.idle = True
                self.status = "waiting"

                waiting_start = self.env.now
                if self.monitor.record_events:
                    self.monitor.record(self.env.now, "Waiting_Started", crane=self.name,
                                        location=self.location_mapping[self.current_coord].name)

                self.waiting_event = self.env.event()
                target_coord = yield self.waiting_event

                waiting_finish = self.env.now
                if self.monitor.record_events:
                    self.monitor.record(self.env.now, "Waiting_Finished", crane=self.name,
                                        location=self.location_mapping[self.current_coord].name)

                self.idle_time += waiting_finish - waiting_start

                if target_coord is not None:
                    yield self.env.process(self._moving(target_coord))
                    self.opposite.update_location(self.env.now)
                    self.update_location(self.env.now)
                    self.target_coord = (-1.0, -1.0)
            else:
                self.idle = False

                location_name, job_id = self.queue.pop(0)
                location = self.locations[location_name]

                self.status = "loading"
                self.working_start = self.env.now
                self.target_coord = location.coord
                self.from_location = self.location_mapping[self.current_coord].name
                self.to_location = self.location_mapping[self.target_coord].name

                yield self.env.process(self._moving(location.coord))
                self.job = self.locations[self.to_location].get_job()

                self.status = "unloading"
                self.working_start = self.env.now
                self.target_coord = self.locations[self.job.next_location].coord
                self.from_location = self.location_mapping[self.current_coord].name
                self.to_location = self.location_mapping[self.target_coord].name

                yield self.env.process(self._moving(location.coord))
                self.locations[self.to_location].put(self.job)

                self.target_coord = (-1.0, -1.0)
                self.from_location = None
                self.to_location = None
                self.job = None

    def _moving(self, target_coord):
        self.target_coord = target_coord

        added_travel_time = 0.0
        while True:
            avoidance, safety_xcoord = self._check_interference()
            if avoidance:
                self.safety_coord = (safety_xcoord, target_coord[1])
                opposite_direction = True if np.sign(safety_xcoord - self.current_coord[0]) \
                                             != np.sign(self.target_coord[0] - self.current_coord[0]) else False
                travel_time = self.get_travel_time(self.safety_coord)
                travel_time_opposite = self.opposite.get_travel_time(self.opposite.target_coord)

                if self.monitor.record_events:
                    self.monitor.record(self.env.now, "Move_from", crane=self.name,
                                        location=self.location_mapping[self.current_coord].name, plate=None)

                yield self.env.timeout(travel_time)
                self.opposite.update_location(self.env.now)
                self.update_location(self.env.now)

                if self.monitor.record_events:
                    self.monitor.record(self.env.now, "Move_to", crane=self.name,
                                        location=self.location_mapping[self.current_coord].name, plate=None)

                self.safety_coord = (-1.0, -1.0)

                if opposite_direction:
                    self.avoiding_time += travel_time
                    added_travel_time += travel_time
                else:
                    if self.status == "loading":
                        self.empty_travel_time += travel_time

                if travel_time_opposite > travel_time:
                    avoiding_start = self.env.now
                    if self.monitor.record_events:
                        self.monitor.record(self.env.now, "Avoiding_wait_start", crane=self.name,
                                            location=self.location_mapping[self.current_coord].name, plate=None)

                    yield self.env.timeout(travel_time_opposite - travel_time)
                    self.opposite.update_location(self.env.now, on_location=True)
                    self.update_location(self.env.now)

                    avoiding_finish = self.env.now
                    if self.monitor.record_events:
                        self.monitor.record(self.env.now, "Avoiding_wait_finish", crane=self.name,
                                            location=self.location_mapping[self.current_coord].name, plate=None)

                    self.avoiding_time += avoiding_finish - avoiding_start
            else:
                if self.monitor.record_events:
                    self.monitor.record(self.env.now, "Move_from", crane=self.name,
                                        location=self.location_mapping[self.current_coord].name, plate=None)

                travel_time = self.get_travel_time(self.target_coord)

                if self.opposite.idle:
                    xcoord = (self.current_coord[0] + travel_time * self.x_velocity
                              * np.sign(self.target_coord[0] - self.current_coord[0]))
                    xcoord_opposite = self.opposite.current_coord[0]

                    if self.id == 0 and xcoord > xcoord_opposite - self.safety_margin:
                        target_coord_opposite = (xcoord + self.safety_margin, self.opposite.current_coord[1])
                    elif self.id == 1 and xcoord < xcoord_opposite + self.safety_margin:
                        target_coord_opposite = (xcoord - self.safety_margin, self.opposite.current_coord[1])

                    if not self.opposite.waiting_event.triggered:
                        self.opposite.waiting_event.succeed(target_coord_opposite)

                yield self.env.timeout(travel_time)
                self.opposite.update_location(self.env.now)
                self.update_location(self.env.now, on_location=True)

                if self.monitor.record_events:
                    self.monitor.record(self.env.now, "Move_to", crane=self.name,
                                        location=self.location_mapping[self.current_coord].name, plate=None)

                if added_travel_time > 0.0:
                    self.avoiding_time += added_travel_time

                if self.status == "loading":
                    self.empty_travel_time += (travel_time - added_travel_time)

                break

    def _check_interference(self):
        flag = False
        if self.opposite.idle:
            flag = True
        else:
            if self.working_start < self.opposite.working_start:
                flag = True

        if flag:
            avoidance = False
            safety_xcoord = None
        else:
            dx = self.target_coord[0] - self.current_coord[0]
            dy = self.target_coord[1] - self.current_coord[1]
            direction = np.sign(dx)
            travel_time = max(abs(dx) / self.x_velocity, abs(dy) / self.y_velocity)

            dx_opposite = self.opposite.target_coord[0] - self.opposite.current_coord[0]
            dy_opposite = self.opposite.target_coord[1] - self.opposite.current_coord[1]
            direction_opposite = np.sign(dx_opposite)
            travel_time_opposite = max(abs(dx_opposite) / self.opposite.x_velocity,
                                       abs(dy_opposite) / self.opposite.y_velocity)

            min_travel_time = min(travel_time, travel_time_opposite)
            xcoord = self.current_coord[0] + min_travel_time * self.x_velocity * direction
            xcoord_opposite = (self.opposite.current_coord[0]
                               + min_travel_time * self.opposite.x_velocity * direction_opposite)

            avoidance = False
            safety_xcoord = None

            if (self.id == 0 and xcoord > xcoord_opposite - self.safety_margin) \
                    or (self.id == 1 and xcoord < xcoord_opposite + self.safety_margin):
                avoidance = True
                if self.id == 0:
                    safety_xcoord = self.opposite.target_coord[0] - self.safety_margin
                else:
                    safety_xcoord = self.opposite.target_coord[0] + self.safety_margin

        return avoidance, safety_xcoord


class Operation:
    def __init__(self, name, id, options):
        self.name = name
        self.id = id
        self.options = options

        self.progress = 0.0
        self.start_time = None
        self.finish_time = None
        self.allocated_machine = None

    def get_processing_time(self, machine_id):
        return self.options[machine_id]


class Job:
    def __init__(self, name, id, arrival_time, operations):
        self.name = name
        self.id = id
        self.arrival_time = arrival_time
        self.operations = operations

        self.step = 0
        self.current_location = None
        self.next_location = None
        self.status = None # 'waiting', 'processing', 'completed'

    def get_current_operation(self):
        if len(self.operations) == self.step:
            operation = None
        else:
            operation = self.operations[self.step]
        return operation


class Source:
    def __init__(self, env, name, jobs, locations, monitor):
        self.env = env
        self.name = name
        self.jobs = jobs
        self.locations = locations
        self.monitor = monitor

        self.input_locations = [location.name for location in locations.items() if location.category == 0]
        self.process = env.process(self._generate())

        self.sent = 0

    def _initialize(self):
        for job in self.jobs:
            self.monitor.jobs_before_system[job.id] = job
            for i, operation in enumerate(job.operations):
                self.monitor.operations_unscheduled[operation.id] = operation

    def _generate(self):
        self._initialize()

        while True:
            job = self.jobs[self.sent]

            inter_arrival_time = job.arrival_time - self.env.now
            if inter_arrival_time > 0:
                yield self.env.timeout(inter_arrival_time)

            location_name = np.random.choice(self.input_locations)
            del self.monitor.jobs_before_system[job.id]

            self.locations[location_name].put(job)
            self.sent += 1

            if len(self.jobs) == self.sent:
                break


class InputPoint:
    def __init__(self, env, name, global_id, local_id, category, coord,
                 locations, resources, monitor, capacity=float('inf')):
        self.env = env
        self.name = name
        self.global_id = global_id
        self.local_id = local_id
        self.category = category
        self.coord = coord
        self.locations = locations
        self.resources = resources
        self.monitor = monitor
        self.capacity = capacity

        self.processes = {}
        self.jobs_after_process = {}

        self.call_for_machine_scheduling = {}
        self.call_for_crane_scheduling = {}
        self.call_for_transporting = {}

    def put(self, job):
        job.current_location = self.name
        job.next_location = None

        self.processes[job.id] = self.env.process(self._arrive(job))
        self.jobs_after_process[job.id] = job

    def get(self, job_id):
        job = self.jobs_after_process[job_id]

        del self.processes[job.id]
        del self.jobs_after_process[job_id]

        return job

    def _arrive(self, job):
        self.monitor.jobs_in_system[job.id] = job
        operation = job.get_current_operation()

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name, event="Job_Arrived")

        self.monitor.add_to_queue(job, from_buffer=False, scheduling_tag="machine")
        self.call_for_machine_scheduling[job.name] = self.env.event()
        location_name = yield self.call_for_machine_scheduling[job.name]
        del self.call_for_machine_scheduling[job.name]

        if len(self.locations[location_name].processes) + 1 == self.locations[location_name].capacity:
            self.locations[location_name].fully_occupied = True

        job.next_location = location_name
        del self.monitor.operations_unscheduled[operation.id]

        self.monitor.add_to_queue(job, scheduling_tag="crane")
        self.call_for_crane_scheduling[job.name] = self.env.event()
        crane_name = yield self.call_for_crane_scheduling[job.name]
        del self.call_for_crane_scheduling[job.name]

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name, event="Crane_Called")

        crane = self.resources[crane_name]
        crane.add_to_queue((self.name, job.id))
        if crane.idle:
            if not crane.waiting_event.triggered:
                crane.waiting_event.succeed()
        else:
            self.call_for_transporting[job.name] = self.env.event()
            yield self.call_for_transporting[job.name]


class Machine:
    def __init__(self, env, name, global_id, local_id, category, coord,
                 locations, resources, monitor, capacity=1):
        self.env = env
        self.name = name
        self.global_id = global_id
        self.local_id = local_id
        self.category = category
        self.coord = coord
        self.locations = locations
        self.resources = resources
        self.monitor = monitor
        self.capacity = capacity

        self.processes = {}
        self.jobs_in_process = {}
        self.jobs_after_process = {}

        self.call_for_machine_scheduling = {}
        self.call_for_crane_scheduling = {}
        self.call_for_transporting = {}

        self.idle = True
        self.fully_occupied = False
        self.working_time = 0.0
        self.completion_time = 0.0

    def put(self, job):
        job.current_location = self.name
        job.next_location = None

        self.processes[job.id] = self.env.process(self._work(job))
        self.jobs_in_process[job.id] = job

    def get(self, job_id):
        job = self.jobs_after_process[job_id]

        del self.processes[job.id]
        del self.jobs_after_process[job_id]

        self.fully_occupied = False

        if "Output" in job.next_location:
            if len(self.monitor.operations_waiting) > 0:
                self.monitor.machine_scheduling = True

        return job

    def check_status(self):
        fully_occupied = self.fully_occupied
        return fully_occupied

    def get_available_time(self):
        available_time = self.env.now

        for id, job in self.jobs_in_process.items():
            operation = job.get_current_operation()
            proc_time = operation.get_processing_time(machine_id=self.id)
            temp = operation.start_time + proc_time

            if temp > available_time:
                available_time = temp

        return available_time



    def _work(self, job):
        if len(self.jobs_in_process) == self.capacity:
            self.idle = False

        operation = job.get_current_operation()
        self.monitor.operations_working[operation.id] = operation

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name,
                                operation=operation.name, event="Working_Started")

        processing_time = operation.get_processing_time(self.id)
        operation.working_start = self.env.now
        operation.allocated_machine = self.name
        yield self.env.timeout(processing_time)

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name,
                                operation=operation.name, event="Working Finished")

        self.working_time += processing_time
        self.completion_time = self.env.now
        operation.working_finish = self.env.now

        job.step += 1
        self.monitor.operations_done[operation.id] = operation
        del self.monitor.operations_in_machine[operation.id]
        del self.jobs_in_process[job.id]
        self.jobs_after_process[job.id] = job

        self.idle = True

        self.monitor.add_to_queue(job, from_buffer=False, scheduling_tag="machine")
        self.call_for_machine_scheduling[job.name] = self.env.event()
        location_name = yield self.call_for_machine_scheduling[job.name]
        del self.call_for_machine_scheduling[job.name]

        if len(self.locations[location_name].processes) + 1 == self.locations[location_name].capacity:
            self.locations[location_name].fully_occupied = True

        job.next_location = location_name

        self.monitor.add_to_queue(job, scheduling_tag="crane")
        self.call_for_crane_scheduling[job.name] = self.env.event()
        crane_name = yield self.call_for_crane_scheduling[job.name]
        del self.call_for_crane_scheduling[job.name]

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name, event="Crane_Called")

        crane = self.resources[crane_name]
        crane.add_to_queue((self.name, job.id))
        if crane.idle:
            if not crane.waiting_event.triggered:
                crane.waiting_event.succeed()
        else:
            self.call_for_transporting[job.name] = self.env.event()
            yield self.call_for_transporting[job.name]


class Buffer:
    def __init__(self, env, name, global_id, local_id, category, coord,
                 locations, resources, monitor, capacity=float('inf')):
        self.env = env
        self.name = name
        self.global_id = global_id
        self.local_id = local_id
        self.category = category
        self.coord = coord
        self.locations = locations
        self.resources = resources
        self.monitor = monitor
        self.capacity = capacity

        self.processes = {}
        self.jobs_in_process = {}
        self.jobs_after_process = {}

        self.call_for_machine_scheduling = {}
        self.call_for_crane_scheduling = {}
        self.call_for_transporting = {}

        self.fully_occupied = False

    def put(self, job):
        job.current_location = self.name
        job.next_location = None

        self.processes[job.id] = self.env.process(self._wait(job))
        self.jobs_in_process[job.id] = job

    def get(self, job_id):
        job = self.jobs_after_process[job_id]

        del self.processes[job.id]
        del self.jobs_after_process[job_id]

        self.fully_occupied = False

        return job

    def check_status(self):
        fully_occupied = self.fully_occupied
        return fully_occupied

    def _wait(self, job):
        operation = job.get_current_operation()
        self.monitor.operations_in_buffer[operation.id] = operation

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name,
                                operation=operation.name, event="Waiting Started")

        operation.waiting_start = self.env.now
        self.monitor.add_to_queue(job, from_buffer=True, scheduling_tag="machine")
        self.call_for_machine_scheduling[job.name] = self.env.event()
        location_name = yield self.call_for_machine_scheduling[job.name]
        del self.call_for_machine_scheduling[job.name]

        del self.monitor.operations_waiting[operation.id]

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name,
                                operation=operation.name, event="Waiting Finished")

        del self.jobs_in_process[job.id]
        self.jobs_after_process[job.id] = job

        if len(self.locations[location_name].processes) + 1 == self.locations[location_name].capacity:
            self.locations[location_name].fully_occupied = True

        job.next_location = location_name

        self.monitor.add_to_queue(job, scheduling_tag="crane")
        self.call_for_crane_scheduling[job.name] = self.env.event()
        crane_name = yield self.call_for_crane_scheduling[job.name]
        del self.call_for_crane_scheduling[job.name]

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name, event="Crane_Called")

        crane = self.resources[crane_name]
        crane.add_to_queue((self.name, job.id))
        if crane.idle:
            if not crane.waiting_event.triggered:
                crane.waiting_event.succeed()
        else:
            self.call_for_transporting[job.name] = self.env.event()
            yield self.call_for_transporting[job.name]


class OutputPoint:
    def __init__(self, env, name, global_id, local_id, category, coord,
                 sink, monitor, capacity=float('inf')):
        self.env = env
        self.name = name
        self.global_id = global_id
        self.local_id = local_id
        self.category = category
        self.coord = coord
        self.sink = sink
        self.monitor = monitor
        self.capacity = capacity

        self.processes = {}
        self.jobs_in_process = {}

        self.fully_occupied = False
        self.completion_time = 0

    def put(self, job):
        job.current_location = self.name
        job.next_location = None

        self.processes[job.id] = self.env.process(self._departure(job))
        self.jobs_in_process[job.id] = job

    def check_status(self):
        fully_occupied = self.fully_occupied
        return fully_occupied

    def _departure(self, job):
        self.monitor.jobs_after_system[job.id] = job
        del self.monitor.jobs_in_system[job.id]

        self.completion_time = self.env.now

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name, event="Job_Completed")

        yield self.env.timeout(0)

        self.sink.put(job)
        self.fully_occupied = False


class Sink:
    def __init__(self, env, name, monitor):
        self.env = env
        self.name = name
        self.monitor = monitor

    def put(self, job):
        job.current_location = self.name
        job.next_location = None

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name, event="Job_Degenerated")


class Monitor:
    def __init__(self, record_events=True):
        self.record_events = record_events

        self.queue_for_machine_scheduling = {}
        self.queue_for_crane_scheduling = {}
        self.machine_scheduling = False
        self.crane_scheduling = False

        self.operations_unscheduled = {}
        self.operations_working = {}
        self.operations_waiting = {}
        self.operations_done = {}

        self.jobs_before_system = {}
        self.jobs_in_system = {}
        self.jobs_after_system = {}

        self.time = []
        self.location = []
        self.job = []
        self.operation = []
        self.event = []
        self.info = []

    def add_to_queue(self, job, from_buffer=False, scheduling_tag='machine'):
        if scheduling_tag == "machine":
            self.queue_for_machine_scheduling[job.id] = job
            if not self.machine_scheduling and not from_buffer:
                self.machine_scheduling = True
        elif scheduling_tag == "crane":
            self.queue_for_crane_scheduling[job.id] = job

    def remove_from_queue(self, job_id, scheduling_tag='machine'):
        if scheduling_tag == "machine":
            job = self.queue_for_machine_scheduling[job_id]
            del self.queue_for_machine_scheduling[job_id]

            new_operations = []
            for job_in_queue in self.queue_for_machine_scheduling.values():
                operation = job_in_queue.get_current_operation()
                if not operation.id in self.operations_waiting.keys():
                    new_operations.append(operation.id)

            if len(new_operations) == 0:
                self.machine_scheduling = False

            return job

        elif scheduling_tag == "crane":
            pass

    def record(self, time, location=None, job=None, operation=None, event=None, info=None):
        self.time.append(time)
        self.location.append(location)
        self.job.append(job)
        self.operation.append(operation)
        self.event.append(event)

    def get_logs(self, file_path=None):
        df_log = pd.DataFrame(columns=['Time', 'Location', 'Job', 'Operation', 'Event'])
        df_log['Time'] = self.time
        df_log['Location'] = self.location
        df_log['Job'] = self.job
        df_log['Operation'] = self.operation
        df_log['Event'] = self.event

        if file_path is not None:
            df_log.to_excel(file_path, index=False)

        return df_log