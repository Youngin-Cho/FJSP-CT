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
        self.waiting = False
        self.status = "waiting" # "loading", "unloading"
        self.job = None
        self.to_location = None
        self.working_start = None

        self.update_time = 0.0

        self.idle_time = 0.0
        self.empty_travel_time = 0.0
        self.avoiding_time = 0.0

        self.waiting_event = None
        self.process = env.process(self._run())

    def add_to_queue(self, working_order):
        self.queue.append(working_order)

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

        flag_update = False
        if time_elapsed > 0.0:
            if self.idle:
                if self.target_coord[0] != -1.0:
                    flag_update = True
                    x_direction = np.sign(self.target_coord[0] - xcoord)
                    y_direction = np.sign(self.target_coord[1] - ycoord)
            else:
                if not self.waiting:
                    if self.safety_coord[0] != -1.0:
                        flag_update = True
                        x_direction = np.sign(self.safety_coord[0] - xcoord)
                        y_direction = np.sign(self.target_coord[1] - ycoord)
                    else:
                        flag_update = True
                        x_direction = np.sign(self.target_coord[0] - xcoord)
                        y_direction = np.sign(self.target_coord[1] - ycoord)

            if flag_update:
                xcoord = xcoord + time_elapsed * self.x_velocity * x_direction
                ycoord = ycoord + time_elapsed * self.y_velocity * y_direction

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
                    self.monitor.record(self.env.now, event="Waiting_Started",
                                        location=self.location_mapping[self.current_coord].name, resource=self.name)

                self.waiting_event = self.env.event()
                target_coord = yield self.waiting_event

                waiting_finish = self.env.now
                if self.monitor.record_events:
                    self.monitor.record(self.env.now, event="Waiting_Finished",
                                        location=self.location_mapping[self.current_coord].name, resource=self.name)

                self.idle_time += waiting_finish - waiting_start

                if target_coord is not None:
                    self.target_coord = target_coord
                    yield self.env.process(self._moving())
                    self.target_coord = (-1.0, -1.0)

                self.opposite.update_location(self.env.now)
                self.update_location(self.env.now)

            else:
                self.idle = False

                job_id, current_location, next_location = self.queue.pop(0)

                location = self.locations[current_location]

                self.status = "loading"
                self.working_start = self.env.now
                self.target_coord = location.coord
                # self.from_location = self.location_mapping[self.current_coord].name
                self.to_location = location.name

                yield self.env.process(self._moving())
                self.job = self.locations[self.to_location].get(job_id)

                location = self.locations[next_location]

                self.status = "unloading"
                self.working_start = self.env.now
                self.target_coord = location.coord
                # self.from_location = self.location_mapping[self.current_coord].name
                self.to_location = location.name

                yield self.env.process(self._moving())
                self.locations[self.to_location].put(self.job)

                self.target_coord = (-1.0, -1.0)
                # self.from_location = None
                self.to_location = None
                self.job = None

    def _moving(self):
        added_travel_time = 0.0
        while True:
            avoidance, safety_xcoord = self._check_interference()
            if avoidance:
                self.safety_coord = (safety_xcoord, self.target_coord[1])
                opposite_direction = True if np.sign(safety_xcoord - self.current_coord[0]) \
                                             != np.sign(self.target_coord[0] - self.current_coord[0]) else False
                travel_time = self.get_travel_time(self.safety_coord)
                travel_time_opposite = self.opposite.get_travel_time(self.opposite.target_coord)

                if self.monitor.record_events:
                    self.monitor.record(self.env.now, event="Move_from",
                                        location=self.location_mapping[self.current_coord].name, resource=self.name)

                yield self.env.timeout(travel_time)

                self.opposite.update_location(self.env.now)
                self.update_location(self.env.now)

                if self.monitor.record_events:
                    self.monitor.record(self.env.now, event="Move_to",
                                        location=self.location_mapping[self.current_coord].name, resource=self.name)

                self.safety_coord = (-1.0, -1.0)

                if opposite_direction:
                    self.avoiding_time += travel_time
                    added_travel_time += travel_time
                else:
                    if self.status == "loading":
                        self.empty_travel_time += travel_time

                if travel_time_opposite > travel_time:
                    self.waiting = True
                    avoiding_start = self.env.now
                    if self.monitor.record_events:
                        self.monitor.record(self.env.now, event="Avoiding_wait_start",
                                            location=self.location_mapping[self.current_coord].name, resource=self.name)

                    yield self.env.timeout(travel_time_opposite - travel_time)

                    self.opposite.update_location(self.env.now, on_location=True)
                    self.update_location(self.env.now)

                    self.waiting = False
                    avoiding_finish = self.env.now
                    if self.monitor.record_events:
                        self.monitor.record(self.env.now, event="Avoiding_wait_finish",
                                            location=self.location_mapping[self.current_coord].name, resource=self.name)

                    self.avoiding_time += avoiding_finish - avoiding_start
            else:
                if self.monitor.record_events:
                    self.monitor.record(self.env.now, event="Move_from",
                                        location=self.location_mapping[self.current_coord].name, resource=self.name)

                travel_time = self.get_travel_time(self.target_coord)

                if self.opposite.idle:
                    xcoord = (self.current_coord[0] + travel_time * self.x_velocity
                              * np.sign(self.target_coord[0] - self.current_coord[0]))
                    xcoord_opposite = self.opposite.current_coord[0]

                    if self.id == 0 and xcoord > xcoord_opposite - self.safety_margin:
                        flag = True
                        target_coord_opposite = (xcoord + self.safety_margin, self.opposite.current_coord[1])
                    elif self.id == 1 and xcoord < xcoord_opposite + self.safety_margin:
                        flag = True
                        target_coord_opposite = (xcoord - self.safety_margin, self.opposite.current_coord[1])
                    else:
                        flag = False

                    if flag:
                        if not self.opposite.waiting_event.triggered:
                            self.opposite.waiting_event.succeed(target_coord_opposite)

                yield self.env.timeout(travel_time)

                self.opposite.update_location(self.env.now)
                self.update_location(self.env.now, on_location=True)

                if self.monitor.record_events:
                    self.monitor.record(self.env.now, event="Move_to",
                                        location=self.location_mapping[self.current_coord].name, resource=self.name)

                if added_travel_time > 0.0:
                    self.avoiding_time += added_travel_time

                if self.status == "loading":
                    self.empty_travel_time += (travel_time - added_travel_time)

                break

    def _check_interference(self):
        priority_flag = self._check_priority()

        if priority_flag:
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

    def _check_priority(self):
        priority_flag = False
        if self.opposite.idle:
            priority_flag = True
        else:
            if self.status == "loading":
                if self.working_start < self.opposite.working_start:
                    priority_flag = True
                else:
                    priority_flag = False
            elif self.status == "unloading":
                location_list = []

                if self.opposite.status == "loading":
                    location_list.append(self.opposite.to_location)

                for working_order in self.opposite.queue:
                    job_id, current_location, next_location = working_order
                    location = self.locations[current_location]
                    if location.category == 1 or location.category == 2:
                        if location.fully_occupied:
                            location_list.append(location.name)

                if self.to_location in location_list:
                    priority_flag = False
                else:
                    if self.working_start < self.opposite.working_start:
                        priority_flag = True
                    else:
                        priority_flag = False

        return priority_flag


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
    def __init__(self, env, name, jobs, locations, input_points, monitor):
        self.env = env
        self.name = name
        self.jobs = jobs
        self.locations = locations
        self.input_points = input_points
        self.monitor = monitor

        self.process = env.process(self._generate())

        self.sent = 0

    def _generate(self):
        for job in self.jobs:
            self.monitor.jobs_before_system[job.id] = job
            for i, operation in enumerate(job.operations):
                self.monitor.operations_unscheduled[operation.id] = operation

        while True:
            job = self.jobs[self.sent]

            inter_arrival_time = job.arrival_time - self.env.now
            if inter_arrival_time > 0:
                yield self.env.timeout(inter_arrival_time)

            location_name = np.random.choice(self.input_points)
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

        if self.call_for_transporting.get(job_id) is not None:
            del self.call_for_transporting[job_id]

        return job

    def _arrive(self, job):
        self.monitor.jobs_in_system[job.id] = job
        operation = job.get_current_operation()

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name, event="Job_Arrived")

        self.monitor.add_to_queue(job, scheduling_mode="machine")
        self.monitor.set_scheduling_flag(scheduling_mode="machine")
        self.call_for_machine_scheduling[job.id] = self.env.event()
        location_name = yield self.call_for_machine_scheduling[job.id]
        del self.call_for_machine_scheduling[job.id]

        if len(self.locations[location_name].processes) + 1 >= self.locations[location_name].capacity:
            self.locations[location_name].fully_occupied = True

        job.next_location = location_name
        del self.monitor.operations_unscheduled[operation.id]

        self.monitor.add_to_queue(job, scheduling_mode="crane")
        self.monitor.set_scheduling_flag(scheduling_mode="crane")
        self.call_for_crane_scheduling[job.id] = self.env.event()
        crane_name = yield self.call_for_crane_scheduling[job.id]
        del self.call_for_crane_scheduling[job.id]

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name, event="Crane_Called", resource=crane_name)

        crane = self.resources[crane_name]
        crane.add_to_queue((job.id, job.current_location, job.next_location))
        if crane.idle:
            if not crane.waiting_event.triggered:
                crane.waiting_event.succeed()
        else:
            self.call_for_transporting[job.id] = self.env.event()
            yield self.call_for_transporting[job.id]


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

        if self.call_for_transporting.get(job_id) is not None:
            del self.call_for_transporting[job_id]

        if "Output" in job.next_location:
            if len(self.monitor.operations_waiting) > 0:
                self.monitor.machine_scheduling = True

        return job

    def check_status(self):
        fully_occupied = self.fully_occupied
        return fully_occupied

    def get_available_time(self):
        available_time = self.env.now

        for job in self.jobs_in_process.values():
            operation = job.get_current_operation()
            proc_time = operation.get_processing_time(machine_id=self.local_id)
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

        processing_time = operation.get_processing_time(self.local_id)
        operation.start_time = self.env.now
        operation.allocated_machine = self.name
        yield self.env.timeout(processing_time)

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name,
                                operation=operation.name, event="Working Finished")

        self.working_time += processing_time
        self.completion_time = self.env.now
        operation.finish_time = self.env.now

        job.step += 1
        self.monitor.operations_done[operation.id] = operation
        del self.monitor.operations_working[operation.id]
        del self.jobs_in_process[job.id]
        self.jobs_after_process[job.id] = job

        self.idle = True

        self.monitor.add_to_queue(job, scheduling_mode="machine")
        self.monitor.set_scheduling_flag(scheduling_mode="machine")
        self.call_for_machine_scheduling[job.id] = self.env.event()
        location_name = yield self.call_for_machine_scheduling[job.id]
        del self.call_for_machine_scheduling[job.id]

        if len(self.locations[location_name].processes) + 1 >= self.locations[location_name].capacity:
            self.locations[location_name].fully_occupied = True

        job.next_location = location_name

        self.monitor.add_to_queue(job, scheduling_mode="crane")
        self.monitor.set_scheduling_flag(scheduling_mode="crane")
        self.call_for_crane_scheduling[job.id] = self.env.event()
        crane_name = yield self.call_for_crane_scheduling[job.id]
        del self.call_for_crane_scheduling[job.id]

        if crane_name is not None:
            self.fully_occupied = False

            if self.monitor.record_events:
                self.monitor.record(self.env.now, location=self.name, job=job.name, event="Crane_Called", resource=crane_name)

            crane = self.resources[crane_name]
            crane.add_to_queue((job.id, job.current_location, job.next_location))
            if crane.idle:
                if not crane.waiting_event.triggered:
                    crane.waiting_event.succeed()
            else:
                self.call_for_transporting[job.id] = self.env.event()
                yield self.call_for_transporting[job.id]
        else:
            self.get(job.id)
            self.put(job)


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

        if self.call_for_transporting.get(job_id) is not None:
            del self.call_for_transporting[job_id]

        return job

    def check_status(self):
        fully_occupied = self.fully_occupied
        return fully_occupied

    def _wait(self, job):
        operation = job.get_current_operation()
        self.monitor.operations_waiting[operation.id] = operation

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name,
                                operation=operation.name, event="Waiting Started")

        operation.waiting_start = self.env.now
        self.monitor.add_to_queue(job, scheduling_mode="machine")
        self.call_for_machine_scheduling[job.id] = self.env.event()
        location_name = yield self.call_for_machine_scheduling[job.id]
        del self.call_for_machine_scheduling[job.id]

        del self.monitor.operations_waiting[operation.id]

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name,
                                operation=operation.name, event="Waiting Finished")

        del self.jobs_in_process[job.id]
        self.jobs_after_process[job.id] = job

        if len(self.locations[location_name].processes) + 1 >= self.locations[location_name].capacity:
            self.locations[location_name].fully_occupied = True

        job.next_location = location_name

        self.monitor.add_to_queue(job, scheduling_mode="crane")
        self.monitor.set_scheduling_flag(scheduling_mode="crane")
        self.call_for_crane_scheduling[job.id] = self.env.event()
        crane_name = yield self.call_for_crane_scheduling[job.id]
        del self.call_for_crane_scheduling[job.id]

        self.fully_occupied = False

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name, event="Crane_Called", resource=crane_name)

        crane = self.resources[crane_name]
        crane.add_to_queue((job.id, job.current_location, job.next_location))
        if crane.idle:
            if not crane.waiting_event.triggered:
                crane.waiting_event.succeed()
        else:
            self.call_for_transporting[job.id] = self.env.event()
            yield self.call_for_transporting[job.id]


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

        self.fully_occupied = False
        self.completion_time = 0

    def put(self, job):
        job.current_location = self.name
        job.next_location = None

        self.processes[job.id] = self.env.process(self._departure(job))

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

        del self.processes[job.id]
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
        self.queue_for_crane_scheduling = None
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
        self.resource = []

    def set_scheduling_flag(self, scheduling_mode='machine'):
        if scheduling_mode == 'machine':
            self.machine_scheduling = True
        elif scheduling_mode == 'crane':
            self.crane_scheduling = True

    def add_to_queue(self, job, scheduling_mode='machine'):
        if scheduling_mode == "machine":
            self.queue_for_machine_scheduling[job.id] = job
        elif scheduling_mode == "crane":
            self.queue_for_crane_scheduling = job

    def remove_from_queue(self, job_id=None, scheduling_mode='machine'):
        if scheduling_mode == "machine":
            assert job_id is not None

            job = self.queue_for_machine_scheduling[job_id]
            del self.queue_for_machine_scheduling[job_id]

            self.machine_scheduling = False

            return job

        elif scheduling_mode == "crane":
            job = self.queue_for_crane_scheduling
            self.queue_for_crane_scheduling = None

            self.crane_scheduling = False

            return job

    def record(self, time, location=None, job=None, operation=None, event=None, resource=None):
        self.time.append(time)
        self.location.append(location)
        self.job.append(job)
        self.operation.append(operation)
        self.event.append(event)
        self.resource.append(resource)

    def get_logs(self, file_path=None):
        df_log = pd.DataFrame(columns=['Time', 'Location', 'Job', 'Operation', 'Event', 'Resource'])
        df_log['Time'] = self.time
        df_log['Location'] = self.location
        df_log['Job'] = self.job
        df_log['Operation'] = self.operation
        df_log['Event'] = self.event
        df_log['Resource'] = self.resource

        if file_path is not None:
            df_log.to_excel(file_path, sheet_name="logs", index=False)

        return df_log