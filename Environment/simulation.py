import simpy
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

        self.x_min = int(np.min([location.coord[0] for location in locations.values()]))
        self.y_min = int(np.min([location.coord[1] for location in locations.values()]))
        self.x_max = int(np.max([location.coord[0] for location in locations.values()]))
        self.y_max = int(np.max([location.coord[1] for location in locations.values()]))

        self.opposite = None
        self.queue = []
        self.location_mapping = {}

        self.current_coord = initial_coord
        self.target_coord = (-1.0, -1.0)
        self.safety_coord = (-1.0, -1.0)

        self.idle = True
        self.waiting = False
        self.blocked = False
        self.blocked_expected = False
        self.priority_queue = []
        self.status = "waiting" # "loading", "unloading"
        self.job = None
        self.to_location = None
        self.current_working_order = None
        self.working_start = None

        self.update_time = 0.0

        self.idle_time = 0.0
        self.empty_travel_time = 0.0
        self.avoiding_time = 0.0

        self.waiting_event = None
        self.moving_process = None
        self.process = env.process(self._run())

    def add_to_queue(self, working_order):
        for index, temp in enumerate(self.queue):
            if temp[0] == working_order[0]:
                del self.queue[index]
                break

        self.queue.append(working_order)

        def get_mapping(x):
            mapping_idx_to_cnt = {}
            to_location_cnt = {}
            for index, temp in enumerate(x):
                cnt = to_location_cnt.get(temp[2])
                if cnt is not None:
                    mapping_idx_to_cnt[index] = cnt + 1
                    to_location_cnt[temp[2]] += 1
                else:
                    mapping_idx_to_cnt[index] = 1
                    to_location_cnt[temp[2]] = 1
            return mapping_idx_to_cnt

        reorder_flag = True
        mapping_idx_to_cnt = get_mapping(self.queue)
        basis = working_order[1]
        new_basis = working_order[1]
        while reorder_flag:
            for index, temp in enumerate(self.queue):
                if temp[2] == basis:
                    location = self.locations[temp[2]]
                    if location.fully_occupied and mapping_idx_to_cnt[index] == location.capacity:
                        del self.queue[index]
                        self.queue.append(temp)
                        new_basis = temp[1]
                        mapping_idx_to_cnt = get_mapping(self.queue)
                        break

            if new_basis == basis:
                reorder_flag = False
            else:
                basis = new_basis

        for index, temp in enumerate(self.opposite.queue):
            if temp[0] == working_order[0]:
                del self.opposite.queue[index]
                break

        queue_to = set([temp[2] for temp in self.queue if self.locations[temp[2]].category == 1])
        queue_from_opposite = set([temp[1] for temp in self.opposite.queue if self.locations[temp[1]].category == 1])
        self.blocked_expected = len(set.intersection(queue_to, queue_from_opposite)) > 0

        if (self.opposite.status == "unloading") and (self.opposite.to_location in [temp[1] for temp in self.queue]):
            self.opposite.moving_process.interrupt()

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

                    x_limit = self.target_coord[0]
                    y_limit = self.target_coord[1]
            else:
                if not self.waiting:
                    if self.safety_coord[0] != -1.0:
                        flag_update = True
                        x_direction = np.sign(self.safety_coord[0] - xcoord)
                        y_direction = np.sign(self.target_coord[1] - ycoord)

                        x_limit = self.safety_coord[0]
                        y_limit = self.safety_coord[1]
                    else:
                        flag_update = True
                        x_direction = np.sign(self.target_coord[0] - xcoord)
                        y_direction = np.sign(self.target_coord[1] - ycoord)

                        x_limit = self.target_coord[0]
                        y_limit = self.target_coord[1]

            if flag_update:
                xcoord = xcoord + time_elapsed * self.x_velocity * x_direction
                ycoord = ycoord + time_elapsed * self.y_velocity * y_direction

                if x_direction == 1:
                    xcoord = np.clip(xcoord, a_min=self.x_min, a_max=x_limit)
                else:
                    xcoord = np.clip(xcoord, a_min=x_limit, a_max=self.x_max)

                if y_direction == 1:
                    ycoord = np.clip(ycoord, a_min=self.y_min, a_max=y_limit)
                else:
                    ycoord = np.clip(ycoord, a_min=y_limit, a_max=self.y_max)

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

                if not self.opposite.idle:
                    xcoord = self.current_coord[0]
                    xcoord_opposite = self.opposite.target_coord[0]

                    if self.id == 0 and xcoord > xcoord_opposite - self.safety_margin:
                        flag = True
                        self.target_coord = (xcoord_opposite - self.safety_margin, self.current_coord[1])
                    elif self.id == 1 and xcoord < xcoord_opposite + self.safety_margin:
                        flag = True
                        self.target_coord = (xcoord_opposite + self.safety_margin, self.current_coord[1])
                    else:
                        flag = False

                    if flag:
                        self.priority_queue.append(self.name)
                        self.opposite.priority_queue.append(self.name)

                        yield self.env.process(self._moving())
                        self.target_coord = (-1.0, -1.0)

                if len(self.queue) == 0:
                    waiting_start = self.env.now
                    if self.monitor.use_recording:
                        self.monitor.record(self.env.now,
                                            event="Waiting_Started",
                                            resource=self.name,
                                            current_coord=self.current_coord)

                    self.waiting_event = self.env.event()
                    target_coord = yield self.waiting_event

                    waiting_finish = self.env.now
                    if self.monitor.use_recording:
                        self.monitor.record(self.env.now,
                                            event="Waiting_Finished",
                                            resource=self.name,
                                            current_coord=self.current_coord)

                    self.idle_time += waiting_finish - waiting_start

                    if target_coord is not None:
                        self.priority_queue.append(self.name)
                        self.opposite.priority_queue.append(self.name)

                        self.target_coord = target_coord
                        yield self.env.process(self._moving())
                        self.target_coord = (-1.0, -1.0)

                    self.opposite.update_location(self.env.now)
                    self.update_location(self.env.now)

            else:
                self.idle = False

                working_order = self.queue.pop(0)
                self.current_working_order = working_order
                job_id, current_location, next_location = self.current_working_order

                if self.monitor.use_recording:
                    self.monitor.record(self.env.now,
                                        event="Order_Assigned",
                                        resource=self.name,
                                        current_order=working_order,
                                        current_coord=self.current_coord,
                                        destination=current_location,
                                        queue=self.queue[:])

                location = self.locations[current_location]
                location.reserve_job(job_id)

                self.status = "loading"
                self.working_start = self.env.now
                self.target_coord = location.coord
                self.to_location = location.name

                self.priority_queue.append(self.name)
                self.opposite.priority_queue.append(self.name)

                self.moving_process = self.env.process(self._moving())
                yield self.moving_process
                self.job = self.locations[self.to_location].get(job_id)

                if self.monitor.use_recording:
                    self.monitor.record(self.env.now,
                                        event="Get",
                                        location=self.locations[self.to_location].name,
                                        resource=self.name,
                                        current_order=working_order,
                                        current_coord=self.current_coord,
                                        item=self.job.name)

                self.priority_queue.append(self.name)
                self.opposite.priority_queue.append(self.name)

                location = self.locations[next_location]

                self.working_start = self.env.now
                self.target_coord = location.coord
                self.to_location = location.name

                self.moving_process = self.env.process(self._moving())
                yield self.moving_process
                self.locations[self.to_location].put(self.job)

                if self.monitor.use_recording:
                    self.monitor.record(self.env.now,
                                        event="Put",
                                        location=self.locations[self.to_location].name,
                                        resource=self.name,
                                        current_order=working_order,
                                        current_coord=self.current_coord,
                                        item=self.job.name)

                self.target_coord = (-1.0, -1.0)
                self.to_location = None
                self.job = None

    def _moving(self):
        added_travel_time = 0.0

        cnt = 0
        while True:
            cnt += 1
            if cnt > 1000:
                self.monitor.get_logs("./temp.xlsx")
                print("Error!")
                assert cnt < 1000

            avoidance, safety_xcoord = self._check_interference()

            if self.monitor.use_recording:
                self.monitor.record(self.env.now,
                                    event="Check_Priority",
                                    resource=self.name,
                                    current_order=self.current_working_order,
                                    current_coord=self.current_coord,
                                    item=self.job.name if self.job is not None else None,
                                    blocked=self.blocked,
                                    avoidance=avoidance,
                                    destination=self.to_location)

            try:
                if avoidance:
                    self.safety_coord = (safety_xcoord, self.target_coord[1])
                    opposite_direction = True if np.sign(safety_xcoord - self.current_coord[0]) \
                                                 != np.sign(self.target_coord[0] - self.current_coord[0]) else False
                    travel_time = self.get_travel_time(self.safety_coord)
                    travel_time_opposite = self.opposite.get_travel_time(self.opposite.target_coord)

                    if self.monitor.use_recording:
                        self.monitor.record(self.env.now,
                                            event="Move_from",
                                            location=self.location_mapping[self.current_coord].name
                                                if self.location_mapping.get(self.current_coord) is not None else None,
                                            resource=self.name,
                                            current_order=self.current_working_order,
                                            current_coord=self.current_coord,
                                            safety_coord=self.safety_coord,
                                            target_coord=self.target_coord,
                                            item=self.job.name if self.job is not None else None,
                                            destination=self.to_location)

                    yield self.env.timeout(travel_time)

                    self.opposite.update_location(self.env.now)
                    self.update_location(self.env.now)

                    if self.monitor.use_recording:
                        self.monitor.record(self.env.now,
                                            event="Move_to",
                                            location=self.location_mapping[self.current_coord].name
                                                if self.location_mapping.get(self.current_coord) is not None else None,
                                            resource=self.name,
                                            current_order=self.current_working_order,
                                            current_coord=self.current_coord,
                                            safety_coord=self.safety_coord,
                                            target_coord=self.target_coord,
                                            item=self.job.name if self.job is not None else None,
                                            destination=self.to_location)

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
                        if self.monitor.use_recording:
                            self.monitor.record(self.env.now,
                                                event="Avoiding_wait_start",
                                                location=self.location_mapping[self.current_coord].name
                                                    if self.location_mapping.get(self.current_coord) is not None else None,
                                                resource=self.name,
                                                current_order=self.current_working_order,
                                                current_coord=self.current_coord,
                                                target_coord=self.target_coord,
                                                item=self.job.name if self.job is not None else None,
                                                destination=self.to_location)

                        yield self.env.timeout(travel_time_opposite - travel_time)

                        self.opposite.update_location(self.env.now, on_location=True)
                        self.update_location(self.env.now)

                        self.waiting = False
                        avoiding_finish = self.env.now
                        if self.monitor.use_recording:
                            self.monitor.record(self.env.now,
                                                event="Avoiding_wait_finish",
                                                location=self.location_mapping[self.current_coord].name
                                                    if self.location_mapping.get(self.current_coord) is not None else None,
                                                resource=self.name,
                                                current_order=self.current_working_order,
                                                current_coord=self.current_coord,
                                                target_coord=self.target_coord,
                                                item=self.job.name if self.job is not None else None,
                                                destination=self.to_location)

                        self.avoiding_time += avoiding_finish - avoiding_start
                else:
                    if self.monitor.use_recording:
                        self.monitor.record(self.env.now,
                                            event="Move_from",
                                            location=self.location_mapping[self.current_coord].name
                                                if self.location_mapping.get(self.current_coord) is not None else None,
                                            resource=self.name,
                                            current_order=self.current_working_order,
                                            current_coord=self.current_coord,
                                            target_coord=self.target_coord,
                                            item=self.job.name if self.job is not None else None,
                                            destination=self.to_location)

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

                    if self.monitor.use_recording:
                        self.monitor.record(self.env.now,
                                            event="Move_to",
                                            location=self.location_mapping[self.current_coord].name
                                                if self.location_mapping.get(self.current_coord) is not None else None,
                                            resource=self.name,
                                            current_order=self.current_working_order,
                                            current_coord=self.current_coord,
                                            target_coord=self.target_coord,
                                            item=self.job.name if self.job is not None else None,
                                            destination=self.to_location)

                    if added_travel_time > 0.0:
                        self.avoiding_time += added_travel_time

                    self.priority_queue.remove(self.name)
                    self.opposite.priority_queue.remove(self.name)

                    if self.status == "loading":
                        self.empty_travel_time += (travel_time - added_travel_time)
                        self.status = "unloading"
                    elif self.status == "unloading":
                        self.current_working_order = None
                        self.status = "waiting"

                    break

            except simpy.Interrupt as i:
                if self.waiting:
                    self.waiting = False
                    if self.monitor.use_recording:
                        self.monitor.record(self.env.now,
                                            event="Avoiding_wait_finish",
                                            location=self.location_mapping[self.current_coord].name
                                            if self.location_mapping.get(self.current_coord) is not None else None,
                                            resource=self.name,
                                            current_order=self.current_working_order,
                                            current_coord=self.current_coord,
                                            target_coord=self.target_coord,
                                            item=self.job.name if self.job is not None else None,
                                            destination=self.to_location)

                self.opposite.update_location(self.env.now)
                self.update_location(self.env.now)

                if self.monitor.use_recording:
                    self.monitor.record(self.env.now,
                                        event="Recheck_Required",
                                        location=self.location_mapping[self.current_coord].name
                                            if self.location_mapping.get(self.current_coord) is not None else None,
                                        resource=self.name,
                                        current_order=self.current_working_order,
                                        current_coord=self.current_coord,
                                        target_coord=self.target_coord,
                                        item=self.job.name if self.job is not None else None,
                                        destination=self.to_location)


    def _check_interference(self):
        blocking_flag, min_xcoord, max_xcoord = self._check_blocking()
        if blocking_flag:
            avoidance = True
            if (self.id == 0 and self.target_coord[0] > min_xcoord - self.safety_margin) \
                or (self.id == 1 and self.target_coord[0] < max_xcoord + self.safety_margin):
                if self.id == 0:
                    safety_xcoord = min_xcoord - self.safety_margin
                else:
                    safety_xcoord = max_xcoord + self.safety_margin
            else:
                if self.id == 0:
                    safety_xcoord = self.target_coord[0] - self.safety_margin
                else:
                    safety_xcoord = self.target_coord[0] + self.safety_margin
        else:
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

    def _check_blocking(self):
        blocking_flag = False

        min_xcoord = float('inf')
        max_xcoord = float('-inf')

        travel_time_opposite = 0.0

        if self.opposite.current_working_order is not None:
            working_order_list = [self.opposite.current_working_order] + self.opposite.queue
        else:
            working_order_list = self.opposite.queue

        current_coord = self.opposite.current_coord
        for i, working_order in enumerate(working_order_list):
            job_id, current_location, next_location = working_order
            if i == 0 and self.opposite.status == "unloading":
                dx_opposite = self.locations[next_location].coord[0] - current_coord[0]
                dy_opposite = self.locations[next_location].coord[1] - current_coord[1]
                travel_time_temp = max(abs(dx_opposite) / self.opposite.x_velocity,
                                       abs(dy_opposite) / self.opposite.y_velocity)

                travel_time_opposite += travel_time_temp

                current_coord = self.locations[next_location].coord

                if self.locations[next_location].coord[0] < min_xcoord:
                    min_xcoord = self.locations[next_location].coord[0]
                if self.locations[next_location].coord[0] > max_xcoord:
                    max_xcoord = self.locations[next_location].coord[0]
            else:
                location = self.locations[current_location]

                dx_opposite = self.locations[current_location].coord[0] - current_coord[0]
                dy_opposite = self.locations[current_location].coord[1] - current_coord[1]
                travel_time_temp = max(abs(dx_opposite) / self.opposite.x_velocity,
                                       abs(dy_opposite) / self.opposite.y_velocity)

                travel_time_opposite += travel_time_temp

                current_coord = self.locations[current_location].coord

                if (location.category == 1 or location.category == 2) and location.fully_occupied:
                    if self.to_location == current_location:
                        blocking_flag = True
                        break

                dx_opposite = self.locations[next_location].coord[0] - current_coord[0]
                dy_opposite = self.locations[next_location].coord[1] - current_coord[1]
                travel_time_temp = max(abs(dx_opposite) / self.opposite.x_velocity,
                                       abs(dy_opposite) / self.opposite.y_velocity)

                travel_time_opposite += travel_time_temp

                current_coord = self.locations[next_location].coord

                if self.locations[current_location].coord[0] < min_xcoord:
                    min_xcoord = self.locations[next_location].coord[0]
                if self.locations[current_location].coord[0] > max_xcoord:
                    max_xcoord = self.locations[next_location].coord[0]

                if self.locations[next_location].coord[0] < min_xcoord:
                    min_xcoord = self.locations[next_location].coord[0]
                if self.locations[next_location].coord[0] > max_xcoord:
                    max_xcoord = self.locations[next_location].coord[0]

        if self.to_location is not None:
            dx = self.locations[self.to_location].coord[0] - self.current_coord[0]
            dy = self.locations[self.to_location].coord[1] - self.current_coord[1]
            direction = np.sign(dx)
            travel_time = max(abs(dx) / self.x_velocity,
                              abs(dy) / self.y_velocity)

            min_travel_time = min(travel_time, travel_time_opposite)
            xcoord = self.current_coord[0] + min_travel_time * self.x_velocity * direction
            xcoord_target = self.locations[self.to_location].coord[0]

            if (self.id == 0 and xcoord <= xcoord_target - self.safety_margin) \
                    or (self.id == 1 and xcoord >= xcoord_target + self.safety_margin):
                blocking_flag = False
                min_xcoord = float('inf')
                max_xcoord = float('-inf')

        self.blocked = blocking_flag

        return blocking_flag, min_xcoord, max_xcoord

    def _check_priority(self):
        priority_flag = False

        if self.opposite.idle or self.opposite.blocked:
            priority_flag = True
        else:
            idx = self.priority_queue.index(self.name)
            if idx == 0:
                priority_flag = True

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
        self.in_transportation = False
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
        operation = job.get_current_operation()

        del self.monitor.operations_loading[operation.id]
        self.monitor.operations_unloading[operation.id] = operation

        del self.processes[job.id]
        del self.jobs_after_process[job_id]

        if self.call_for_transporting.get(job_id) is not None:
            self.call_for_transporting[job_id].succeed(None)

        return job

    def reserve_job(self, job_id):
        self.jobs_after_process[job_id].in_transportation = True

    def _arrive(self, job):
        self.monitor.jobs_in_system[job.id] = job
        operation = job.get_current_operation()

        if self.monitor.use_recording:
            self.monitor.record(self.env.now,
                                location=self.name,
                                job=job.name,
                                event="Job_Arrived")

        self.monitor.add_to_queue(job, scheduling_mode="machine")
        self.monitor.set_scheduling_flag(scheduling_mode="machine")
        self.call_for_machine_scheduling[job.id] = self.env.event()
        location_name = yield self.call_for_machine_scheduling[job.id]
        del self.call_for_machine_scheduling[job.id]

        del self.monitor.operations_unscheduled[operation.id]
        self.monitor.operations_loading[operation.id] = operation

        while location_name is not None:
            job.next_location = location_name

            if len(self.locations[location_name].processes) + 1 >= self.locations[location_name].capacity:
                self.locations[location_name].fully_occupied = True

            self.monitor.add_to_queue(job, scheduling_mode="crane")
            self.monitor.set_scheduling_flag(scheduling_mode="crane")
            self.call_for_crane_scheduling[job.id] = self.env.event()
            crane_name = yield self.call_for_crane_scheduling[job.id]
            del self.call_for_crane_scheduling[job.id]

            crane = self.resources[crane_name]
            crane.add_to_queue((job.id, job.current_location, job.next_location))

            if self.monitor.use_recording:
                self.monitor.record(self.env.now,
                                    location=self.name,
                                    job=job.name,
                                    event="Crane_Called",
                                    resource=crane_name,
                                    current_order=crane.current_working_order,
                                    idle=crane.idle,
                                    destination=job.next_location,
                                    queue=crane.queue[:])

            if crane.idle and (not crane.waiting_event.triggered):
                crane.waiting_event.succeed()
                location_name = None
            else:
                self.call_for_transporting[job.id] = self.env.event()
                location_name = yield self.call_for_transporting[job.id]
                if location_name is not None:
                    self.locations[job.next_location].fully_occupied = False
                del self.call_for_transporting[job.id]


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

        operation = job.get_current_operation()
        del self.monitor.operations_unloading[operation.id]

        self.processes[job.id] = self.env.process(self._work(job))
        self.jobs_in_process[job.id] = job

    def get(self, job_id):
        job = self.jobs_after_process[job_id]
        operation = job.get_current_operation()
        if operation is None:
            operation = job.operations[-1]

        del self.monitor.operations_loading[operation.id]
        self.monitor.operations_unloading[operation.id] = operation

        del self.processes[job.id]
        del self.jobs_after_process[job_id]

        if self.call_for_transporting.get(job_id) is not None:
            self.call_for_transporting[job_id].succeed(None)

        if "Output" in job.next_location:
            if len(self.monitor.operations_waiting) > 0:
                self.monitor.machine_scheduling = True

        return job

    def reserve_job(self, job_id):
        self.jobs_after_process[job_id].in_transportation = True

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
        job.in_transportation = False
        if len(self.jobs_in_process) == self.capacity:
            self.idle = False

        operation = job.get_current_operation()
        self.monitor.operations_working[operation.id] = operation

        if self.monitor.use_recording:
            self.monitor.record(self.env.now,
                                location=self.name,
                                job=job.name,
                                operation=operation.name,
                                event="Working_Started")

        processing_time = operation.get_processing_time(self.local_id)
        operation.start_time = self.env.now
        operation.allocated_machine = self.name
        yield self.env.timeout(processing_time)

        if self.monitor.use_recording:
            self.monitor.record(self.env.now,
                                location=self.name,
                                job=job.name,
                                operation=operation.name,
                                event="Working Finished")

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

        operation = job.get_current_operation()
        if operation is not None:
            del self.monitor.operations_unscheduled[operation.id]
            self.monitor.operations_loading[operation.id] = operation
        else:
            operation = job.operations[-1]
            self.monitor.operations_loading[operation.id] = operation

        reassign = False
        while location_name is not None:
            job.next_location = location_name

            if len(self.locations[location_name].processes) + 1 >= self.locations[location_name].capacity:
                self.locations[location_name].fully_occupied = True

            self.monitor.add_to_queue(job, scheduling_mode="crane")
            self.monitor.set_scheduling_flag(scheduling_mode="crane")
            self.call_for_crane_scheduling[job.id] = self.env.event()
            crane_name = yield self.call_for_crane_scheduling[job.id]
            del self.call_for_crane_scheduling[job.id]

            if crane_name is not None:
                crane = self.resources[crane_name]
                crane.add_to_queue((job.id, job.current_location, job.next_location))

                if self.monitor.use_recording:
                    self.monitor.record(self.env.now,
                                        location=self.name,
                                        job=job.name,
                                        event="Crane_Called",
                                        resource=crane_name,
                                        current_order=crane.current_working_order,
                                        idle=crane.idle,
                                        destination=job.next_location,
                                        queue=crane.queue[:])

                if not reassign:
                    self.fully_occupied = False
                    self.monitor.set_scheduling_flag(scheduling_mode="machine")

                if crane.idle and (not crane.waiting_event.triggered):
                    crane.waiting_event.succeed()
                    location_name = None
                else:
                    self.call_for_transporting[job.id] = self.env.event()
                    location_name = yield self.call_for_transporting[job.id]
                    if location_name is not None:
                        self.locations[job.next_location].fully_occupied = False
                        reassign = True
                    del self.call_for_transporting[job.id]
            else:
                self.get(job.id)
                self.put(job)
                location_name = None


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

        operation = job.get_current_operation()
        if operation is None:
            operation = job.operations[-1]
        del self.monitor.operations_unloading[operation.id]

        self.processes[job.id] = self.env.process(self._wait(job))
        self.jobs_in_process[job.id] = job

    def get(self, job_id):
        job = self.jobs_after_process[job_id]
        operation = job.get_current_operation()
        if operation is None:
            operation = job.operations[-1]

        del self.monitor.operations_loading[operation.id]
        self.monitor.operations_unloading[operation.id] = operation

        del self.processes[job.id]
        del self.jobs_after_process[job_id]

        if self.call_for_transporting.get(job_id) is not None:
            self.call_for_transporting[job_id].succeed(None)

        return job

    def reserve_job(self, job_id):
        self.jobs_after_process[job_id].in_transportation = True

    def check_status(self):
        fully_occupied = self.fully_occupied
        return fully_occupied

    def _wait(self, job):
        job.in_transportation = False
        operation = job.get_current_operation()

        if operation is not None:
            self.monitor.operations_waiting[operation.id] = operation

        if self.monitor.use_recording:
            if operation is not None:
                self.monitor.record(self.env.now,
                                    location=self.name,
                                    job=job.name,
                                    operation=operation.name,
                                    event="Waiting Started")
            else:
                self.monitor.record(self.env.now,
                                    location=self.name,
                                    job=job.name,
                                    event="Waiting Started")

        if operation is not None:
            operation.waiting_start = self.env.now

        self.monitor.set_scheduling_flag(scheduling_mode="machine")
        self.call_for_machine_scheduling[job.id] = self.env.event()
        location_name = yield self.call_for_machine_scheduling[job.id]
        del self.call_for_machine_scheduling[job.id]

        if operation is not None:
            del self.monitor.operations_waiting[operation.id]

        if self.monitor.use_recording:
            if operation is not None:
                self.monitor.record(self.env.now,
                                    location=self.name,
                                    job=job.name,
                                    operation=operation.name,
                                    event="Waiting Finished")
            else:
                self.monitor.record(self.env.now,
                                    location=self.name,
                                    job=job.name,
                                    event="Waiting Finished")

        del self.jobs_in_process[job.id]
        self.jobs_after_process[job.id] = job

        if operation is not None:
            self.monitor.operations_loading[operation.id] = operation
        else:
            operation = job.operations[-1]
            self.monitor.operations_loading[operation.id] = operation

        while location_name is not None:
            job.next_location = location_name

            if len(self.locations[location_name].processes) + 1 >= self.locations[location_name].capacity:
                self.locations[location_name].fully_occupied = True

            self.monitor.add_to_queue(job, scheduling_mode="crane")
            self.monitor.set_scheduling_flag(scheduling_mode="crane")
            self.call_for_crane_scheduling[job.id] = self.env.event()
            crane_name = yield self.call_for_crane_scheduling[job.id]
            del self.call_for_crane_scheduling[job.id]

            self.fully_occupied = False

            crane = self.resources[crane_name]
            crane.add_to_queue((job.id, job.current_location, job.next_location))

            if self.monitor.use_recording:
                self.monitor.record(self.env.now,
                                    location=self.name,
                                    job=job.name,
                                    event="Crane_Called",
                                    resource=crane_name,
                                    current_order=crane.current_working_order,
                                    idle=crane.idle,
                                    destination=job.next_location,
                                    queue=crane.queue[:])

            if crane.idle and (not crane.waiting_event.triggered):
                crane.waiting_event.succeed()
                location_name = None
            else:
                self.call_for_transporting[job.id] = self.env.event()
                location_name = yield self.call_for_transporting[job.id]
                if location_name is not None:
                    self.locations[job.next_location].fully_occupied = False
                del self.call_for_transporting[job.id]


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

    def put(self, job):
        job.current_location = self.name
        job.next_location = None

        operation = job.get_current_operation()
        if operation is None:
            operation = job.operations[-1]
        del self.monitor.operations_unloading[operation.id]

        self.processes[job.id] = self.env.process(self._departure(job))

    def check_status(self):
        fully_occupied = self.fully_occupied
        return fully_occupied

    def _departure(self, job):
        if self.monitor.use_recording:
            self.monitor.record(self.env.now,
                                location=self.name,
                                job=job.name,
                                event="Job_Completed")

        yield self.env.timeout(0)

        del self.processes[job.id]
        self.sink.put(job)
        self.fully_occupied = False


class Sink:
    def __init__(self, env, name, monitor):
        self.env = env
        self.name = name
        self.monitor = monitor

        self.num_jobs_degenerated = 0
        self.completion_time = 0.0

    def put(self, job):
        self.monitor.jobs_after_system[job.id] = job
        del self.monitor.jobs_in_system[job.id]

        job.current_location = self.name
        job.next_location = None

        self.num_jobs_degenerated += 1
        self.completion_time = self.env.now

        if self.monitor.use_recording:
            self.monitor.record(self.env.now,
                                location=self.name,
                                job=job.name,
                                event="Job_Degenerated")


class Monitor:
    def __init__(self, use_recording=True):
        self.use_recording = use_recording

        self.queue_for_machine_scheduling = {}
        self.queue_for_crane_scheduling = None
        self.machine_scheduling = False
        self.crane_scheduling = False

        self.operations_unscheduled = {}
        self.operations_loading = {}
        self.operations_unloading = {}
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
        self.next_location = []
        self.event = []
        self.resource = []
        self.current_order = []
        self.current_coord = []
        self.safety_coord = []
        self.target_coord = []
        self.idle = []
        self.blocked = []
        self.avoidance = []
        self.item = []
        self.destination = []
        self.queue = []

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

    def record(self, time, location=None, job=None, operation=None, next_location=None, event=None,
               resource=None, current_order=None, current_coord=None, safety_coord=None, target_coord=None,
               idle=None, blocked=None, avoidance=None, item=None, destination=None, queue=None):
        self.time.append(time)
        self.location.append(location)
        self.job.append(job)
        self.operation.append(operation)
        self.next_location.append(next_location)
        self.event.append(event)
        self.resource.append(resource)
        self.current_order.append(current_order)
        self.current_coord.append(current_coord)
        self.safety_coord.append(safety_coord)
        self.target_coord.append(target_coord)
        self.idle.append(idle)
        self.blocked.append(blocked)
        self.avoidance.append(avoidance)
        self.item.append(item)
        self.destination.append(destination)
        self.queue.append(queue)

    def get_logs(self, file_path=None):
        df_log = pd.DataFrame(columns=['Time', 'Location', 'Job', 'Operation', 'Next_Location', 'Event',
                                       'Resource', 'Current_Order', 'Current_Coord', 'Safety_Coord', 'Target_Coord',
                                       'Idle', 'Blocked', 'Avoidance', 'Item', 'Destination', 'Queue'])
        df_log['Time'] = self.time
        df_log['Location'] = self.location
        df_log['Job'] = self.job
        df_log['Operation'] = self.operation
        df_log['Next_Location'] = self.next_location
        df_log['Event'] = self.event
        df_log['Resource'] = self.resource
        df_log['Current_Coord'] = self.current_coord
        df_log['Safety_Coord'] = self.safety_coord
        df_log['Target_Coord'] = self.target_coord
        df_log['Idle'] = self.idle
        df_log['Blocked'] = self.blocked
        df_log['Avoidance'] = self.avoidance
        df_log['Item'] = self.item
        df_log['Destination'] = self.destination
        df_log['Current_Order'] = self.current_order
        df_log['Queue'] = self.queue

        if file_path is not None:
            df_log.to_excel(file_path, sheet_name="logs", index=False)

        return df_log