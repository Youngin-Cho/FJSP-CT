import numpy as np
import pandas as pd


class DataGenerator:
    def __init__(self,
                 # Flexible Job Shop
                 num_inputs=1,
                 num_outputs=1,
                 num_machines=5,
                 num_buffers=2,
                 num_jobs=10,
                 num_options_min=1,
                 num_options_max=5,
                 num_operations_min=4,
                 num_operations_max=6,
                 proctime_min=1,
                 proctime_max=20,
                 inter_arrival_time=4,
                 # Crane Transportation
                 num_cranes=2,
                 safety_margin=2,
                 x_velocity=0.5,
                 y_velocity=1.0,
                 #Factory Layout
                 num_rows=1,
                 x_spacing=2,
                 y_spacing=1,
                 division=(0, 1, 1, 2, 1, 2, 1, 1, 3)
                 ):

        assert len([temp for temp in division if temp == 0]) == num_inputs
        assert len([temp for temp in division if temp == 1]) == num_machines
        assert len([temp for temp in division if temp == 2]) == num_buffers
        assert len([temp for temp in division if temp == 3]) == num_outputs

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_machines = num_machines
        self.num_buffers = num_buffers
        self.num_jobs = num_jobs
        self.num_options_min = num_options_min
        self.num_options_max = num_options_max
        self.num_operations_min = num_operations_min
        self.num_operations_max = num_operations_max
        self.proctime_min = proctime_min
        self.proctime_max = proctime_max
        self.inter_arrival_time = inter_arrival_time

        self.num_cranes = num_cranes
        self.safety_margin = safety_margin
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity

        self.num_rows = num_rows
        self.num_bays = len(division)
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing
        self.division = division

    def generate(self, file_path=None):
        columns_operations = (["Job_Name", "Job_Index", "Arrival_Date", "Operation_Name", "Operation_Index", "Order"]
                              + ["Machine-%d" % i for i in range(self.num_machines)])
        columns_locations = ["Location_Name", "Location_Index", "Location_Type", "X_Coordinate", "Y_Coordinate"]
        columns_resources = ["Crane_Name", "Crane_Index", "X_Velocity", "Y_Velocity",
                             "Initial_X_Coordinate", "Initial_Y_Coordinate"]

        df_operations = []
        df_locations = []
        df_resources = []

        for j in range(self.num_jobs):
            job_name = "J-%d" % j
            job_index = j

            if j == 0:
                arrival_date = 0
            else:
                inter_arrival_time = int(np.random.geometric(1 / self.inter_arrival_time))
                arrival_date += inter_arrival_time

            num_operations = np.random.randint(self.num_operations_min, self.num_operations_max + 1)

            for k in range(num_operations):
                operation_name = "O-%d_%d" % (j, k)
                operation_index = len(df_operations)
                order = k

                proctime = np.zeros(self.num_machines)
                num_options = np.random.randint(self.num_options_min, self.num_options_max + 1)
                options = np.random.choice(range(self.num_machines), num_options, replace=False)
                proctime_avg = np.random.randint(self.proctime_min, self.proctime_max + 1)
                proctime_sampled = [np.random.randint(np.ceil(0.8 * proctime_avg),
                                                      np.floor(1.2 * proctime_avg) + 1)
                                    for _ in range(num_options)]
                proctime[options] = proctime_sampled

                row = ([job_name, job_index, arrival_date, operation_name, operation_index, order] + list(proctime))
                df_operations.append(row)

        num_locations = int(self.num_rows * self.num_bays)
        count = {0: 0, 1: 0, 2: 0, 3: 0}
        for i in range(num_locations):
            location_index = i

            row_index = i % self.num_rows
            bay_index = i // self.num_rows

            location_type = self.division[bay_index]
            if location_type == 0:
                location_name = "Input-%d" % count[location_type]
            elif location_type == 1:
                location_name = "Machine-%d" % count[location_type]
            elif location_type == 2:
                location_name = "Buffer-%d" % count[location_type]
            else:
                location_name = "Output-%d" % count[location_type]

            count[location_type] += 1

            x_coordinate = self.x_spacing * bay_index
            y_coordinate = self.y_spacing * row_index

            df_locations.append([location_name, location_index, location_type, x_coordinate, y_coordinate])

        for i in range(self.num_cranes):
            crane_name = "Crane-%d" % i
            crane_index = i

            x_velocity = self.x_velocity
            y_velocity = self.y_velocity

            if i == 0:
                initial_x_coordinate = 0
            else:
                initial_x_coordinate = self.num_bays - 1

            initial_y_coordinate = 0

            df_resources.append([crane_name, crane_index, x_velocity, y_velocity,
                                 initial_x_coordinate, initial_y_coordinate])

        df_operations = pd.DataFrame(df_operations, columns=columns_operations)
        df_locations = pd.DataFrame(df_locations, columns=columns_locations)
        df_resources = pd.DataFrame(df_resources, columns=columns_resources)

        if file_path is not None:
            writer = pd.ExcelWriter(file_path)
            df_operations.to_excel(writer, sheet_name="operations", index=False)
            df_locations.to_excel(writer, sheet_name="locations", index=False)
            df_resources.to_excel(writer, sheet_name="resources", index=False)
            writer.close()

        return df_operations, df_locations, df_resources


if __name__ == '__main__':
    import os

    num_inputs = 1
    num_outputs = 1
    num_machines = 5
    num_buffers = 2
    num_jobs = 10
    num_options_min = 1
    num_options_max = 5
    num_operations_min = 4
    num_operations_max = 6
    proctime_min = 1
    proctime_max = 20
    inter_arrival_time = 4

    num_cranes = 2
    safety_margin = 2
    x_velocity = 0.5
    y_velocity = 1.0

    num_rows = 1
    x_spacing = 2
    y_spacing = 1
    division = (0, 1, 1, 2, 1, 2, 1, 1, 3)

    file_dir = "../input/validation/%d-%d/" % (num_jobs, num_machines)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    data_generator = DataGenerator(num_inputs=num_inputs,
                                   num_outputs=num_outputs,
                                   num_machines=num_machines,
                                   num_buffers=num_buffers,
                                   num_jobs=num_jobs,
                                   num_options_min=num_options_min,
                                   num_options_max=num_options_max,
                                   num_operations_min=num_operations_min,
                                   num_operations_max=num_operations_max,
                                   proctime_min=proctime_min,
                                   proctime_max=proctime_max,
                                   inter_arrival_time=inter_arrival_time,
                                   num_cranes=num_cranes,
                                   safety_margin=safety_margin,
                                   x_velocity=x_velocity,
                                   y_velocity=y_velocity,
                                   num_rows=num_rows,
                                   x_spacing=x_spacing,
                                   y_spacing=y_spacing,
                                   division=division)

    n_instance = 20
    for i in range(1, n_instance + 1):
        file_path = file_dir + "instance-{0}.xlsx".format(i)
        data_generator.generate(file_path=file_path)