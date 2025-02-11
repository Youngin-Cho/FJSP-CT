import numpy as np
import pandas as pd


class DataGenerator:
    def __init__(self,
                 num_machines=10,
                 num_jobs=20,
                 num_options_min=1,
                 num_options_max=10,
                 num_operations_min=4,
                 num_operations_max=6,
                 proctime_min=1,
                 proctime_max=20,
                 iat=0.5):

        self.num_machines = num_machines
        self.num_jobs = num_jobs
        self.num_options_min = num_options_min
        self.num_options_max = num_options_max
        self.num_operations_min = num_operations_min
        self.num_operations_max = num_operations_max
        self.proctime_min = proctime_min
        self.proctime_max = proctime_max
        self.iat = iat

    def generate(self, file_path=None):
        columns = ["Job_Name", "Job_Index", "Arrival_Date", "Operation_Name", "Operation_Index", "Order"]
        columns = columns + ["Machine %d" % i for i in range(self.n_machines)]

        temp = []
        offset = 0
        for i in range(self.n_jobs):
            job_name = "J-%d" % i
            job_index = i

            if i == 0:
                arrival_date = 0
            else:
                iat = int(np.random.geometric(1 / self.iat))
                arrival_date += iat

            num_operations = np.random.randint(self.n_operations_min, self.n_operations_max + 1)

            for j in range(num_operations):
                operation_name = "O-%d%d" % (i, j)
                operation_index = offset + j
                order = j

                proctime = np.zeros(self.n_machines)
                num_options = np.random.randint(self.n_options_min, self.n_options_max + 1)
                options = np.random.choice(range(self.n_machines), num_options, replace=False)
                proctime_avg = np.random.randint(self.proctime_min, self.proctime_max + 1)
                proctime_sampled = [np.random.randint(np.ceil(0.8 * proctime_avg),
                                                      np.floor(1.2 * proctime_avg) + 1)
                                    for _ in range(num_options)]
                proctime[options] = proctime_sampled

                row = ([job_name, job_index, arrival_date, operation_name, operation_index, order] + list(proctime))
                temp.append(row)

            offset += num_operations

        df_scenario = pd.DataFrame(temp, columns=columns)

        if file_path is not None:
            writer = pd.ExcelWriter(file_path)
            df_scenario.to_excel(writer, sheet_name="scenario", index=False)
            writer.close()

        return df_scenario


if __name__ == '__main__':
    import os

    num_machines = 10,
    num_jobs = 20,
    num_options_min = 1,
    num_options_max = 10,
    num_operations_min = 4,
    num_operations_max = 6,
    proctime_min = 1,
    proctime_max = 20,
    iat = 0.5

    file_dir = "../input/validation/%d-%d/" % (num_jobs, num_machines)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    data_generator = DataGenerator(config)
    n_instance = 20
    for i in range(1, n_instance + 1):
        file_path = file_dir + "instance-{0}.xlsx".format(i)
        data_generator.generate(file_path=file_path)
        # flag = True
        # while flag:
        #     file_path = file_dir + "instance-{0}.xlsx".format(i)
        #     df_scenario, df_initial = data_generator.generate(file_path=file_path)
        #     max_wip = WIP_graph(df_scenario)
        #     if config.n_machines * 0.8 <= max_wip <= config.n_machines * 1.2:
        #         flag = False