import os
import json
import pandas as pd


def calculate_lower_bound():
    file_dirs = ["../input/case1/test/10-5/",
                 "../input/case1/test/15-5/",
                 "../input/case1/test/15-10/",
                 "../input/case1/test/20-10/",
                 "../input/case1/test/20-15/",
                 "../input/case1/test/25-15/"]

    index = [int(os.path.splitext(filename)[0].split("-")[1])
             for filename in os.listdir(file_dirs[0])
             if os.path.splitext(filename)[1] == '.xlsx']
    columns = [dir.split('/')[-2] for dir in file_dirs]
    df_lower_bound = pd.DataFrame(index=index, columns=columns)

    for dir in file_dirs:
        file_paths = os.listdir(dir)

        with open(dir + "setting.json", 'r') as f:
            setting = json.load(f)

        lower_bounds = []
        for path in file_paths:
            if path.split(".")[-1] != "xlsx":
                continue

            df_operations = pd.read_excel(dir + path, sheet_name="operations", engine='openpyxl')

            expected_completion_times = []
            df_jobs = df_operations.groupby("Job_Index")
            for _, df_temp in df_jobs:
                df_temp["Processing_Time_Min"] = df_temp.iloc[:, 6:].apply(lambda x: x[x > 0].min(), axis=1)

                expected_completion_time = (df_temp["Arrival_Date"].iloc[0] + df_temp["Processing_Time_Min"].sum()
                                            + (len(setting["division"]) - 1) * setting["x_spacing"] / setting["x_velocity"])

                expected_completion_times.append(expected_completion_time)

            lower_bound = max(expected_completion_times)
            lower_bounds.append(lower_bound)

        df_lower_bound[dir.split('/')[-2]] = lower_bounds

        writer = pd.ExcelWriter("../output/test/lower bound.xlsx")
        df_lower_bound.to_excel(writer, sheet_name="lower_bound")
        writer.close()


def transform():
    summary_file = "../output/test/results.xlsx"
    lower_bound_file = "../output/test/lower bound.xlsx"

    df_lower_bound = pd.read_excel(lower_bound_file, sheet_name="lower_bound", engine='openpyxl', index_col=0)
    df_summary = {col: pd.read_excel(summary_file, sheet_name=col, engine='openpyxl', index_col=0).iloc[:-1]
                  for col in df_lower_bound.columns}
    df_summary_transformed = {}

    for col in df_lower_bound.columns:
        df_summary_transformed[col] = (df_summary[col].subtract(df_lower_bound[col], axis=0)).div(df_lower_bound[col], axis=0) * 100

    writer = pd.ExcelWriter("../output/test/results_transformed.xlsx")
    for problem_size in df_summary_transformed.keys():
        df_summary_transformed[problem_size].to_excel(writer, sheet_name=problem_size)
    writer.close()


def summary(scaling=False):
    test_case = ["10-5", "15-5", "15-10", "20-10", "20-15", "25-15"]

    file_mapping = {"SPT+SETT": "Heuristics/(SPT+SETT) test results.xlsx",
                    "SPT+TDD": "Heuristics/(SPT+TDD) test results.xlsx",
                    "SPT+TDT": "Heuristics/(SPT+TDT) test results.xlsx",
                    "MOR+SETT": "Heuristics/(MOR+SETT) test results.xlsx",
                    "MOR+TDD": "Heuristics/(MOR+TDD) test results.xlsx",
                    "MOR+TDT": "Heuristics/(MOR+TDT) test results.xlsx",
                    "MWKR+SETT": "Heuristics/(MWKR+SETT) test results.xlsx",
                    "MWKR+TDD": "Heuristics/(MWKR+TDD) test results.xlsx",
                    "MWKR+TDT": "Heuristics/(MWKR+TDT) test results.xlsx",
                    "RL+SETT": "SARL/FJSP/RL-SETT/(RL+SETT) test results.xlsx",
                    "RL+TDD": "SARL/FJSP/RL-TDD/(RL+TDD) test results.xlsx",
                    "RL+TDT": "SARL/FJSP/RL-TDT/(RL+TDT) test results.xlsx",
                    "SPT+RL": "SARL/CT/SPT-RL/(SPT+RL) test results.xlsx",
                    "MOR+RL": "SARL/CT/MOR-RL/(MOR+RL) test results.xlsx",
                    "MWKR+RL": "SARL/CT/MWKR-RL/(MWKR+RL) test results.xlsx",
                    "SPT+RL (comm.)": "SARL/CT/SPT-RL (+comm)/(SPT+RL) test results.xlsx",
                    "MOR+RL (comm.)": "SARL/CT/MOR-RL (+comm)/(MOR+RL) test results.xlsx",
                    "MWKR+RL (comm.)": "SARL/CT/MWKR-RL (+comm)/(MWKR+RL) test results.xlsx",
                    "RL+RL": "MARL/SARL/RL-RL/(RL+RL) test results.xlsx",
                    "RL+RL (comm.)": "MARL/SARL/RL-RL (+comm)/(RL+RL) test results.xlsx",
                    "CTCE": "MARL/CTCE/RL-RL/(RL+RL) test results.xlsx",
                    "DTDE": "MARL/IL/RL-RL/(RL+RL) test results.xlsx",
                    "DTDE with pt": "MARL/IL/RL-RL with pt/(RL+RL) test results.xlsx",
                    "DTDE (comm.)": "MARL/IL/RL-RL (+comm)/(RL+RL) test results.xlsx",
                    "DTDE with pt (comm.)": "MARL/IL/RL-RL with pt (+comm)/(RL+RL) test results.xlsx",
                    "CTDE with CL": "MARL/CTDE/RL-RL with CL/(RL+RL) test results.xlsx",
                    "CTDE with EP": "MARL/CTDE/RL-RL with EP/(RL+RL) test results.xlsx",
                    "CTDE with CL (comm.)": "MARL/CTDE/RL-RL with CL (+comm)/(RL+RL) test results.xlsx",
                    "CTDE with EP (comm.)": "MARL/CTDE/RL-RL with EP (+comm)/(RL+RL) test results.xlsx"}

    results = {}
    for problem_size in test_case:
        if scaling:
            df_baseline = pd.read_excel("../output/test/" + problem_size + "/Heuristics/(RAND+RAND) test results.xlsx",
                                        sheet_name="makespan", engine='openpyxl', index_col=0)

        df_total = pd.DataFrame()
        for col, file_name in file_mapping.items():
            file_path = "../output/test/" + problem_size + "/" + file_name
            df_temp = pd.read_excel(file_path, sheet_name="makespan", engine='openpyxl', index_col=0)
            if scaling:
                df_total[col] = df_baseline["avg"].subtract(df_temp["avg"], axis=0).div(df_baseline["avg"], axis=0) * 100
            else:
                df_total[col] = df_temp["avg"]

        results[problem_size] = df_total

    if scaling:
        writer = pd.ExcelWriter("../output/test/results_scaled_by_rand+rand.xlsx")
    else:
        writer = pd.ExcelWriter("../output/test/results.xlsx")

    for problem_size in test_case:
        results[problem_size].to_excel(writer, sheet_name=problem_size)

    writer.close()


def summary_computing_time():
    test_case = ["10-5", "15-5", "15-10", "20-10", "20-15", "25-15"]

    file_mapping = {"SPT+SETT": "Heuristics/(SPT+SETT) test results.xlsx",
                    "SPT+TDD": "Heuristics/(SPT+TDD) test results.xlsx",
                    "SPT+TDT": "Heuristics/(SPT+TDT) test results.xlsx",
                    "MOR+SETT": "Heuristics/(MOR+SETT) test results.xlsx",
                    "MOR+TDD": "Heuristics/(MOR+TDD) test results.xlsx",
                    "MOR+TDT": "Heuristics/(MOR+TDT) test results.xlsx",
                    "MWKR+SETT": "Heuristics/(MWKR+SETT) test results.xlsx",
                    "MWKR+TDD": "Heuristics/(MWKR+TDD) test results.xlsx",
                    "MWKR+TDT": "Heuristics/(MWKR+TDT) test results.xlsx",
                    "RL+SETT": "SARL/FJSP/RL-SETT/(RL+SETT) test results.xlsx",
                    "RL+TDD": "SARL/FJSP/RL-TDD/(RL+TDD) test results.xlsx",
                    "RL+TDT": "SARL/FJSP/RL-TDT/(RL+TDT) test results.xlsx",
                    "SPT+RL": "SARL/CT/SPT-RL/(SPT+RL) test results.xlsx",
                    "MOR+RL": "SARL/CT/MOR-RL/(MOR+RL) test results.xlsx",
                    "MWKR+RL": "SARL/CT/MWKR-RL/(MWKR+RL) test results.xlsx",
                    "SPT+RL (comm.)": "SARL/CT/SPT-RL (+comm)/(SPT+RL) test results.xlsx",
                    "MOR+RL (comm.)": "SARL/CT/MOR-RL (+comm)/(MOR+RL) test results.xlsx",
                    "MWKR+RL (comm.)": "SARL/CT/MWKR-RL (+comm)/(MWKR+RL) test results.xlsx",
                    "RL+RL": "MARL/SARL/RL-RL/(RL+RL) test results.xlsx",
                    "RL+RL (comm.)": "MARL/SARL/RL-RL (+comm)/(RL+RL) test results.xlsx",
                    "CTCE": "MARL/CTCE/RL-RL/(RL+RL) test results.xlsx",
                    "DTDE": "MARL/IL/RL-RL/(RL+RL) test results.xlsx",
                    "DTDE with pt": "MARL/IL/RL-RL with pt/(RL+RL) test results.xlsx",
                    "DTDE (comm.)": "MARL/IL/RL-RL (+comm)/(RL+RL) test results.xlsx",
                    "DTDE with pt (comm.)": "MARL/IL/RL-RL with pt (+comm)/(RL+RL) test results.xlsx",
                    "CTDE with CL": "MARL/CTDE/RL-RL with CL/(RL+RL) test results.xlsx",
                    "CTDE with EP": "MARL/CTDE/RL-RL with EP/(RL+RL) test results.xlsx",
                    "CTDE with CL (comm.)": "MARL/CTDE/RL-RL with CL (+comm)/(RL+RL) test results.xlsx",
                    "CTDE with EP (comm.)": "MARL/CTDE/RL-RL with EP (+comm)/(RL+RL) test results.xlsx"}

    results = {}
    for problem_size in test_case:
        df_total = pd.DataFrame()

        data_dir = "../input/case1/test/" + problem_size + "/"
        data_paths = os.listdir(data_dir)

        df_num_operations = pd.DataFrame(columns=["num_operations"])
        for path in data_paths:
            if path.split(".")[-1] != "xlsx":
                continue

            df_data = pd.read_excel(data_dir + path, sheet_name="operations", engine='openpyxl')
            num_operations = len(df_data)

            instance_name = path.split(".")[-2]
            df_num_operations.loc[int(instance_name.split("-")[1])] = num_operations

        df_total["num_operations"] = df_num_operations["num_operations"]

        for col, file_name in file_mapping.items():
            file_path = "../output/test/" + problem_size + "/" + file_name
            df_temp = pd.read_excel(file_path, sheet_name="computing_time", engine='openpyxl', index_col=0).iloc[:-1]
            df_total[col] = df_temp["avg"]

        results[problem_size] = df_total

    writer = pd.ExcelWriter("../output/test/results_computing_time.xlsx")

    for problem_size in test_case:
        results[problem_size].to_excel(writer, sheet_name=problem_size)

    writer.close()


if __name__ == "__main__":
    # calculate_lower_bound()
    summary_computing_time()