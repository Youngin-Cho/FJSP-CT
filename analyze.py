import os
import pandas as pd


def summarize(res_dir):
    makespan = {}
    computing_time = {}

    for filename in os.listdir(res_dir):
        algorithm = filename.split("(")[-1].split(")")[0]
        df_makespan = pd.read_excel(res_dir + filename, sheet_name="makespan", engine='openpyxl')
        df_computing_time = pd.read_excel(res_dir + filename, sheet_name="computing_time", engine='openpyxl')

        makespan[algorithm] = df_makespan.loc["avg", "avg"]
        computing_time[algorithm] = df_computing_time.loc["avg", "avg"]

    return makespan, computing_time


if __name__ == "__main__":
    res_dirs = ["./output/test/10-5/"]

    df_makespan_lst = []
    df_computing_time_lst = []

    for res_dir in res_dirs:
        makespan, computing_time = summarize(res_dir)
        df_makespan_temp = pd.DataFrame.from_dict(makespan)
        df_computing_time_temp = pd.DataFrame.from_dict(computing_time)

        df_makespan_lst.append(df_makespan_temp)
        df_computing_time_lst.append(df_computing_time_temp)

    df_makespan = pd.concat(df_makespan_lst)
    df_computing_time = pd.concat(df_computing_time_lst)

    file_path = "./output/test/summary.xlsx"
    writer = pd.ExcelWriter(file_path)
    df_makespan.to_excel(writer, sheet_name="makespan")
    df_computing_time.to_excel(writer, sheet_name="computing_time")
    writer.close()