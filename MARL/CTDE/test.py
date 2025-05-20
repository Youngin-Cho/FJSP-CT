import os
import time
import json
import random
import argparse
import torch
import numpy as np
import pandas as pd

from Environment.environment import Factory
from Agent.FlexibleJobShop.network import FJSPScheduler
from Agent.CraneTransportation.network import CTScheduler


def get_config():
    parser = argparse.ArgumentParser(description="FJSP")

    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_record', action='store_true', help="Disable Recording events")
    parser.add_argument('--no_communication', action='store_true', help="Disable communication")

    parser.add_argument("--num_iterations", type=int, default=10, help="number of iterations")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")

    parser.add_argument("--num_jobs", type=int, default=10, help="number of jobs")
    parser.add_argument("--num_machines", type=int, default=5, help="number of machines")

    parser.add_argument("--fjsp_model_path", type=str, default=None, help="model file path for the fjsp agent")
    parser.add_argument("--ct_model_path", type=str, default=None, help="model file path for the ct agent")
    parser.add_argument("--param_path", type=str, default=None, help="hyper-parameter file path")

    parser.add_argument("--data_dir", type=str, default=None, help="test data path")
    parser.add_argument("--res_dir", type=str, default=None, help="test result file path")
    parser.add_argument("--param_dir", type=str, default=None, help="hyperparameter file directory")
    parser.add_argument("--fjsp_model_dir", type=str, default=None, help="model file directory for the fjsp agent")
    parser.add_argument("--ct_model_dir", type=str, default=None, help="model file directory for the ct agent")
    parser.add_argument("--sim_dir", type=str, default=None, help="simulation log file path")

    return parser.parse_args()


def test(config):
    use_cuda = torch.cuda.is_available() and not config.no_cuda
    use_recording = False if config.no_record else True
    use_communication = False if config.use_communication else True

    if use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    random_seed = config.random_seed

    data_dir = config.data_dir
    test_paths = os.listdir(data_dir)

    makespans = [[] for _ in range(config.num_iterations)]
    computing_times = [[] for _ in range(config.num_iterations)]

    print("==========Test of CTDE approach started==========")

    for filename in test_paths:
        if filename.split(".")[-1] != "xlsx":
            continue

        instance_name = filename.split(".")[-2]

        data_src = data_dir + filename
        env = Factory(data_src,
                      algorithm=("RL", "RL"),
                      use_recording=use_recording,
                      use_communication=use_communication,
                      return_global_state=False)

        param_path = config.param_path

        with open(param_path, 'r') as f:
            parameters = json.load(f)

        fjsp_model_path = config.fjsp_model_path

        fjsp_agent = FJSPScheduler(meta_data=env.fjsp_meta_data,
                                   state_size=env.fjsp_state_size,
                                   num_nodes=env.fjsp_num_nodes,
                                   embed_dim=parameters["embed_dim"],
                                   num_heads=parameters["num_heads"],
                                   num_HGT_layers=parameters["num_HGT_layers"],
                                   num_actor_layers=parameters["num_actor_layers"],
                                   num_critic_layers=parameters["num_critic_layers"],
                                   use_local_critic=False).to(device)

        checkpoint = torch.load(fjsp_model_path, map_location=torch.device(device), weights_only=True)
        fjsp_agent.load_state_dict(checkpoint['model_state_dict'])

        ct_model_path = config.ct_model_path

        ct_agent = CTScheduler(meta_data=env.ct_meta_data,
                               state_size=env.ct_state_size,
                               num_nodes=env.ct_num_nodes,
                               embed_dim=parameters["embed_dim"],
                               num_heads=parameters["num_heads"],
                               num_HGT_layers=parameters["num_HGT_layers"],
                               num_actor_layers=parameters["num_actor_layers"],
                               num_critic_layers=parameters["num_critic_layers"],
                               use_local_critic=False).to(device)

        checkpoint = torch.load(ct_model_path, map_location=torch.device(device), weights_only=True)
        ct_agent.load_state_dict(checkpoint['model_state_dict'])

        for i in range(config.num_iterations):
            random.seed(random_seed + i)

            start = time.time()
            fjsp_state, _ = env.reset()
            done = False

            while not done:
                mode = "fjsp" if env.scheduling_mode == "machine" else "ct"

                if mode == "fjsp":
                    fjsp_action, _ = fjsp_agent.act(graph_feature=fjsp_state.graph_feature,
                                                    pairwise_feature=fjsp_state.pairwise_feature,
                                                    mask=fjsp_state.mask,
                                                    current_operations=fjsp_state.current_operations,
                                                    reorder_idx=fjsp_state.reorder_idx)

                    next_ct_state, _, reward, done = env.step(fjsp_action)
                else:
                    ct_action, _ = ct_agent.act(graph_feature=ct_state.graph_feature,
                                                pairwise_feature=ct_state.pairwise_feature,
                                                mask=ct_state.mask)

                    next_fjsp_state, _, reward, done = env.step(ct_action)

                if mode == "fjsp":
                    ct_state = next_ct_state
                else:
                    fjsp_state = next_fjsp_state

                if done:
                    finish = time.time()
                    makespan = env.sink.completion_time
                    computing_time = finish - start
                    break

            makespans[i].append(makespan)
            computing_times[i].append(computing_time)

            print("%d/%d iteration for %s done" % (i + 1, config.num_iterations, instance_name))

    makespans = np.array(makespans).transpose()
    computing_times = np.array(computing_times).transpose()

    return makespans, computing_times


if __name__ == "__main__":
    config = get_config()

    test_case = [(10, 5), (15, 5), (20, 5),
                 (15, 10), (20, 10), (25, 10),
                 (20, 15), (25, 15), (30, 15)]

    for num_jobs, num_machines in test_case:
        config.data_dir = "./input/case1/test/%d-%d/" % (num_jobs, num_machines)
        config.res_dir = "./output/case1/test/%d-%d/CTDE/" % (num_jobs, num_machines)

        if not os.path.exists(config.res_dir):
            os.makedirs(config.res_dir)

        index = [int(os.path.splitext(filename)[0].split("-")[1])
                 for filename in os.listdir(config.data_dir)
                 if os.path.splitext(filename)[1] == '.xlsx']
        columns = [i for i in range(config.num_iterations)]

        if (config.param_dir is not None) and (config.fjsp_model_dir is not None) and (config.ct_model_dir is not None):
            param_dir = config.param_dir
            fjsp_model_dir = config.fjsp_model_dir
            ct_model_dir = config.ct_model_dir
        else:
            param_dir = "./output/train/MARL/log/%d-%d/CTDE/" % (config.num_jobs, config.num_machines)
            fjsp_model_dir = "./output/train/MARL/model/%d-%d/CTDE/%s/" % (config.num_jobs, config.num_machines, "FJSP")
            ct_model_dir = "./output/train/MARL/model/%d-%d/CTDE/%s/" % (config.num_jobs, config.num_machines, "CT")

        fjsp_episode = max(
            int(os.path.splitext(filename)[0].split("-")[1])
            for filename in os.listdir(fjsp_model_dir)
            if os.path.splitext(filename)[1] == '.pt'
        )

        ct_episode = max(
            int(os.path.splitext(filename)[0].split("-")[1])
            for filename in os.listdir(ct_model_dir)
            if os.path.splitext(filename)[1] == '.pt'
        )

        config.param_path = param_dir + "parameters.json"
        config.fjsp_model_path = fjsp_model_dir + "episode-%d.pt" % fjsp_episode
        config.ct_model_path = ct_model_dir + "episode-%d.pt" % ct_episode

        makespans, computing_times = test(config)

        df_makespan = pd.DataFrame(makespans, index=index, columns=columns)
        df_computing_time = pd.DataFrame(computing_times, index=index, columns=columns)

        df_makespan["avg"] = df_makespan.mean(axis=1)
        df_computing_time["avg"] = df_computing_time.mean(axis=1)

        df_makespan.loc["avg"] = df_makespan.mean(axis=0)
        df_computing_time.loc["avg"] = df_computing_time.mean(axis=0)

        file_name = "(%s+%s) test results.xlsx" % ("RL", "RL")
        writer = pd.ExcelWriter(config.res_dir + file_name)
        df_makespan.to_excel(writer, sheet_name="makespan")
        df_computing_time.to_excel(writer, sheet_name="computing_time")
        writer.close()

        print("==========Test of CTDE approach finished==========")