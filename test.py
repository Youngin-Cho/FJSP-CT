import os
import time
import json
import random
import argparse
import torch
import pandas as pd

from torch.distributions.categorical import Categorical
from Environment.environment import Factory
from Agent.FlexibleJobShop.network import FJSPScheduler
from Agent.CraneTransportation.network import CTScheduler
from Agent.FlexibleJobShop.heuristic import FJSPHeuristic
from Agent.CraneTransportation.heuristic import CTHeuristic


def get_config():
    parser = argparse.ArgumentParser(description="FJSP")

    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_record', action='store_true', help="Disable Recording events")

    parser.add_argument("--random_seed", type=int, default=42, help="random seed")

    parser.add_argument("--model_path", type=str, default=None, help="model file path")
    parser.add_argument("--param_path", type=str, default=None, help="hyper-parameter file path")
    parser.add_argument("--data_dir", type=str, default=None, help="test data path")
    parser.add_argument("--res_dir", type=str, default=None, help="test result file path")
    parser.add_argument("--sim_dir", type=str, default=None, help="simulation log file path")

    return parser.parse_args()


if __name__ == "__main__":
    config = get_config()

    use_cuda = torch.cuda.is_available() and not config.no_cuda
    use_recording = False if config.no_record else True

    if use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    random_seed = config.random_seed

    model_path = config.model_path
    param_path = config.param_path
    data_dir = [config.data_dir]
    res_dir = [config.res_dir]
    sim_dir = [config.sim_dir]

    for res_dir_temp in res_dir:
        if not os.path.exists(res_dir_temp):
            os.makedirs(res_dir_temp)

    fjsp_algorithms = ["SPT", "MOR", "MWKR", "RAND"]
    ct_algorithms = ["RL", "SETT", "LOR", "LWKR", "RAND"]
    algorithms = [fjsp_algo + "+" + ct_algo for fjsp_algo in fjsp_algorithms for ct_algo in ct_algorithms]

    for data_dir_temp, res_dir_temp in zip(data_dir, res_dir):
        test_paths = os.listdir(data_dir_temp)
        index = ["P%d" % i for i in range(1, len(test_paths))] + ["avg"]
        columns = algorithms

        df_makespan = pd.DataFrame(index=index, columns=columns)
        df_computing_time = pd.DataFrame(index=index, columns=columns)

        for name in columns:
            progress = 0
            list_makespan = []
            list_computing_time = []

            for prob, path in zip(index, test_paths):
                if path.split(".")[-1] != "xlsx":
                    continue

                random.seed(random_seed)

                data_src = data_dir_temp + path
                env = Factory(data_src, algorithm=name.split("+"), use_recording=use_recording)

                fjsp_name = name.split("+")[0]
                ct_name = name.split("+")[1]

                if fjsp_name == "RL":
                    with open(param_path, 'r') as f:
                        parameters = json.load(f)

                    fjsp_agent = FJSPScheduler(meta_data=env.fjsp_meta_data,
                                               state_size=env.fjsp_state_size,
                                               num_nodes=env.fjsp_num_nodes,
                                               embed_dim=parameters["embed_dim"],
                                               num_heads=parameters["num_heads"],
                                               num_HGT_layers=parameters["num_HGT_layers"],
                                               num_actor_layers=parameters["num_actor_layers"],
                                               num_critic_layers=parameters["num_critic_layers"]).to(device)

                    checkpoint = torch.load(model_path, map_location=torch.device(device))
                    fjsp_agent.load_state_dict(checkpoint['model_state_dict'])
                else:
                    fjsp_agent = FJSPHeuristic(name.split("+")[0])

                if ct_name == "RL":
                    with open(param_path, 'r') as f:
                        parameters = json.load(f)

                    ct_agent = CTScheduler(meta_data=env.ct_meta_data,
                                           state_size=env.ct_state_size,
                                           num_nodes=env.ct_num_nodes,
                                           embed_dim=parameters["embed_dim"],
                                           num_heads=parameters["num_heads"],
                                           num_HGT_layers=parameters["num_HGT_layers"],
                                           num_actor_layers=parameters["num_actor_layers"],
                                           num_critic_layers=parameters["num_critic_layers"]).to(device)

                    checkpoint = torch.load(model_path, map_location=torch.device(device))
                    ct_agent.load_state_dict(checkpoint['model_state_dict'])
                else:
                    ct_agent = CTHeuristic(name.split("+")[1])

                start = time.time()
                fjsp_state = env.reset()
                done = False

                while not done:
                    mode = "fjsp" if env.scheduling_mode == "machine" else "ct"

                    if mode == "fjsp":
                        if fjsp_name == "RL":
                            fjsp_action, _, _ = fjsp_agent.act(graph_feature=fjsp_state.graph_feature,
                                                               pairwise_feature=fjsp_state.pairwise_feature,
                                                               mask=fjsp_state.mask,
                                                               current_operations=fjsp_state.current_operations,
                                                               reorder_idx=fjsp_state.reorder_idx)
                        else:
                            fjsp_action = fjsp_agent.act(fjsp_state)

                        next_ct_state, reward, done = env.step(fjsp_action)
                    else:
                        if ct_name == "RL":
                            ct_action, _, _ = ct_agent.act(graph_feature=ct_state.graph_feature,
                                                           pairwise_feature=ct_state.pairwise_feature,
                                                           mask=ct_state.mask)
                        else:
                            ct_action = ct_agent.act(ct_state)

                        next_fjsp_state, reward, done = env.step(ct_action)

                    if mode == "fjsp":
                        ct_state = next_ct_state
                    else:
                        fjsp_state = next_fjsp_state

                    if done:
                        finish = time.time()
                        makespan = env.sink.completion_time
                        computing_time = finish - start
                        break

                list_makespan.append(makespan)
                list_computing_time.append(computing_time)

                progress += 1
                print("%d/%d test for %s done" % (progress, len(index) - 1, name))

            df_makespan[name] = list_makespan + [sum(list_makespan) / len(list_makespan)]
            df_computing_time[name] = list_computing_time + [sum(list_computing_time) / len(list_computing_time)]
            print("==========test for %s finished==========" % name)

        writer = pd.ExcelWriter(res_dir_temp + 'test_results.xlsx')
        df_makespan.to_excel(writer, sheet_name="makespan")
        df_computing_time.to_excel(writer, sheet_name="computing_time")
        writer.close()