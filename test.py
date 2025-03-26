import os
import time
import json
import random
import argparse
import torch
import pandas as pd

from torch.distributions.categorical import Categorical
from Environment.environment import Factory
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
    ct_algorithms = ["SETT", "LOR", "LWKR", "RAND"]
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

                if name == "RL":
                    with open(param_path, 'r') as f:
                        parameters = json.load(f)

                    config.embed_dim = parameters['embed_dim']
                    config.n_heads = parameters['n_heads']
                    config.n_layers_hgt = parameters['n_layers_hgt']
                    config.n_layers_ff = parameters['n_layers_ff']

                    config.n_layers_actor = parameters['n_layers_actor']
                    config.hidden_dim_actor = parameters['hidden_dim_actor']
                    config.n_layers_critic = parameters['n_layers_critic']
                    config.hidden_dim_critic = parameters['hidden_dim_critic']

                    # agent = SchedulingNetwork(meta_data=env.meta_data,
                    #                           num_nodes=env.num_nodes,
                    #                           input_dim_g=env.input_dim_g,
                    #                           input_dim_pair=env.input_dim_pair,
                    #                           config=config).to(device)
                    # checkpoint = torch.load(model_path, map_location=torch.device(device))
                    # agent.load_state_dict(checkpoint['model_state_dict'])
                else:
                    fjsp_agent = FJSPHeuristic(name.split("+")[0])
                    ct_agent = CTHeuristic(name.split("+")[1])

                start = time.time()
                state = env.reset()
                done = False

                while not done:
                    if name == "RL":
                        pass
                        # with torch.no_grad():
                        #     fea_g, fea_pair, mask_pair = convert_state(state, device)
                        #     probs, value = agent(fea_g, fea_pair, mask_pair)
                        #
                        # dist = Categorical(probs)
                        # action = dist.sample().item()
                    else:
                        if env.scheduling_mode == "machine":
                            action = fjsp_agent.act(state)
                        else:
                            action = ct_agent.act(state)

                    next_state, reward, done = env.step(action)

                    state = next_state

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