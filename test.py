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

    parser.add_argument("--num_jobs", type=int, default=10, help="number of jobs")
    parser.add_argument("--num_machines", type=int, default=5, help="number of machines")

    parser.add_argument("--fjsp_algorithm", type=str, default="RL", help="fjsp agent")
    parser.add_argument("--ct_algorithm", type=str, default="SETT", help="ct agent")

    parser.add_argument("--fjsp_model_path", type=str, default=None, help="model file path for the fjsp agent")
    parser.add_argument("--fjsp_param_path", type=str, default=None, help="hyper-parameter file path for the fjsp agent")
    parser.add_argument("--ct_model_path", type=str, default=None, help="model file path for the ct agent")
    parser.add_argument("--ct_param_path", type=str, default=None, help="hyper-parameter file path for the ct agent")

    parser.add_argument("--data_dir", type=str, default=None, help="test data path")
    parser.add_argument("--res_dir", type=str, default=None, help="test result file path")
    parser.add_argument("--sim_dir", type=str, default=None, help="simulation log file path")

    return parser.parse_args()


def test(config):
    use_cuda = torch.cuda.is_available() and not config.no_cuda
    use_recording = False if config.no_record else True

    if use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    random_seed = config.random_seed

    fjsp_algorithm = config.fjsp_algorithm
    ct_algorithm = config.ct_algorithm

    data_dir = config.data_dir
    test_paths = os.listdir(data_dir)

    progress = 0
    makespans = []
    computing_times = []

    for filename in test_paths:
        if filename.split(".")[-1] != "xlsx":
            continue

        random.seed(random_seed)

        data_src = data_dir + filename
        env = Factory(data_src, algorithm=(fjsp_algorithm, ct_algorithm), use_recording=use_recording)

        if fjsp_algorithm == "RL":
            model_path = config.fjsp_model_path
            param_path = config.fjsp_param_path

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

            checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=True)
            fjsp_agent.load_state_dict(checkpoint['model_state_dict'])
        else:
            fjsp_agent = FJSPHeuristic(fjsp_algorithm)

        if ct_algorithm == "RL":
            model_path = config.ct_model_path
            param_path = config.ct_param_path

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

            checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=True)
            ct_agent.load_state_dict(checkpoint['model_state_dict'])
        else:
            ct_agent = CTHeuristic(ct_algorithm)

        start = time.time()
        fjsp_state = env.reset()
        done = False

        while not done:
            mode = "fjsp" if env.scheduling_mode == "machine" else "ct"

            if mode == "fjsp":
                if fjsp_algorithm == "RL":
                    fjsp_action, _, _ = fjsp_agent.act(graph_feature=fjsp_state.graph_feature,
                                                       pairwise_feature=fjsp_state.pairwise_feature,
                                                       mask=fjsp_state.mask,
                                                       current_operations=fjsp_state.current_operations,
                                                       reorder_idx=fjsp_state.reorder_idx)
                else:
                    fjsp_action = fjsp_agent.act(fjsp_state)

                next_ct_state, reward, done = env.step(fjsp_action)
            else:
                if ct_algorithm == "RL":
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

        makespans.append(makespan)
        computing_times.append(computing_time)

        progress += 1
        print("%d/%d test for %s done" % (progress, len(test_paths) - 1, fjsp_algorithm + "+" + ct_algorithm))

    return makespans, computing_times


if __name__ == "__main__":
    config = get_config()

    test_case = [(fjsp, ct)
                 for fjsp in ["SPT", "MOR", "MWKR", "RAND"]
                 for ct in ["SETT", "LOR", "LWKR", "RAND"]]
    file_name = "(Heuristics) test results.xlsx"

    # test_case = [("RL", "SETT"), ("RL", "LOR"), ("RL", "LWKR"), ("RL", "RAND"),
    #               ("SPT", "RL"), ("MOR", "RL"), ("MWKR", "RL"), ("RAND", "RL")]
    # file_name = "(Single RL) test results.xlsx"

    config.data_dir = "./input/test/%d-%d/" % (config.num_jobs, config.num_machines)
    config.res_dir = "./output/test/%d-%d/" % (config.num_jobs, config.num_machines)

    if not os.path.exists(config.res_dir):
        os.makedirs(config.res_dir)

    index = [int(os.path.splitext(filename)[0].split("-")[1])
             for filename in os.listdir(config.data_dir)
             if os.path.splitext(filename)[1] == '.xlsx'] + ["avg"]
    columns = [case[0] + "+" + case[1] for case in test_case]

    df_makespan = pd.DataFrame(index=index, columns=columns)
    df_computing_time = pd.DataFrame(index=index, columns=columns)

    for fjsp_algorithm, ct_algorithm in test_case:
        config.fjsp_algorithm = fjsp_algorithm
        config.ct_algorithm = ct_algorithm

        if fjsp_algorithm == "RL":
            param_dir = ("./output/train/log/%d-%d/%s-%s/"
                         % (config.num_jobs, config.num_machines, fjsp_algorithm, ct_algorithm))
            model_dir = ("./output/train/model/%d-%d/%s-%s/"
                         % (config.num_jobs, config.num_machines, fjsp_algorithm, ct_algorithm))

            episode = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(model_dir)
                if os.path.splitext(filename)[1] == '.pt'
            )

            config.fjsp_param_path = param_dir + "parameters.json"
            config.fjsp_model_path = model_dir + "episode-%d.pt" % episode

        if ct_algorithm == "RL":
            param_dir = ("./output/train/log/%d-%d/%s-%s/"
                         % (config.num_jobs, config.num_machines, fjsp_algorithm, ct_algorithm))
            model_dir = ("./output/train/model/%d-%d/%s-%s/"
                         % (config.num_jobs, config.num_machines, fjsp_algorithm, ct_algorithm))

            episode = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(model_dir)
                if os.path.splitext(filename)[1] == '.pt'
            )

            config.ct_param_path = param_dir + "parameters.json"
            config.ct_model_path = model_dir + "episode-%d.pt" % episode

        makespans, computing_times = test(config)

        algorithm = fjsp_algorithm + "+" + ct_algorithm
        df_makespan[algorithm] = makespans + [sum(makespans) / len(makespans)]
        df_computing_time[algorithm] = computing_times + [sum(computing_times) / len(computing_times)]
        print("==========test for %s finished==========" % algorithm)

    writer = pd.ExcelWriter(config.res_dir + file_name)
    df_makespan.to_excel(writer, sheet_name="makespan")
    df_computing_time.to_excel(writer, sheet_name="computing_time")
    writer.close()