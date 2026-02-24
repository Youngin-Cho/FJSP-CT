import os
import sys
import time
import json
import random
import argparse
import torch
import numpy as np
import pandas as pd
import time

from datetime import datetime
from collections import OrderedDict
from Environment.environment import Factory
from Agent.FlexibleJobShop.network import FJSPScheduler
from Agent.CraneTransportation.network import CTScheduler


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, OrderedDict):
            return dict(obj)  # OrderedDict도 dict로 바꿔주기
        else:
            return super(NpEncoder, self).default(obj)


def get_config():
    parser = argparse.ArgumentParser(description="FJSP")

    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_record', action='store_true', help="Disable Recording events")
    parser.add_argument('--no_communication', action='store_true', help="Disable communication")

    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument("--num_jobs", type=int, default=10, help="number of jobs")
    parser.add_argument("--num_machines", type=int, default=5, help="number of machines")

    parser.add_argument("--fjsp_model_path", type=str, default=None, help="model file path for the fjsp agent")
    parser.add_argument("--ct_model_path", type=str, default=None, help="model file path for the ct agent")
    parser.add_argument("--param_path", type=str, default=None, help="hyper-parameter file path")

    parser.add_argument("--data_path", type=str, default=None, help="data path")
    parser.add_argument("--log_dir", type=str, default=None, help="log file path")

    return parser.parse_args()


def simulate(config):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_filename = f"log_{timestamp}.xlsx"

    use_cuda = torch.cuda.is_available() and not config.no_cuda
    use_recording = False if config.no_record else True
    use_communication = False if config.no_communication else True

    if use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    fjsp_model_path = config.fjsp_model_path
    ct_model_path = config.ct_model_path
    param_path = config.param_path

    data_path = config.data_path
    log_dir = config.log_dir

    ymd = time.strftime('%Y%m%d')
    hour = str(time.localtime().tm_hour)
    minute = str(time.localtime().tm_min)
    second = str(time.localtime().tm_sec)

    log_path = log_dir + 'UnityLog_%s_%sh_%sm_%ss.json' % (ymd, hour, minute, second)
    print('File Stored in:', log_path)
    unity_log = []

    env = Factory(
        data_path,
        algorithm=("RL", "RL"),
        use_recording=use_recording,
        use_communication=use_communication,
        return_global_state=False
    )

    # 에이전트별 네트워크 구성 관련 파라미터 로딩
    with open(param_path, 'r') as f:
        parameters = json.load(f)

    # FJSP 에이전트 로딩
    fjsp_agent = FJSPScheduler(
        meta_data=env.fjsp_meta_data,
        state_size=env.fjsp_state_size,
        num_nodes=env.fjsp_num_nodes,
        embed_dim=parameters["embed_dim"],
        num_heads=parameters["num_heads"],
        num_HGT_layers=parameters["num_HGT_layers"],
        num_actor_layers=parameters["num_actor_layers"],
        num_critic_layers=parameters["num_critic_layers"],
        use_local_critic=True
    ).to(device)

    checkpoint = torch.load(fjsp_model_path, map_location=torch.device(device), weights_only=True)
    fjsp_agent.load_state_dict(checkpoint['model_state_dict'])

    # CT 에이전트 로딩
    ct_agent = CTScheduler(
        meta_data=env.ct_meta_data,
        state_size=env.ct_state_size,
        num_nodes=env.ct_num_nodes,
        embed_dim=parameters["embed_dim"],
        num_heads=parameters["num_heads"],
        num_HGT_layers=parameters["num_HGT_layers"],
        num_actor_layers=parameters["num_actor_layers"],
        num_critic_layers=parameters["num_critic_layers"],
        use_local_critic=True,
        use_communication=use_communication
    ).to(device)

    checkpoint = torch.load(ct_model_path, map_location=torch.device(device), weights_only=True)
    ct_agent.load_state_dict(checkpoint['model_state_dict'])

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    start = time.time()
    fjsp_state, _ = env.reset()
    done = False
    idx = 0

    while not done:
        if env.scheduling_mode == "machine":
            mode = "fjsp"
            message = OrderedDict([
                ("MessageID", idx),
                ("SimulTime", env.sim_env.now),
                ("MDP", OrderedDict([
                    ("FJSP_Agent", dict()),
                    ("CT_Agent", dict())
                ]))
            ])
        else:
            mode = "ct"

        if mode == "fjsp":
            # FJSP 에이전트 행동 선택
            fjsp_action, _, _, fjsp_probs = fjsp_agent.act(
                graph_feature=fjsp_state.graph_feature,
                pairwise_feature=fjsp_state.pairwise_feature,
                mask=fjsp_state.mask,
                current_operations=fjsp_state.current_operations,
                reorder_idx=fjsp_state.reorder_idx,
                return_probs=True
            )

            # State 저장
            node_feature_0 = fjsp_state.graph_feature.node_stores[0]['x'].cpu().tolist()
            node_feature_1 = fjsp_state.graph_feature.node_stores[1]['x'].cpu().tolist()
            node_feature_2 = fjsp_state.graph_feature.node_stores[2]['x'].cpu().tolist()
            node_feature_3 = fjsp_state.graph_feature.node_stores[3]['x'].cpu().tolist()
            pairwise_feature = fjsp_state.pairwise_feature.cpu().tolist()

            state = {
                'node_operation': node_feature_0,
                'node_machine': node_feature_1,
                'node_buffer': node_feature_2,
                'node_output': node_feature_3,
                'pairwise_feature': pairwise_feature
            }

            message['MDP']['FJSP_Agent'] = {'State': state}

            # Action, Mask, Prob 저장
            message['MDP']['FJSP_Agent']['Action'] = fjsp_action
            message['MDP']['FJSP_Agent']['Action_prob'] = fjsp_probs.cpu().tolist()
            message['MDP']['FJSP_Agent']['Mask'] = fjsp_state.mask.cpu().tolist()

            next_ct_state, _, reward, done = env.step(fjsp_action)
        else:
            ct_action, _, _, ct_probs = ct_agent.act(
                graph_feature=ct_state.graph_feature,
                pairwise_feature=ct_state.pairwise_feature,
                mask=ct_state.mask,
                return_probs=True
            )

            # State 저장
            node_feature_0 = ct_state.graph_feature.node_stores[0]['x'].cpu().tolist()
            node_feature_1 = ct_state.graph_feature.node_stores[1]['x'].cpu().tolist()
            pairwise_feature = ct_state.pairwise_feature.cpu().tolist()

            # JSON에 저장할 state 구성
            state = {
                'node_crane': node_feature_0,
                'node_operation': node_feature_1,
                'pairwise_feature': pairwise_feature
            }

            message['MDP']['CT_Agent'] = {'State': state}

            # Action, Mask, Prob 저장
            message['MDP']['CT_Agent']['Action'] = ct_action
            message['MDP']['CT_Agent']['Action_prob'] = ct_probs.cpu().tolist()
            message['MDP']['CT_Agent']['Mask'] = ct_state.mask.cpu().tolist()

            next_fjsp_state, _, reward, done = env.step(ct_action)

        if mode == "fjsp":
            ct_state = next_ct_state
        else:
            fjsp_state = next_fjsp_state

            print(message['MessageID'], 'Received!')
            print(env.sink.num_jobs_degenerated, "Completed!")
            unity_log.append(message)
            idx += 1

        if done:
            finish = time.time()
            makespan = env.sink.completion_time
            computing_time = finish - start

            _ = env.monitor.get_logs(log_dir + log_filename)
            break

    # JSON 파일 저장
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(unity_log, f, cls=NpEncoder, ensure_ascii=False, indent=2)
    print("Makespan: %.4f" % makespan)


if __name__ == "__main__":
    config = get_config()

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    simulate(config)