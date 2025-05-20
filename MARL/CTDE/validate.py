import os
import torch

from Environment.environment import Factory


def evaluate(agent, val_dir):
    agent.fjsp_network.eval()
    agent.ct_network.eval()
    device = agent.device

    val_paths = os.listdir(val_dir)
    makespan_lst = []

    with torch.no_grad():
        for path in val_paths:
            if path.split(".")[-1] != "xlsx":
                continue

            env = Factory(val_dir + path,
                          device=device,
                          algorithm=("RL", "RL"),
                          use_recording=False,
                          use_communication=agent.ct_network.use_communication,
                          return_global_state=False)

            fjsp_state, _ = env.reset()

            while True:
                mode = "fjsp" if env.scheduling_mode == "machine" else "ct"

                if mode == "fjsp":
                    fjsp_action, _, _ = agent.get_action(local_state=fjsp_state,
                                                         scheduling_mode="fjsp")
                    next_ct_state, _, reward, done = env.step(fjsp_action)
                else:
                    ct_action, _, _ = agent.get_action(local_state=ct_state,
                                                       scheduling_mode="ct")
                    next_fjsp_state, _, reward, done = env.step(ct_action)

                if mode == "fjsp":
                    ct_state = next_ct_state
                else:
                    fjsp_state = next_fjsp_state

                if done:
                    break

            makespan_lst.append(env.sink.completion_time)

        makespan_avg = sum(makespan_lst) / len(makespan_lst)

        return makespan_avg