import os
import torch

from Environment.environment import Factory


def evaluate(agent, val_dir):
    agent.network.eval()
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
                          return_global_state=True)

            fjsp_state = env.reset()

            while True:
                mode = "fjsp" if env.scheduling_mode == "machine" else "ct"

                if mode == "fjsp":
                    fjsp_action, _, _ = fjsp_agent.get_action(fjsp_state)
                    next_ct_state, reward, done = env.step(fjsp_action)
                else:
                    ct_action, _, _ = ct_agent.get_action(ct_state)
                    next_fjsp_state, reward, done = env.step(ct_action)

                if mode == "fjsp":
                    ct_state = next_ct_state
                else:
                    fjsp_state = next_fjsp_state

                if done:
                    break

            makespan_lst.append(env.sink.completion_time)

        makespan_avg = sum(makespan_lst) / len(makespan_lst)

        return makespan_avg