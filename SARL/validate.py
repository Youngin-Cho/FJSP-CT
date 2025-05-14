import os
import torch

from Environment.environment import Factory


def evaluate(fjsp_agent, ct_agent, val_dir):
    if fjsp_agent.name == "RL":
        fjsp_agent.network.eval()
        device = fjsp_agent.device
    if ct_agent.name == "RL":
        ct_agent.network.eval()
        device = ct_agent.device

    val_paths = os.listdir(val_dir)
    makespan_lst = []

    with torch.no_grad():
        for path in val_paths:
            if path.split(".")[-1] != "xlsx":
                continue

            env = Factory(val_dir + path,
                          device=device,
                          algorithm=(fjsp_agent.name, ct_agent.name),
                          use_recording=False,
                          return_global_state=False)

            fjsp_state = env.reset()

            while True:
                mode = "fjsp" if env.scheduling_mode == "machine" else "ct"

                if mode == "fjsp":
                    if fjsp_agent.name == "RL":
                        fjsp_action, _, _ = fjsp_agent.get_action(fjsp_state)
                    else:
                        fjsp_action = fjsp_agent.act(fjsp_state)

                    next_ct_state, _, reward, done = env.step(fjsp_action)
                else:
                    if ct_agent.name == "RL":
                        ct_action, _, _ = ct_agent.get_action(ct_state)
                    else:
                        ct_action = ct_agent.act(ct_state)

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