import os
import torch

from Environment.environment import Factory


def evaluate(fjsp_agent, ct_agent, val_dir):
    if fjsp_agent.name == "RL":
        fjsp_agent.network.eval()
    if ct_agent.name == "RL":
        ct_agent.network.eval()

    val_paths = os.listdir(val_dir)
    makespan_lst = []

    with torch.no_grad():
        for path in val_paths:
            env = Factory(val_dir + path,
                          algorithm=(fjsp_agent.name, ct_agent.name),
                          use_recording=False)

            state = env.reset()

            while True:
                if env.scheduling_mode == "machine":
                    action = fjsp_agent.act(state)
                else:
                    action = ct_agent.act(state)

                next_state, reward, done = env.step(action)
                state = next_state

                if done:
                    break

            makespan_lst.append(env.sink.completion_time)

        makespan_avg = sum(makespan_lst) / len(makespan_lst)

        return makespan_avg