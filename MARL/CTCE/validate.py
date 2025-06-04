import os
import torch

from Environment.environment import Factory


def evaluate(agent, val_dir):
    agent.global_network.eval()
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
                          use_centralized_scheduling=True,
                          return_global_state=True,
                          global_state_encoding="EP")

            _, global_state = env.reset()

            while True:
                mode = "fjsp" if env.scheduling_mode == "machine" else "ct"

                if mode == "fjsp":
                    action, _, _ = agent.get_action(global_state=global_state)

                    job_id = action // int((env.num_machines + env.num_buffers + env.num_outputpoints) * (env.num_cranes + 1))
                    location_id = (action % int((env.num_machines + env.num_buffers + env.num_outputpoints) * (env.num_cranes + 1))) // (env.num_cranes + 1)
                    crane_id = (action % int((env.num_machines + env.num_buffers + env.num_outputpoints) * (env.num_cranes + 1))) % (env.num_cranes + 1)

                    fjsp_action = job_id * (env.num_machines + env.num_buffers + env.num_outputpoints) + location_id
                    _, _, reward, done = env.step(fjsp_action)
                else:
                    ct_action = env.num_operations * (env.num_cranes + 1) + crane_id

                    _, next_global_state, ct_reward, done = env.step(ct_action)

                if mode == "ct":
                    global_state = next_global_state

                if done:
                    break

            makespan_lst.append(env.sink.completion_time)

        makespan_avg = sum(makespan_lst) / len(makespan_lst)

        return makespan_avg