import numpy as np


class MachineSchedulingHeuristic:
    def __init__(self):
        pass

    def act(self, state):
        priority_idx = state.data.flatten()
        mask = state.mask.flatten()
        priority_idx[~mask] = 0.0

        candidates = np.where(priority_idx == np.max(priority_idx))[0]
        action = np.random.choice(candidates)

        return int(action)