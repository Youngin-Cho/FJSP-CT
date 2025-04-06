import numpy as np


class FJSPHeuristic:
    def __init__(self, name):
        self.name = name

    def act(self, state):
        priority_idx = state.priority_idx.flatten()
        mask = state.mask.transpose(0, 1).flatten().cpu().numpy()
        priority_idx[~mask] = 0.0

        candidates = np.where(priority_idx == np.max(priority_idx))[0]
        action = np.random.choice(candidates)

        return int(action)