import numpy as np


class ReplayBuffer:
    """A circular memory of past experiences: (state, action, reward, next_state, terminated)."""

    def __init__(self, capacity, obs_size):
        self.capacity = capacity
        self.states = np.zeros((capacity, obs_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_size), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.position = 0                     # where the next experience will be written
        self.size = 0                         # how many slots are filled

    def add(self, state, action, reward, next_state, terminated):
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.terminated[i] = float(terminated)
        self.position = (i + 1) % self.capacity          # wrap around: overwrite the oldest
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)   # random, so batches aren't correlated
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.terminated[idx])

    def __len__(self):
        return self.size
    