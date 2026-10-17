import numpy as np
import torch
import torch.nn as nn
import config
from agent.replay_buffer import ReplayBuffer
from agent.network import QNetwork
from agent.numpy_network import NumpyQNetwork, huber_loss, Adam


class DQNAgent:
    def __init__(self, obs_size, n_actions, backend=None, seed=None, **settings):
        # Every setting comes from config.py unless overridden here (the tests use this)
        self.gamma = settings.get("gamma", config.GAMMA)
        self.learning_rate = settings.get("learning_rate", config.LEARNING_RATE)
        self.batch_size = settings.get("batch_size", config.BATCH_SIZE)
        self.warmup_steps = settings.get("warmup_steps", config.WARMUP_STEPS)
        self.epsilon_start = settings.get("epsilon_start", config.EPSILON_START)
        self.epsilon_end = settings.get("epsilon_end", config.EPSILON_END)
        self.epsilon_decay_steps = settings.get("epsilon_decay_steps", config.EPSILON_DECAY_STEPS)
        self.target_update_every = settings.get("target_update_every", config.TARGET_UPDATE_EVERY)
        self.double_dqn = settings.get("double_dqn", config.DOUBLE_DQN)
        self.grad_clip = settings.get("grad_clip", config.GRAD_CLIP)
        buffer_size = settings.get("buffer_size", config.BUFFER_SIZE)

        self.n_actions = n_actions
        self.backend = backend or config.AGENT_BACKEND
        self.rng = np.random.default_rng(seed)
        self.buffer = ReplayBuffer(buffer_size, obs_size)
        self.steps_done = 0                  # decisions remembered so far
        self.learn_steps = 0

        if self.backend == "torch":
            if seed is not None:
                torch.manual_seed(seed)
            self.online = QNetwork(obs_size, n_actions)
            self.target = QNetwork(obs_size, n_actions)
            self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.learning_rate)
            self.loss_fn = nn.HuberLoss()
        elif self.backend == "numpy":
            self.online = NumpyQNetwork(obs_size, n_actions, seed=seed)
            self.target = NumpyQNetwork(obs_size, n_actions)
            self.optimizer = Adam(self.online.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        self.update_target()

    # ------------------------------------------------------------------
    #  Exploration and acting
    # ------------------------------------------------------------------
    def epsilon(self):
        fraction = min(1.0, self.steps_done / self.epsilon_decay_steps)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def act(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon():
            return int(self.rng.integers(self.n_actions))      # explore: random action
        return self.online.act(state)                         # exploit: best known action

    def q_values(self, states, network=None):
        """Q-values as a NumPy array, for one state or a batch (handy for tests and watching)."""
        network = network or self.online
        states = np.asarray(states, dtype=np.float32)
        single = states.ndim == 1
        if single:
            states = states[None, :]
        if self.backend == "torch":
            with torch.no_grad():
                q = network(torch.from_numpy(states)).numpy()
        else:
            q = network(states)
        return q[0] if single else q

    # ------------------------------------------------------------------
    #  Memory
    # ------------------------------------------------------------------
    def remember(self, state, action, reward, next_state, terminated):
        self.buffer.add(state, action, reward, next_state, terminated)
        self.steps_done += 1
        if self.steps_done % self.target_update_every == 0:
            self.update_target()

    def update_target(self):
        self.target.copy_from(self.online)

    # ------------------------------------------------------------------
    #  Learning
    # ------------------------------------------------------------------
    def learn(self):
        if len(self.buffer) < max(self.warmup_steps, self.batch_size):
            return None                                       # still warming up
        batch = self.buffer.sample(self.batch_size, self.rng)
        self.learn_steps += 1
        if self.backend == "torch":
            return self._learn_torch(*batch)
        return self._learn_numpy(*batch)

    def _learn_torch(self, states, actions, rewards, next_states, terminated):
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        next_states = torch.from_numpy(next_states)
        terminated = torch.from_numpy(terminated)

        # 1. Bellman targets (no gradients: these are the "answers")
        with torch.no_grad():
            if self.double_dqn:
                best_next = self.online(next_states).argmax(dim=1, keepdim=True)   # online CHOOSES
                next_value = self.target(next_states).gather(1, best_next).squeeze(1)  # target EVALUATES
            else:
                next_value = self.target(next_states).max(dim=1).values
            targets = rewards + self.gamma * (1 - terminated) * next_value

        # 2. Q-values of the actions actually taken
        q_taken = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 3. Loss, backprop, clip, update
        loss = self.loss_fn(q_taken, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()
        return loss.item()

    def _learn_numpy(self, states, actions, rewards, next_states, terminated):
        rows = np.arange(len(actions))

        # 1. Bellman targets. Done FIRST, because every forward pass overwrites the
        #    inputs each layer remembers, and backward() needs the ones from `states`.
        next_q_target = self.target(next_states)
        if self.double_dqn:
            best_next = np.argmax(self.online(next_states), axis=1)        # online CHOOSES
            next_value = next_q_target[rows, best_next]                     # target EVALUATES
        else:
            next_value = next_q_target.max(axis=1)
        targets = rewards + self.gamma * (1 - terminated) * next_value

        # 2. Q-values of the actions actually taken
        q_all = self.online(states)
        q_taken = q_all[rows, actions]

        # 3. Loss and gradient: only the taken action's output gets a gradient
        loss, grad_taken = huber_loss(q_taken, targets)
        grad_all = np.zeros_like(q_all)
        grad_all[rows, actions] = grad_taken
        self.online.backward(grad_all)

        # 4. Gradient clipping by total norm, then your Adam
        grads = self.online.gradients()
        total_norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
        if total_norm > self.grad_clip:
            grads = [g * (self.grad_clip / total_norm) for g in grads]
        self.optimizer.step(grads)
        return float(loss)

    # ------------------------------------------------------------------
    #  Saving
    # ------------------------------------------------------------------
    def save(self, path):
        self.online.save(path)

    def load(self, path):
        self.online.load(path)
        self.update_target()