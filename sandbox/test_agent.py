import numpy as np
import config
from env.racing_env import RacingEnv
from agent.replay_buffer import ReplayBuffer
from agent.dqn_agent import DQNAgent

results = []


def check(name, passed, detail=""):
    results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}  {detail}")


env = RacingEnv()
OBS, ACTIONS = env.obs_size, env.n_actions
rng = np.random.default_rng(0)

# 1. Replay buffer ---------------------------------------------------------
buffer = ReplayBuffer(capacity=500, obs_size=OBS)
for i in range(1000):                                  # twice the capacity, so it must wrap
    buffer.add(np.full(OBS, i), i % ACTIONS, float(i), np.full(OBS, i + 1), i % 10 == 0)
states, actions, rewards, next_states, terminated = buffer.sample(64, rng)
shapes_ok = (states.shape == (64, OBS) and actions.shape == (64,) and rewards.shape == (64,)
             and next_states.shape == (64, OBS) and terminated.shape == (64,))
check("1. Buffer shapes", shapes_ok)
check("1. Buffer wraps", len(buffer) == 500 and rewards.min() >= 500,
      f"size {len(buffer)}, oldest reward kept {buffer.rewards.min():.0f} (should be 500)")

# 2. Epsilon schedule --------------------------------------------------------
agent = DQNAgent(OBS, ACTIONS, backend="torch", seed=0)
values = []
for steps in [0, config.EPSILON_DECAY_STEPS // 2, config.EPSILON_DECAY_STEPS, config.EPSILON_DECAY_STEPS * 3]:
    agent.steps_done = steps
    values.append(round(agent.epsilon(), 3))
check("2. Epsilon schedule", values == [1.0, 0.525, 0.05, 0.05], f"{values}")

# 3. Act -----------------------------------------------------------------------
obs = env.reset()
agent.steps_done = 0
random_actions = {agent.act(obs) for _ in range(200)}
greedy = agent.act(obs, explore=False)
check("3. Act", random_actions <= set(range(ACTIONS)) and greedy == agent.online.act(obs),
      f"explored actions {sorted(random_actions)}, greedy {greedy}")

# 4. Learn (both backends) + 5. Target network -------------------------------------
for backend in ["torch", "numpy"]:
    agent = DQNAgent(OBS, ACTIONS, backend=backend, seed=0, warmup_steps=200,
                     target_update_every=10_000_000)
    obs = env.reset()
    losses = []
    for _ in range(400):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.remember(obs, action, reward, next_obs, terminated)
        loss = agent.learn()
        if loss is not None:
            losses.append(loss)
        obs = env.reset() if (terminated or truncated) else next_obs
    check(f"4. Learn ({backend})", len(losses) > 0 and np.all(np.isfinite(losses)),
          f"{len(losses)} learning steps, last loss {losses[-1]:.4f}")

    sample = np.random.rand(8, OBS).astype(np.float32)
    differ = not np.allclose(agent.q_values(sample), agent.q_values(sample, agent.target))
    agent.update_target()
    same = np.allclose(agent.q_values(sample), agent.q_values(sample, agent.target))
    check(f"5. Target network ({backend})", differ and same,
          f"different before refresh: {differ}, identical after: {same}")

# 6. Maths sanity check: Q should converge to a value you can work out by hand -----
def sanity(backend, terminal, expected):
    agent = DQNAgent(OBS, 2, backend=backend, seed=1, gamma=0.9, learning_rate=1e-3,
                     warmup_steps=64, target_update_every=10_000_000)
    state = np.full(OBS, 0.5, dtype=np.float32)
    for _ in range(500):
        agent.remember(state, int(rng.integers(2)), 1.0, state, terminal)
    for step in range(3000):
        agent.learn()
        if step % 50 == 0:
            agent.update_target()
    value = float(agent.q_values(state).mean())
    check(f"6. Sanity Q = {expected} ({backend}, {'terminal' if terminal else 'never ends'})",
          abs(value - expected) < 0.3 * max(1, expected / 10), f"Q = {value:.3f}")

for backend in ["torch", "numpy"]:
    sanity(backend, terminal=False, expected=10.0)     # Q = 1 + 0.9 Q  ->  Q = 10
    sanity(backend, terminal=True, expected=1.0)       # no future       ->  Q = 1

print(f"\n{sum(results)} / {len(results)} checks passed")