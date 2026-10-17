import sys
import time
import random
import numpy as np
import config
from env.racing_env import RacingEnv


def manual_test():
    """You drive using the AI's 5 actions. Rays visible, rewards printed."""
    import pygame
    env = RacingEnv(render=True)
    observation = env.reset()
    print("Manual test: Up = accelerate, Up+Left / Up+Right = turn, Down = brake, nothing = coast")

    while not env.closed:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and keys[pygame.K_LEFT]:
            action = 1
        elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
            action = 2
        elif keys[pygame.K_UP]:
            action = 0
        elif keys[pygame.K_DOWN]:
            action = 3
        else:
            action = 4

        observation, reward, terminated, truncated, info = env.step(action)

        if abs(reward - config.REWARD_STEP) > 1e-6:
            print(f"  reward {reward:+.2f}   (checkpoints {info['checkpoints']}, laps {info['laps']})")

        if terminated or truncated:
            print(f"EPISODE OVER: {info['reason']} | total reward {info['episode_reward']:+.2f} | "
                  f"{info['episode_steps']} steps")
            if not env.closed:
                observation = env.reset()


def random_test(episodes=20):
    """Random actions. Checks shapes, ranges and episode endings."""
    env = RacingEnv(render=False)
    lowest = np.full(env.obs_size, np.inf)
    highest = np.full(env.obs_size, -np.inf)

    for episode in range(episodes):
        observation = env.reset()
        assert observation.shape == (env.obs_size,), observation.shape
        done = False
        while not done:
            action = random.randrange(env.n_actions)
            observation, reward, terminated, truncated, info = env.step(action)
            lowest = np.minimum(lowest, observation)
            highest = np.maximum(highest, observation)
            done = terminated or truncated
        print(f"Episode {episode + 1:2d}: {info['reason']:15s} reward {info['episode_reward']:+7.2f}   "
              f"steps {info['episode_steps']:4d}   checkpoints {info['checkpoints']}")

    print("\nObservation size:", env.obs_size, "| actions:", env.n_actions)
    print("Lowest  values seen:", np.round(lowest, 2))
    print("Highest values seen:", np.round(highest, 2))


def benchmark(decisions=5000):
    """How fast does the environment run without a window?"""
    env = RacingEnv(render=False)
    env.reset()
    start = time.perf_counter()
    for _ in range(decisions):
        action = random.choice([0, 0, 0, 1, 2])          # mostly accelerate, so episodes last a bit
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    seconds = time.perf_counter() - start
    print(f"{decisions} decisions in {seconds:.2f} s")
    print(f"= {decisions / seconds:,.0f} decisions per second "
          f"({decisions * config.FRAME_SKIP / seconds:,.0f} physics steps per second)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "manual"
    if mode == "manual":
        manual_test()
    elif mode == "random":
        random_test()
    elif mode == "bench":
        benchmark()
    else:
        print("Use: python -m sandbox.test_env [manual | random | bench]")