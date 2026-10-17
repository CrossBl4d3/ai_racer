"""Train the AI driver.   Run with:  python train.py   (options: python train.py --help)"""
import argparse
import csv
import os
import time
import numpy as np
import config
import torch
torch.set_num_threads(1)          # tiny network: one CPU thread is fastest
from env.racing_env import RacingEnv
from agent.dqn_agent import DQNAgent


def lap_seconds(steps):
    return None if steps is None else steps / config.STEPS_PER_SECOND


def evaluate(agent):
    """One greedy test drive (no random actions). The game is deterministic, so one is enough."""
    env = RacingEnv()
    observation = env.reset()
    done = False
    while not done:
        action = agent.act(observation, explore=False)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info


def plot_training(csv_path, png_path):
    import matplotlib
    matplotlib.use("Agg")                     # draw to a file, no window needed
    import matplotlib.pyplot as plt

    episodes, rewards, checkpoints, epsilons = [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            checkpoints.append(int(row["checkpoints"]) + int(row["laps"]) * config.NUM_CHECKPOINTS)
            epsilons.append(float(row["epsilon"]))
    if not episodes:
        return

    def moving_average(values, window=50):
        values = np.array(values)
        if len(values) < window:
            return values
        return np.convolve(values, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(episodes, rewards, alpha=0.25, label="each episode")
    smooth = moving_average(rewards)
    axes[0].plot(episodes[len(episodes) - len(smooth):], smooth, label="average of 50")
    axes[0].set_ylabel("Episode reward")
    axes[0].legend()

    axes[1].plot(episodes, checkpoints, alpha=0.25)
    smooth = moving_average(checkpoints)
    axes[1].plot(episodes[len(episodes) - len(smooth):], smooth)
    axes[1].set_ylabel(f"Gates passed\n({config.NUM_CHECKPOINTS} = 1 lap)")

    axes[2].plot(episodes, epsilons)
    axes[2].set_ylabel("Epsilon (exploration)")
    axes[2].set_xlabel("Episode")

    fig.suptitle("AI Racer training")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def train(steps, backend, name, seed):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    extension = ".pt" if backend == "torch" else ".npz"
    best_path = os.path.join("models", f"{name}_best{extension}")
    latest_path = os.path.join("models", f"{name}_latest{extension}")
    csv_path = os.path.join("logs", f"{name}.csv")
    png_path = os.path.join("logs", f"{name}.png")

    env = RacingEnv()
    agent = DQNAgent(env.obs_size, env.n_actions, backend=backend, seed=seed)

    log_file = open(csv_path, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["episode", "step", "reward", "checkpoints", "laps", "reason", "epsilon", "loss"])

    print(f"Training '{name}' with the {backend} network for {steps:,} decisions.")
    print("Press Ctrl+C to stop early (progress is saved).\n")

    observation = env.reset()
    episode = 0
    recent_rewards, recent_gates, recent_losses = [], [], []
    best_eval_reward = -np.inf
    best_lap_ever = None
    start_time = time.time()

    try:
        for step in range(1, steps + 1):
            # --- one decision ---
            action = agent.act(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.remember(observation, action, reward, next_observation, terminated)
            loss = agent.learn()
            if loss is not None:
                recent_losses.append(loss)
            observation = next_observation

            # --- end of an episode ---
            if terminated or truncated:
                episode += 1
                gates = info["checkpoints"] + info["laps"] * config.NUM_CHECKPOINTS
                recent_rewards.append(info["episode_reward"])
                recent_gates.append(gates)
                average_loss = np.mean(recent_losses[-100:]) if recent_losses else 0.0
                log.writerow([episode, step, round(info["episode_reward"], 3), info["checkpoints"],
                              info["laps"], info["reason"], round(agent.epsilon(), 4), round(average_loss, 5)])
                if info["best_lap_steps"] is not None:
                    if best_lap_ever is None or info["best_lap_steps"] < best_lap_ever:
                        best_lap_ever = info["best_lap_steps"]
                observation = env.reset()

            # --- progress line ---
            if step % config.PRINT_EVERY == 0:
                elapsed = time.time() - start_time
                speed = step / elapsed
                eta_minutes = (steps - step) / speed / 60
                best = f"{lap_seconds(best_lap_ever):.2f}s" if best_lap_ever else "--"
                print(f"step {step:>8,} | ep {episode:>5} | eps {agent.epsilon():.2f} | "
                      f"avg reward {np.mean(recent_rewards[-50:]) if recent_rewards else 0:7.2f} | "
                      f"avg gates {np.mean(recent_gates[-50:]) if recent_gates else 0:6.1f} | "
                      f"best lap {best} | {speed:,.0f} dec/s | ETA {eta_minutes:.0f} min")
                log_file.flush()

            # --- greedy test drive, keep the best brain ---
            if step % config.EVAL_EVERY == 0 and step >= config.WARMUP_STEPS:
                result = evaluate(agent)
                agent.save(latest_path)
                lap = lap_seconds(result["best_lap_steps"])
                lap_text = f"{lap:.2f}s" if lap else "--"
                marker = ""
                if result["episode_reward"] > best_eval_reward:
                    best_eval_reward = result["episode_reward"]
                    agent.save(best_path)
                    marker = "  <-- NEW BEST, saved"
                print(f"   TEST DRIVE: reward {result['episode_reward']:7.2f} | laps {result['laps']} | "
                      f"best lap {lap_text} | ended: {result['reason']}{marker}")

    except KeyboardInterrupt:
        print("\nStopped early by you.")

    agent.save(latest_path)
    log_file.close()
    plot_training(csv_path, png_path)

    minutes = (time.time() - start_time) / 60
    print(f"\nDone in {minutes:.1f} minutes, {episode} episodes.")
    print(f"Best model:   {best_path}")
    print(f"Latest model: {latest_path}")
    print(f"Log:          {csv_path}")
    print(f"Chart:        {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the AI Racer driver.")
    parser.add_argument("--steps", type=int, default=config.TRAIN_STEPS, help="total decisions to train for")
    parser.add_argument("--backend", choices=["torch", "numpy"], default=config.AGENT_BACKEND)
    parser.add_argument("--name", default="run1", help="name for the saved model and logs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args.steps, args.backend, args.name, args.seed)