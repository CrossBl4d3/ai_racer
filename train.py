"""Train the AI driver.

    python train.py --name run3                                   one track (the oval)
    python train.py --name multi1 --tracks train                  every training track
    python train.py --name multi2 --tracks train --init models/numpy2_best.npz   start from a trained brain
    python train.py --name rand1 --tracks random --backend numpy  a pool of random tracks only
    python train.py --name mix1 --tracks mix --backend numpy      training tracks + random pool
"""
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
from game.tracks import get_definition, expand_track_words
from agent.brains import load_brain


def lap_seconds(steps):
    return None if steps is None else steps / config.STEPS_PER_SECOND


def drive(agent, env, start_gate=None, nudge=None, track=None):
    """One greedy drive (no random actions).
    Returns (finished, total_seconds, best_lap_steps, reason, laps_driven)."""
    env.set_lookahead(agent.lookahead)      # give this brain the inputs it was trained with
    observation = env.reset(start_gate, nudge, track)
    env.race.best_lap_steps = None          # count only THIS drive's laps
    done = False
    while not done:
        action = agent.act(observation, explore=False)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    finished = info["reason"] == "finished"
    seconds = info["episode_steps"] / config.STEPS_PER_SECOND
    laps_driven = info["laps"] + info["checkpoints"] / env.race.track.num_gates()   # e.g. 1.5 = halfway round lap 2
    return finished, seconds, info["best_lap_steps"], info["reason"], laps_driven


def evaluate(agent, env, per_track=None, starts=None):
    """Test drives on every track in env: evenly spaced gates, or the given (track, gate, nudge) starts.
    Returns how many finished, and the pace: seconds per lap driven, counting crashed drives too
    (their time and their part-laps), so a driver can't look fast just by skipping hard races."""
    tracks = list(env.races)
    if starts is None:
        per_track = per_track or (config.EVAL_DRIVES if len(tracks) == 1 else config.EVAL_DRIVES_PER_TRACK)
        starts = [(track, i * env.races[track].track.num_gates() // per_track, None)
                  for track in tracks for i in range(per_track)]
    finished_count, total_seconds, total_laps, best_lap, failed_on = 0, 0.0, 0.0, None, []
    for track, gate, nudge in starts:
        finished, seconds, lap_steps, _, laps_driven = drive(agent, env, gate, nudge, track)
        finished_count += finished
        total_seconds += seconds
        total_laps += laps_driven
        if not finished and track not in failed_on:
            failed_on.append(track)
        if lap_steps is not None and (best_lap is None or lap_steps < best_lap):
            best_lap = lap_steps
    pace = total_seconds / max(total_laps, 1e-6)
    return {"finished": finished_count, "drives": len(starts), "pace": pace,
            "best_lap_steps": best_lap, "failed_on": failed_on}


def is_better(result, best):
    """Selection rule: finish more races first; with equal finishes, the faster pace wins."""
    if best is None:
        return True
    if result["finished"] != best["finished"]:
        return result["finished"] > best["finished"]
    return result["pace"] < best["pace"]


def plot_training(csv_path, png_path):
    import matplotlib
    matplotlib.use("Agg")                     # draw to a file, no window needed
    import matplotlib.pyplot as plt

    episodes, rewards, progress, epsilons = [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            progress.append(float(row["progress"]))
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

    axes[1].plot(episodes, progress, alpha=0.25)
    smooth = moving_average(progress)
    axes[1].plot(episodes[len(episodes) - len(smooth):], smooth)
    axes[1].set_ylabel(f"Laps driven\n({config.EPISODE_LAPS} = finished)")

    axes[2].plot(episodes, epsilons)
    axes[2].set_ylabel("Epsilon (exploration)")
    axes[2].set_xlabel("Episode")

    fig.suptitle("AI Racer training")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def train(steps, backend, name, seed, tracks, init=None, epsilon_start=None, learning_rate=None, pool=0):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    extension = ".pt" if backend == "torch" else ".npz"
    best_path = os.path.join("models", f"{name}_best{extension}")
    latest_path = os.path.join("models", f"{name}_latest{extension}")
    csv_path = os.path.join("logs", f"{name}.csv")
    png_path = os.path.join("logs", f"{name}.png")
    snaps_dir = os.path.join("models", f"{name}_snaps")         # a copy of the brain at every test
    os.makedirs(snaps_dir, exist_ok=True)

    if config.TEST_TRACK in tracks:
        print(f"WARNING: '{config.TEST_TRACK}' is the exam track. Training on it spoils the exam!")
    env = RacingEnv(random_start=config.RANDOM_START, seed=seed, tracks=tracks, random_pool=pool,
                    hard_starts=config.HARD_START_FRACTION > 0 and not pool)
    # quick test drives: the training tracks, plus 8 validation tracks when random tracks are used
    eval_env = RacingEnv(tracks=list(tracks) + (config.VALIDATION_TRACKS[:8] if pool else []))
    settings = {}
    if init:
        settings["epsilon_start"] = config.FINE_TUNE_EPSILON       # it can already drive: explore less
    if epsilon_start is not None:
        settings["epsilon_start"] = epsilon_start
    lr_end = config.LEARNING_RATE_END
    if learning_rate is not None:
        settings["learning_rate"] = learning_rate                 # e.g. small steps when fine-tuning
        lr_end = min(lr_end, learning_rate)
    agent = DQNAgent(env.obs_size, env.n_actions, backend=backend, seed=seed,
                     learning_rate_end=lr_end, lr_decay_steps=steps, **settings)
    agent.lookahead = env.lookahead                 # remember which inputs this brain uses
    if init:
        old_inputs = load_brain(agent, init)
        widened = f" (widened from {old_inputs} to {env.obs_size} inputs)" if old_inputs < env.obs_size else ""
        print(f"Starting from the trained brain in {init}{widened}")
    print(f"Inputs: {env.obs_size} (look-ahead gates {env.lookahead}) | mirror memories: "
          f"{'on' if config.MIRROR_AUGMENT else 'off'} | hard starts: {config.HARD_START_FRACTION:.0%} | "
          f"step penalty: {config.REWARD_STEP}")

    log_file = open(csv_path, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["episode", "step", "track", "reward", "checkpoints", "laps", "progress",
                  "reason", "epsilon", "loss"])

    print(f"Training '{name}' with the {backend} network for {steps:,} decisions.")
    if pool:
        mix = f"{', '.join(tracks)} + " if tracks else ""
        print(f"Tracks: {mix}a pool of {pool} random tracks (one swapped every "
              f"{config.RANDOM_POOL_REFRESH} episodes)" + (f", {config.RANDOM_SHARE:.0%} of episodes random" if tracks else ""))
    else:
        print(f"Tracks: {', '.join(tracks)}")
    print("Press Ctrl+C to stop early (progress is saved).\n")

    observation = env.reset()
    episode = 0
    recent_rewards, recent_progress, recent_finishes, recent_losses = [], [], [], []
    best_result = None
    best_lap_ever = None
    start_time = time.time()

    try:
        for step in range(1, steps + 1):
            # --- one decision ---
            action = agent.act(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.remember(observation, action, reward, next_observation, terminated)
            if config.MIRROR_AUGMENT:
                # the mirror-image experience goes straight into memory (it isn't a real
                # decision, so it doesn't count towards epsilon or target-network updates)
                agent.buffer.add(RacingEnv.mirror_observation(observation), RacingEnv.MIRROR_ACTION[action],
                                 reward, RacingEnv.mirror_observation(next_observation), terminated)
            loss = agent.learn()
            if loss is not None:
                recent_losses.append(loss)
            observation = next_observation

            # --- end of an episode ---
            if terminated or truncated:
                episode += 1
                progress = info["laps"] + info["checkpoints"] / env.race.track.num_gates()
                recent_rewards.append(info["episode_reward"])
                recent_progress.append(progress)
                recent_finishes.append(info["reason"] == "finished")
                average_loss = np.mean(recent_losses[-100:]) if recent_losses else 0.0
                log.writerow([episode, step, env.race.track.name, round(info["episode_reward"], 3),
                              info["checkpoints"], info["laps"], round(progress, 3), info["reason"],
                              round(agent.epsilon(), 4), round(average_loss, 5)])
                if len(tracks) == 1 and info["best_lap_steps"] is not None:
                    if best_lap_ever is None or info["best_lap_steps"] < best_lap_ever:
                        best_lap_ever = info["best_lap_steps"]
                observation = env.reset()

            # --- progress line ---
            if step % config.PRINT_EVERY == 0:
                elapsed = time.time() - start_time
                speed = step / elapsed
                eta_minutes = (steps - step) / speed / 60
                best = f" | best lap {lap_seconds(best_lap_ever):.2f}s" if best_lap_ever else ""
                print(f"step {step:>8,} | ep {episode:>5} | eps {agent.epsilon():.2f} | "
                      f"avg reward {np.mean(recent_rewards[-50:]) if recent_rewards else 0:7.2f} | "
                      f"avg laps {np.mean(recent_progress[-50:]) if recent_progress else 0:4.2f} | "
                      f"finished {sum(recent_finishes[-50:]):>2}/{min(len(recent_finishes), 50)}"
                      f"{best} | {speed:,.0f} dec/s | ETA {eta_minutes:.0f} min")
                log_file.flush()

            # --- greedy test drive, keep the best brain ---
            if step % config.EVAL_EVERY == 0 and step >= config.WARMUP_STEPS:
                result = evaluate(agent, eval_env)
                agent.save(latest_path)
                agent.save(os.path.join(snaps_dir, f"step_{step:07d}{extension}"))
                lap = lap_seconds(result["best_lap_steps"])
                lap_text = f" | best lap {lap:.2f}s" if lap and len(tracks) == 1 else ""
                marker = ""
                if is_better(result, best_result):
                    best_result = result
                    agent.save(best_path)
                    marker = "  <-- NEW BEST, saved"
                print(f"   TEST DRIVES: {result['finished']}/{result['drives']} finished | "
                      f"pace {result['pace']:.2f} s/lap{lap_text} | "
                      f"lr {agent.current_learning_rate():.1e}{marker}")
                if result["failed_on"]:
                    print(f"   still failing on: {', '.join(result['failed_on'])}")

    except KeyboardInterrupt:
        print("\nStopped early by you.")

    agent.save(latest_path)
    log_file.close()
    plot_training(csv_path, png_path)

    minutes = (time.time() - start_time) / 60
    print(f"\nDone in {minutes:.1f} minutes, {episode} episodes.")
    print(f"Best model:   {best_path}")
    print(f"Latest model: {latest_path}")
    print(f"Snapshots:    {snaps_dir}  (choose one with: python pick_best.py {name})")
    print(f"Log:          {csv_path}")
    print(f"Chart:        {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the AI Racer driver.")
    parser.add_argument("--steps", type=int, default=None,
                        help=f"total decisions (default {config.TRAIN_STEPS:,} for one track, "
                             f"{config.MULTI_TRAIN_STEPS:,} for several)")
    parser.add_argument("--backend", choices=["torch", "numpy"], default=config.AGENT_BACKEND)
    parser.add_argument("--name", default="run1", help="name for the saved model and logs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tracks", nargs="+", default=[config.DEFAULT_TRACK],
                        help="tracks to train on: names, 'train' (config.TRAIN_TRACKS), "
                             "'random' (a random track pool) or 'mix' (training tracks + random pool)")
    parser.add_argument("--init", default=None, help="start from this trained model instead of from scratch")
    parser.add_argument("--step-penalty", type=float, default=None,
                        help=f"reward per decision (default {config.REWARD_STEP}); more negative = hurry more")
    parser.add_argument("--epsilon", type=float, default=None, help="exploration to start from (0 to 1)")
    parser.add_argument("--lr", type=float, default=None,
                        help=f"starting learning rate (default {config.LEARNING_RATE}); smaller is gentler")
    args = parser.parse_args()

    if args.step_penalty is not None:
        config.REWARD_STEP = args.step_penalty           # the environment reads it from config

    pool = 0
    if args.tracks == ["random"]:
        tracks, pool = [], config.RANDOM_POOL_SIZE
    elif args.tracks == ["mix"]:
        tracks, pool = list(config.TRAIN_TRACKS), config.RANDOM_POOL_SIZE
    else:
        tracks = expand_track_words(args.tracks)
    for track in tracks:
        get_definition(track)                         # stops here with a clear message if a name is wrong
    if pool:
        default_steps = config.RANDOM_TRAIN_STEPS
    else:
        default_steps = config.TRAIN_STEPS if len(tracks) == 1 else config.MULTI_TRAIN_STEPS
    steps = args.steps or default_steps
    backend = args.backend
    if args.init:
        backend = "numpy" if args.init.endswith(".npz") else "torch"    # must match the saved brain
    train(steps, backend, args.name, args.seed, tracks, args.init, args.epsilon, args.lr, pool)