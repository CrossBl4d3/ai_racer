"""Choose the best brain from a training run's snapshots, using a big, fair test.

    python pick_best.py multi3                tests every file in models/multi3_snaps/
    python pick_best.py speed1 --races 5      fewer races per track (faster)
    python pick_best.py rand1 --tracks val
Each snapshot drives the same fair random starts on every TRAINING track (never the exam).
Rule: among the snapshots within PICK_FINISH_TOLERANCE races of the most finishes, the
fastest pace wins. The winner is copied to models/<name>_pick.<ext>, and a chart of finish
rate against pace is saved to logs/<name>_pick.png.
"""
import argparse
import glob
import os
import shutil
import numpy as np
import config
import torch
torch.set_num_threads(1)
from env.racing_env import RacingEnv
from train import evaluate
from watch import load_agent
from game.tracks import expand_track_words


def fair_starts(env, races, seed):
    """The same random (track, gate, nudge) starts for every snapshot; reset() makes them fair."""
    rng = np.random.default_rng(seed)
    starts = []
    for track in env.races:
        for _ in range(races):
            gate = int(rng.integers(env.races[track].track.num_gates()))
            nudge = (rng.uniform(-config.START_JITTER_POS, config.START_JITTER_POS),
                     rng.uniform(-config.START_JITTER_POS, config.START_JITTER_POS),
                     rng.uniform(-config.START_JITTER_ANGLE, config.START_JITTER_ANGLE))
            starts.append((track, gate, nudge))
    return starts


def pareto_front(points):
    """Snapshots that no other snapshot beats on BOTH finishes and pace."""
    return [p for p in points
            if not any(q["finished"] >= p["finished"] and q["pace"] <= p["pace"] and q is not p
                       and (q["finished"] > p["finished"] or q["pace"] < p["pace"]) for q in points)]


def plot(points, chosen, total, png_path, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [p["pace"] for p in points]
    ys = [100 * p["finished"] / total for p in points]
    ax.scatter(xs, ys, color="0.6", zorder=2)
    for p in points:
        ax.annotate(f"{p['step'] // 1000}k", (p["pace"], 100 * p["finished"] / total),
                    textcoords="offset points", xytext=(5, 4), fontsize=8, color="0.4")
    front = sorted(pareto_front(points), key=lambda p: p["pace"])
    ax.plot([p["pace"] for p in front], [100 * p["finished"] / total for p in front],
            color="tab:blue", zorder=1, label="Pareto front (best trade-offs)")
    ax.scatter([chosen["pace"]], [100 * chosen["finished"] / total], s=140, facecolors="none",
               edgecolors="tab:blue", linewidths=2, zorder=3, label=f"picked: step {chosen['step']:,}")
    ax.set_xlabel("Pace: seconds per lap driven (lower = faster)")
    ax.set_ylabel(f"Races finished (%, of {total})")
    ax.set_title(f"{name}: every snapshot, finish rate against pace")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def pick(name, races, seed, tracks, every):
    paths = sorted(glob.glob(os.path.join("models", f"{name}_snaps", "step_*")))[every - 1::every]
    if not paths:
        raise SystemExit(f"No snapshots in models/{name}_snaps/. Train with the new train.py first.")
    exam = set(expand_track_words(["exam", "test_random"]))
    if exam & set(tracks):
        raise SystemExit("Choosing on exam tracks would leak the exam into the choice. Use train or val.")
    env = RacingEnv(tracks=tracks)
    starts = fair_starts(env, races, seed)
    print(f"Testing {len(paths)} snapshots of '{name}' on {len(starts)} fair starts "
          f"({races} per track on {len(tracks)} tracks, never the exam)...\n")

    points = []
    for path in paths:
        agent = load_agent(path, env)
        result = evaluate(agent, env, starts=starts)
        step = int(os.path.splitext(os.path.basename(path))[0].split("_")[1])
        points.append({"step": step, "path": path, **result})
        print(f"   step {step:>8,}: {result['finished']:>3}/{result['drives']} finished | "
              f"pace {result['pace']:.2f} s/lap")

    most = max(p["finished"] for p in points)
    candidates = [p for p in points if p["finished"] >= most - config.PICK_FINISH_TOLERANCE]
    chosen = min(candidates, key=lambda p: p["pace"])

    extension = os.path.splitext(chosen["path"])[1]
    out_path = os.path.join("models", f"{name}_pick{extension}")
    shutil.copyfile(chosen["path"], out_path)
    os.makedirs("logs", exist_ok=True)
    png_path = os.path.join("logs", f"{name}_pick.png")
    plot(points, chosen, len(starts), png_path, name)

    print(f"\nMost finishes: {most}/{len(starts)}. Within {config.PICK_FINISH_TOLERANCE} of that, "
          f"the fastest is step {chosen['step']:,}: {chosen['finished']}/{len(starts)} finished, "
          f"pace {chosen['pace']:.2f} s/lap.")
    print(f"Saved as:  {out_path}")
    print(f"Chart:     {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose the best snapshot of a training run.")
    parser.add_argument("name", help="the training run's name, e.g. multi3")
    parser.add_argument("--races", type=int, default=10, help="races per training track for each snapshot")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tracks", nargs="+", default=["train"],
                        help="tracks to choose on: 'train', 'val', or track names (never the exam)")
    parser.add_argument("--every", type=int, default=1, help="test every Nth snapshot only (faster)")
    args = parser.parse_args()
    pick(args.name, args.races, args.seed, expand_track_words(args.tracks), args.every)