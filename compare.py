"""Tournament: race several trained models over many random starts.

    python compare.py models/run1_best.pt models/numpy1_best.npz
    python compare.py models/*_best.*  --races 100
    python compare.py models/numpy2_best.npz models/multi1_best.npz --tracks exam
    python compare.py models/multi1_best.npz --tracks all --races 20
"""
import argparse
import glob
import os
import numpy as np
import config
import torch
torch.set_num_threads(1)
from env.racing_env import RacingEnv
from train import drive
from watch import load_agent
from game.tracks import track_names


def tournament(model_paths, races, seed, track):
    env = RacingEnv(track_name=track)
    rng = np.random.default_rng(seed)
    names = [os.path.splitext(os.path.basename(path))[0] for path in model_paths]
    agents = [load_agent(path, env) for path in model_paths]

    # Every model gets exactly the same list of start positions: a fair contest
    starts = []
    for _ in range(races):
        gate = int(rng.integers(env.race.track.num_gates()))
        nudge = (rng.uniform(-config.START_JITTER_POS, config.START_JITTER_POS),
                 rng.uniform(-config.START_JITTER_POS, config.START_JITTER_POS),
                 rng.uniform(-config.START_JITTER_ANGLE, config.START_JITTER_ANGLE))
        starts.append((gate, nudge))

    results = {name: {"times": [], "laps": [], "best_lap": None, "wins": 0} for name in names}
    print(f"\n=== {track} ===   racing {len(names)} model(s) over {races} races...")
    for race_number, (gate, nudge) in enumerate(starts, 1):
        race_times = {}
        for name, agent in zip(names, agents):
            finished, seconds, lap_steps, _, laps_driven = drive(agent, env, gate, nudge)
            results[name]["laps"].append(laps_driven)
            if finished:
                results[name]["times"].append(seconds)
                race_times[name] = seconds
            best = results[name]["best_lap"]
            if lap_steps is not None and (best is None or lap_steps < best):
                results[name]["best_lap"] = lap_steps
        if race_times:
            fastest = min(race_times.values())
            for name, seconds in race_times.items():
                if seconds == fastest:                  # ties: everyone with the fastest time wins
                    results[name]["wins"] += 1
        if race_number % 10 == 0:
            print(f"  {race_number}/{races} races done")

    print(f"\n{'model':<20}{'races':>7}{'finished':>10}{'avg laps':>10}{'avg 3-lap':>11}{'best lap':>10}{'wins':>7}")
    print("-" * 75)
    ranking = sorted(names, key=lambda n: (-results[n]["wins"], -np.mean(results[n]["laps"]),
                                           np.mean(results[n]["times"] or [999])))
    for name in ranking:
        r = results[name]
        average = f"{np.mean(r['times']):.2f}s" if r["times"] else "--"
        best = f"{r['best_lap'] / config.STEPS_PER_SECOND:.2f}s" if r["best_lap"] else "--"
        print(f"{name:<20}{races:>7}{len(r['times']):>10}{np.mean(r['laps']):>10.2f}{average:>11}"
              f"{best:>10}{r['wins']:>7}")
    record = config.HUMAN_RECORDS.get(track)
    print(f"\nYour record lap on {track}: {f'{record:.2f}s' if record else 'not set yet'}")
    return {name: (len(results[name]["times"]), np.mean(results[name]["laps"])) for name in names}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Race trained models against each other.")
    parser.add_argument("models", nargs="+", help="model files (.pt or .npz); wildcards like models/*_best.* work")
    parser.add_argument("--races", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tracks", nargs="+", default=[config.DEFAULT_TRACK],
                        help="track names, or: 'train' (training tracks), 'exam' (the test track), 'all'")
    args = parser.parse_args()
    paths = sorted({path for pattern in args.models for path in (glob.glob(pattern) or [pattern])})

    shortcuts = {"train": config.TRAIN_TRACKS, "exam": [config.TEST_TRACK, config.TEST_TRACK + "_mirror"], "all": track_names()}
    tracks = [t for name in args.tracks for t in shortcuts.get(name, [name])]
    finishes = {}
    for track in tracks:
        for name, (count, laps) in tournament(paths, args.races, args.seed, track).items():
            finishes.setdefault(name, {})[track] = f"{count:>3} | {laps:4.2f}"

    if len(tracks) > 1:                               # summary table: one row per track
        print(f"\nSUMMARY: races finished (out of {args.races}) | average laps driven (out of {config.EPISODE_LAPS})")
        print(f"{'track':<18}" + "".join(f"{name[:14]:>16}" for name in finishes))
        for track in tracks:
            marker = "  <- exam" if track.startswith(config.TEST_TRACK) else ""
            print(f"{track:<18}" + "".join(f"{finishes[name][track]:>16}" for name in finishes) + marker)