"""Checks the fair-start rule.
    python -m sandbox.test_fair_starts
1. Every PLAIN start on every track is fair (so the fallback is always safe).
2. Of the two wiggle_mirror races every model failed, race 13 (car placed past its gate) is
   caught, and race 3 is judged fair: it is hard, but a simple autopilot wins from there.
3. How many random nudged starts get rejected (should be a small share).
"""
import numpy as np
import config
from env.racing_env import RacingEnv
from game.tracks import track_names

results = []
def check(name, passed, detail=""):
    results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}  {detail}")

# 1. plain starts
unfair_plain = []
for name in track_names():
    env = RacingEnv(track_name=name)
    for gate in range(env.race.track.num_gates()):
        if not env.is_fair_start(gate, (0.0, 0.0, 0.0)):
            unfair_plain.append((name, gate))
check("1. Every plain start is fair", not unfair_plain, f"unfair: {unfair_plain[:5]}")

# 2. the two tournament starts every model failed (compare.py, seed 123, races 3 and 13)
env = RacingEnv(track_name="wiggle_mirror")
rng = np.random.default_rng(123)
starts = []
for _ in range(20):
    gate = int(rng.integers(env.race.track.num_gates()))
    nudge = (rng.uniform(-config.START_JITTER_POS, config.START_JITTER_POS),
             rng.uniform(-config.START_JITTER_POS, config.START_JITTER_POS),
             rng.uniform(-config.START_JITTER_ANGLE, config.START_JITTER_ANGLE))
    starts.append((gate, nudge))
caught = [i for i, (gate, nudge) in enumerate(starts) if not env.is_fair_start(gate, nudge)]
check("2. Race 13 caught, race 3 kept", 13 in caught and 3 not in caught, f"rejected races: {caught}")

# 3. share of random nudged starts rejected, per track
for name in track_names():
    env = RacingEnv(track_name=name, random_start=True, seed=0)
    for _ in range(300):
        env.reset()
    share = env.unfair_starts / env.starts_checked
    print(f"     {name:<18} {share:5.1%} of random starts made fair")
    results.append(share < 0.15)

print(f"\n{sum(results)} / {len(results)} checks passed")