"""Checks the mirror trick is exactly right: drive a track and its mirror image with
mirrored actions, and the two cars must always see mirror-image observations.
    python -m sandbox.test_mirror
"""
import numpy as np
from env.racing_env import RacingEnv

rng = np.random.default_rng(0)
checks = 0
for track in ["bean", "hairpin", "wiggle"]:
    normal = RacingEnv(track_name=track)
    mirror = RacingEnv(track_name=track + "_mirror")
    for start in range(0, 60, 15):
        a = normal.reset(start_gate=start)
        b = mirror.reset(start_gate=start)
        for step in range(40):
            assert np.allclose(RacingEnv.mirror_observation(a), b, atol=1e-4), (track, start, step, a, b)
            checks += 1
            action = int(rng.integers(5))
            a, _, done_a, cut_a, _ = normal.step(action)
            b, _, done_b, cut_b, _ = mirror.step(RacingEnv.MIRROR_ACTION[action])
            assert (done_a or cut_a) == (done_b or cut_b), (track, start, step)
            if done_a or cut_a:
                break
print(f"{checks} mirror checks passed: the mirror image of an experience is a real experience")