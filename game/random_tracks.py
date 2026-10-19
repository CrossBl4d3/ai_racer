"""Random tracks: a seed (any whole number) becomes a drivable track. The same seed always
gives the same track, so 'random_17' is a real, repeatable track like 'bean'."""
import math
import random
import config
from game.track import (catmull_rom_loop, resample_evenly, offset_walls, polygon_area,
                        circle_radius, check_track)

CENTRE = (500, 350)


def generate_points(rng):
    """Control points around an ellipse, each pushed in or out by a random amount."""
    rx, ry = config.RANDOM_TRACK_RADII
    n = rng.randint(*config.RANDOM_TRACK_POINTS)
    wobble = rng.uniform(*config.RANDOM_TRACK_WOBBLE)          # how wild this track is
    pushes = [rng.uniform(-wobble, wobble) for _ in range(n)]
    pushes = [(pushes[i - 1] + 3 * pushes[i] + pushes[(i + 1) % n]) / 5 for i in range(n)]   # smooth a little
    pushes = [1 + p for p in pushes]
    if rng.random() < config.RANDOM_TRACK_DENT_CHANCE:           # sometimes a deep dent
        k = rng.randrange(n)
        depth = rng.uniform(*config.RANDOM_TRACK_DENT_DEPTH)
        pushes[k] *= depth
        pushes[k - 1] *= (1 + depth) / 2                          # neighbours dip halfway,
        pushes[(k + 1) % n] *= (1 + depth) / 2                    # so the bend stays drivable
    cx, cy = CENTRE
    points = []
    for i in range(n):
        angle = 2 * math.pi * (i + rng.uniform(-0.3, 0.3)) / n
        points.append((cx + rx * pushes[i] * math.cos(angle), cy + ry * pushes[i] * math.sin(angle)))
    if rng.random() < 0.5:
        points.reverse()                                         # clockwise or anticlockwise
    return points


def build_walls(points, width):
    """The walls and gates, built exactly as Track builds them."""
    centre = resample_evenly(catmull_rom_loop(points), config.GATE_SPACING)
    left, right = offset_walls(centre, width)
    if abs(polygon_area(left)) > abs(polygon_area(right)):
        outer, inner = left, right
    else:
        outer, inner = right, left
    return inner, outer, [(inner[i], outer[i]) for i in range(len(inner))]


def hud_spot(inner, outer):
    """Where the HUD panel (380 x 165 px) overlaps the road least: the spot whose nearest wall
    point is furthest away, measured in panel half-widths and half-heights."""
    walls = inner + outer
    best, best_gap = CENTRE, -1.0
    for px in range(200, 801, 20):
        for py in range(100, 601, 20):
            gap = min(max(abs(wx - px) / 190, abs(wy - py) / 82) for wx, wy in walls)
            if gap > best_gap:
                best, best_gap = (px, py + 30), gap      # the panel is drawn 30 px above the HUD point
    return best


def random_definition(seed):
    """A track definition (like the ones in tracks.py) for this seed.
    Draws shapes until one passes every check; returns (definition, attempts)."""
    rng = random.Random(seed)
    for attempt in range(1, config.RANDOM_TRACK_MAX_ATTEMPTS + 1):
        points = generate_points(rng)
        width = round(rng.uniform(*config.RANDOM_TRACK_WIDTH))
        inner, outer, checkpoints = build_walls(points, width)
        if not check_track(inner, outer, checkpoints, width):
            hud = hud_spot(inner, outer)
            return {"type": "spline", "width": width, "points": points,
                    "hud": hud, "trees": []}, attempt
    raise ValueError(f"No drivable track found for seed {seed} in {config.RANDOM_TRACK_MAX_ATTEMPTS} attempts")


def tightest_corner(track):
    """Radius (px) of the sharpest bend along the track's centre line."""
    mids = [track.gate_midpoint(i) for i in range(track.num_gates())]
    n = len(mids)
    return min(circle_radius(mids[i - 2], mids[i], mids[(i + 2) % n]) for i in range(n))