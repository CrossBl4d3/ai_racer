"""The track library. Each track is a closed centre line (control points, listed in
driving order), a road width, and where to draw the HUD. "mirror" versions are made
automatically by flipping left-to-right, which turns every left bend into a right bend."""

TRACKS = {
    "oval": {
        "type": "ellipse",                     # the original, built exactly as before
        "hud": (500, 350),
        "trees": [(250, 290), (250, 410), (750, 290), (750, 410)],
    },
    "bean": {
        "type": "spline",
        "width": 100,
        "points": [(500, 600), (700, 595), (850, 530), (910, 390), (870, 230), (750, 140),
                   (640, 150), (580, 220), (500, 265), (420, 220), (360, 150), (250, 140),
                   (130, 230), (90, 390), (150, 530), (300, 595)],
        "hud": (500, 410),
        "q_panel": (510, 503),
        "trees": [(320, 360), (680, 360), (500, 150)],
    },
    "hairpin": {
        "type": "spline",
        "width": 90,
        "points": [(200, 440), (340, 440), (480, 450), (580, 530), (720, 605), (870, 565),
                   (935, 420), (930, 250), (860, 120), (720, 80), (600, 130), (520, 230),
                   (420, 285), (300, 285), (200, 285), (125, 300), (85, 362), (125, 425)],
        "hud": (280, 140),
        "q_panel": (250, 600),                 # where watch.py draws the AI's action bars
        "trees": [(660, 330), (820, 330), (740, 460)],
    },
    "wiggle": {
        "type": "spline",
        "width": 100,
        "points": [(170, 580), (300, 610), (450, 535), (600, 610), (740, 540), (850, 590),
                   (925, 500), (935, 350), (905, 205), (820, 130), (700, 180), (550, 105),
                   (400, 180), (260, 110), (150, 135), (80, 230), (70, 380), (95, 510)],
        "hud": (500, 335),
        "trees": [(300, 300), (700, 420)],
    },
    "technical": {
        "type": "spline",
        "width": 80,
        "points": [(420, 625), (620, 625), (750, 600), (800, 520), (750, 450), (740, 380),
                   (800, 330), (900, 280), (930, 170), (860, 90), (740, 80), (660, 140),
                   (630, 240), (560, 300), (470, 295), (420, 210), (370, 110), (250, 80),
                   (130, 110), (85, 205), (120, 290), (190, 350), (200, 430), (150, 500),
                   (140, 580), (220, 635), (310, 640)],
        "hud": (475, 445),
        "q_panel": (485, 539),
        "trees": [(330, 230), (820, 190), (850, 620)],
    },
}


def track_names(include_mirrors=True):
    names = list(TRACKS)
    if include_mirrors:
        names += [f"{name}_mirror" for name in TRACKS]
    return names


def random_seed(name):
    """'random_17' -> 17. Any other name -> None."""
    if name.startswith("random_") and name[len("random_"):].isdigit():
        return int(name[len("random_"):])
    return None


def get_definition(name):
    """Returns (definition, mirrored) for a name like 'bean', 'bean_mirror' or 'random_17'."""
    mirrored = name.endswith("_mirror")
    base = name[:-len("_mirror")] if mirrored else name
    seed = random_seed(base)
    if seed is not None:
        from game.random_tracks import random_definition     # imported here to avoid a loop
        return random_definition(seed)[0], mirrored
    if base not in TRACKS:
        raise ValueError(f"Unknown track '{name}'. Available: {', '.join(track_names())}, "
                         f"or random_<number> (e.g. random_17)")
    return TRACKS[base], mirrored


def track_name(text):
    """For argparse: accept any valid track name, including random_<number>."""
    import argparse
    try:
        get_definition(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error))
    return text