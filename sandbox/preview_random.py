"""Look at random tracks and measure the generator.
    python -m sandbox.preview_random               12 tracks from seed 0 -> logs/random_preview.png
    python -m sandbox.preview_random --start 100   12 tracks from seed 100
    python -m sandbox.preview_random --stats 500   also measure 500 seeds
"""
import argparse
import math
import os
import statistics
import pygame
import config
from game.track import Track
from game.random_tracks import random_definition, tightest_corner
from game import race as drawing

COLUMNS, ROWS, SCALE = 4, 3, 0.4


def preview(start):
    pygame.init()
    pygame.display.set_mode((1, 1))                   # a window is needed to load images
    font = pygame.font.Font(None, 26)
    tile_w, tile_h = int(config.SCREEN_WIDTH * SCALE), int(config.SCREEN_HEIGHT * SCALE)
    sheet = pygame.Surface((tile_w * COLUMNS, (tile_h + 28) * ROWS))
    sheet.fill((30, 30, 30))
    for index in range(COLUMNS * ROWS):
        name = f"random_{start + index}"
        track = Track(name)
        image = drawing.build_background(track)
        start_line = track.checkpoints[0]
        pygame.draw.line(image, config.COLOR_WALL, *start_line, 4)
        x, y, angle = track.start_pose(0)             # red arrow: the driving direction
        tip = (x + 45 * math.cos(math.radians(angle)), y + 45 * math.sin(math.radians(angle)))
        pygame.draw.line(image, (255, 60, 60), (x, y), tip, 6)
        pygame.draw.circle(image, (255, 60, 60), (int(tip[0]), int(tip[1])), 8)
        hx, hy = track.hud_center                     # where the HUD panel will sit
        pygame.draw.rect(image, (255, 255, 255), pygame.Rect(hx - 190, hy - 112, 380, 165), 3)
        column, row = index % COLUMNS, index // COLUMNS
        left, top = column * tile_w, row * (tile_h + 28)
        sheet.blit(pygame.transform.smoothscale(image, (tile_w, tile_h)), (left, top + 28))
        width = random_definition(start + index)[0]["width"]
        label = f"{name}  width {width}  {track.num_gates()} gates"
        sheet.blit(font.render(label, True, (255, 255, 255)), (left + 8, top + 6))
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", "random_preview.png")
    pygame.image.save(sheet, path)
    print(f"Saved {path}  (white box = where the HUD sits)")


def stats(count):
    attempts, widths, gates, corners = [], [], [], []
    for seed in range(count):
        definition, tries = random_definition(seed)
        track = Track(f"random_{seed}")
        attempts.append(tries)
        widths.append(definition["width"])
        gates.append(track.num_gates())
        corners.append(tightest_corner(track))
    def line(label, values, unit=""):
        print(f"  {label:<22} min {min(values):6.0f}{unit}   median {statistics.median(values):6.0f}{unit}"
              f"   max {max(values):6.0f}{unit}")
    print(f"\n{count} random tracks:")
    line("attempts per track", attempts)
    line("road width", widths, " px")
    line("length", gates, " gates")
    line("tightest corner", corners, " px")
    print(f"  shapes rejected: {1 - count / sum(attempts):.0%} "
          f"(for comparison: the car's turning circle is about 86 px)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview and measure random tracks.")
    parser.add_argument("--start", type=int, default=0, help="first seed to draw")
    parser.add_argument("--stats", type=int, default=0, help="also measure this many seeds")
    args = parser.parse_args()
    preview(args.start)
    if args.stats:
        stats(args.stats)