"""Draw every track into one image so you can check the shapes.
    python -m sandbox.preview_tracks              -> logs/tracks_preview.png
"""
import os
import pygame
import config
from game.track import Track
from game.tracks import TRACKS
from game import race as drawing

COLUMNS = 3
SCALE = 0.5

pygame.init()
screen = pygame.display.set_mode((1, 1))              # a window is needed to load images
font = pygame.font.Font(None, 28)
small = pygame.font.Font(None, 16)

tile_w, tile_h = int(config.SCREEN_WIDTH * SCALE), int(config.SCREEN_HEIGHT * SCALE)
rows = (len(TRACKS) + COLUMNS - 1) // COLUMNS
sheet = pygame.Surface((tile_w * COLUMNS, (tile_h + 30) * rows))
sheet.fill((30, 30, 30))

for index, name in enumerate(TRACKS):
    track = Track(name)
    image = drawing.build_background(track)
    for i, (start, end) in enumerate(track.checkpoints):
        colour = config.COLOR_WALL if i == 0 else config.COLOR_CHECKPOINT
        pygame.draw.line(image, colour, start, end, 3 if i == 0 else 1)
        if i % 5 == 0:
            image.blit(small.render(str(i), True, config.COLOR_HUD_TEXT), end)
    x, y, angle = track.start_pose(0)                  # an arrow showing the driving direction
    import math
    tip = (x + 40 * math.cos(math.radians(angle)), y + 40 * math.sin(math.radians(angle)))
    pygame.draw.line(image, (255, 60, 60), (x, y), tip, 5)
    pygame.draw.circle(image, (255, 60, 60), (int(tip[0]), int(tip[1])), 7)

    column, row = index % COLUMNS, index // COLUMNS
    left, top = column * tile_w, row * (tile_h + 30)
    sheet.blit(pygame.transform.smoothscale(image, (tile_w, tile_h)), (left, top + 30))
    label = f"{name}   ({track.num_gates()} gates)"
    sheet.blit(font.render(label, True, (255, 255, 255)), (left + 10, top + 6))

os.makedirs("logs", exist_ok=True)
pygame.image.save(sheet, os.path.join("logs", "tracks_preview.png"))
print("Saved logs/tracks_preview.png")