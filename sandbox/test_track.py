import pygame
import config
from game.car import Car
from game.track import Track

pygame.init()
screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
pygame.display.set_caption("AI Racer - Track Test")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 28)
small_font = pygame.font.Font(None, 20)

track = Track()
car = Car(track.start_x, track.start_y, track.start_angle)
next_checkpoint = 1
show_checkpoints = True

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                car.reset(track.start_x, track.start_y, track.start_angle)
                next_checkpoint = 1
            elif event.key == pygame.K_c:
                show_checkpoints = not show_checkpoints

    keys = pygame.key.get_pressed()
    steer = 0
    if keys[pygame.K_LEFT]:
        steer -= 1
    if keys[pygame.K_RIGHT]:
        steer += 1

    old_x, old_y, old_angle = car.x, car.y, car.angle
    old_pos = (old_x, old_y)
    car.update(keys[pygame.K_UP], keys[pygame.K_DOWN], steer)

    crashed = track.check_collision(car.get_corners())
    if crashed:
        car.x, car.y, car.angle = old_x, old_y, old_angle   # undo the move
        car.speed = 0.0                                      # stop dead

    new_pos = (car.x, car.y)

    if track.crossed_checkpoint(old_pos, new_pos, next_checkpoint):
        next_checkpoint = (next_checkpoint + 1) % config.NUM_CHECKPOINTS

    # --- Draw the track ---
    screen.fill(config.COLOR_GRASS)
    pygame.draw.polygon(screen, config.COLOR_ROAD, track.outer_points)
    pygame.draw.polygon(screen, config.COLOR_GRASS, track.inner_points)
    for wall_start, wall_end in track.walls:
        pygame.draw.line(screen, config.COLOR_WALL, wall_start, wall_end, 3)

    if show_checkpoints:
        for i, (gate_start, gate_end) in enumerate(track.checkpoints):
            if i == next_checkpoint:
                colour = config.COLOR_NEXT_GATE  # the gate you need next
            elif i == 0:
                colour = config.COLOR_WALL      # start/finish line
            else:
                colour = config.COLOR_CHECKPOINT
            pygame.draw.line(screen, colour, gate_start, gate_end, 2)
            label = small_font.render(str(i), True, config.COLOR_HUD_TEXT)
            screen.blit(label, gate_end)

    # --- Draw the car ---
    car_colour = config.COLOR_CRASH if crashed else config.COLOR_CAR
    corners = car.get_corners()
    pygame.draw.polygon(screen, car_colour, corners)
    pygame.draw.line(screen, config.COLOR_CHECKPOINT, corners[0], corners[1], 4)

    # --- Info text (in the middle of the oval) ---
    status = "CRASHED!" if crashed else "OK"
    lines = [
        f"Next checkpoint: {next_checkpoint}",
        f"Status: {status}",
        "C: checkpoints   R: reset   Esc: quit",
    ]
    for row, text in enumerate(lines):
        image = font.render(text, True, config.COLOR_HUD_TEXT)
        screen.blit(image, (370, 300 + row * 30))

    pygame.display.flip()
    clock.tick(config.FPS)

pygame.quit()