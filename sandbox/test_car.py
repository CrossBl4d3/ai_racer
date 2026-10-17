import pygame
import config
from game.car import Car

pygame.init()
screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
pygame.display.set_caption("AI Racer - Car Physics Test")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 28)

car = Car(config.SCREEN_WIDTH / 2, config.SCREEN_HEIGHT / 2, angle=0)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                car.reset(config.SCREEN_WIDTH / 2, config.SCREEN_HEIGHT / 2, 0)

    keys = pygame.key.get_pressed()
    throttle = keys[pygame.K_UP]
    brake = keys[pygame.K_DOWN]
    steer = 0
    if keys[pygame.K_LEFT]:
        steer -= 1
    if keys[pygame.K_RIGHT]:
        steer += 1

    car.update(throttle, brake, steer)

    car.x %= config.SCREEN_WIDTH
    car.y %= config.SCREEN_HEIGHT

    screen.fill(config.COLOR_GRASS)
    corners = car.get_corners()
    pygame.draw.polygon(screen, config.COLOR_CAR, corners)
    pygame.draw.line(screen, config.COLOR_CHECKPOINT, corners[0], corners[1], 4)

    info = f"Speed: {car.speed:.2f} / {config.CAR_MAX_SPEED}   Angle: {car.angle:.0f} deg"
    screen.blit(font.render(info, True, config.COLOR_HUD_TEXT), (10, 10))
    screen.blit(font.render("Arrows: drive   R: reset   Esc: quit", True, config.COLOR_HUD_TEXT), (10, 40))

    pygame.display.flip()
    clock.tick(config.FPS)

pygame.quit()