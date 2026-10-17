import pygame

# --- 1. Start Pygame ---
pygame.init()

# --- 2. Create the window ---
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Racer - Pygame Test")

# --- 3. Create the clock ---
clock = pygame.time.Clock()
FPS = 60

# --- 4. Colours and the "car" ---
BACKGROUND_COLOR = (30, 30, 30)   # dark grey
CAR_COLOR = (220, 50, 50)         # red

car_width = 40
car_height = 20
car_x = WIDTH // 2 - car_width // 2
car_y = HEIGHT // 2 - car_height // 2
car_speed = 10                      # pixels per frame

# --- 5. The game loop ---
running = True
while running:

    # Handle events (must do this every frame)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- 6. Read keyboard and move ---
    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        car_x -= car_speed
    if keys[pygame.K_RIGHT]:
        car_x += car_speed
    if keys[pygame.K_UP]:
        car_y -= car_speed         # up = smaller y
    if keys[pygame.K_DOWN]:
        car_y += car_speed

    # Keep the car inside the window
    car_x = max(0, min(car_x, WIDTH - car_width))
    car_y = max(0, min(car_y, HEIGHT - car_height))

    # --- 7. Draw ---
    screen.fill(BACKGROUND_COLOR)
    pygame.draw.rect(screen, CAR_COLOR, (car_x, car_y, car_width, car_height))

    # --- 8. Show the frame and keep 60 FPS ---
    pygame.display.flip()
    clock.tick(FPS)

# --- 9. Shut down cleanly ---
pygame.quit()