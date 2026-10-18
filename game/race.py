import pygame
import config
from game.car import Car
from game.track import Track


def steps_to_seconds(steps):
    return steps / config.STEPS_PER_SECOND


class Race:
    def __init__(self, wall_mode=config.WALL_MODE, allow_reverse=config.ALLOW_REVERSE, track_name=None):
        self.track = Track(track_name)
        self.car = Car(allow_reverse=allow_reverse)
        self.wall_mode = wall_mode
        self.best_lap_steps = None
        self.last_lap_steps = None
        self.reset()

    def reset(self, start_gate=0, nudge=(0.0, 0.0, 0.0)):
        """Start a new race at any gate. nudge = (dx, dy, dangle) shifts the start slightly."""
        x, y, angle = self.track.start_pose(start_gate)
        dx, dy, dangle = nudge
        self.car.reset(x + dx, y + dy, angle + dangle)
        self.start_gate = start_gate                       # this gate is the finish line
        self.next_checkpoint = (start_gate + 1) % self.track.num_gates()
        self.laps_completed = 0
        self.lap_steps = 0
        self.crashed = False

    def checkpoints_passed(self):
        return (self.next_checkpoint - self.start_gate - 1) % self.track.num_gates()

    def step(self, throttle, brake, steer):
        events = {"crashed": False, "checkpoint": False, "lap": False, "timeout": False}

        old_x, old_y, old_angle = self.car.x, self.car.y, self.car.angle
        self.car.update(throttle, brake, steer)
        self.lap_steps += 1

        # --- Walls ---
        if self.track.check_collision(self.car.get_corners()):
            events["crashed"] = True
            self.car.x, self.car.y, self.car.angle = old_x, old_y, old_angle
            self.car.speed = 0.0
            if self.wall_mode == "reset":
                self.crashed = True
                return events

        # --- Checkpoints and laps ---
        old_pos = (old_x, old_y)
        new_pos = (self.car.x, self.car.y)
        if self.track.crossed_checkpoint(old_pos, new_pos, self.next_checkpoint):
            events["checkpoint"] = True
            if self.next_checkpoint == self.start_gate:   # back at the start: a full lap
                events["lap"] = True
                self.laps_completed += 1
                self.last_lap_steps = self.lap_steps
                if self.best_lap_steps is None or self.lap_steps < self.best_lap_steps:
                    self.best_lap_steps = self.lap_steps
                self.lap_steps = 0
            self.next_checkpoint = (self.next_checkpoint + 1) % self.track.num_gates()

        # --- Time limit ---
        if self.lap_steps >= config.LAP_TIME_LIMIT * config.STEPS_PER_SECOND:
            events["timeout"] = True

        return events


# =====================================================================
#  Drawing (only used when a human is playing or watching)
# =====================================================================

def load_image(path):
    """Load a PNG with transparency. Returns None (with a warning) if it can't be loaded."""
    try:
        return pygame.image.load(str(path)).convert_alpha()
    except (FileNotFoundError, pygame.error) as error:
        print(f"Warning: could not load {path} ({error})")
        return None


def shift_hue(surface, degrees):
    """Return a copy of the image with coloured pixels rotated around the colour wheel.
    Greys, whites and blacks (low saturation) are left alone."""
    result = surface.copy()
    width, height = result.get_size()
    for x in range(width):
        for y in range(height):
            colour = result.get_at((x, y))
            if colour.a == 0:
                continue
            hue, saturation, value, alpha = colour.hsva
            if saturation < 25:
                continue
            colour.hsva = ((hue + degrees) % 360, saturation, value, alpha)
            result.set_at((x, y), colour)
    return result


class CarSprite:
    def __init__(self, path, hue_shift):
        image = load_image(path)
        if image is None:
            self.image = None
            return

        if hue_shift:
            image = shift_hue(image, hue_shift)
        image = pygame.transform.rotate(image, -90)          # Kenney cars face up; ours face right
        image = pygame.transform.smoothscale(image, (config.CAR_LENGTH, config.CAR_WIDTH))
        self.image = image

        self.flash_image = image.copy()                       # brighter copy for the crash flash
        self.flash_image.fill((120, 120, 120, 0), special_flags=pygame.BLEND_RGB_ADD)

    def draw(self, screen, car, flashing):
        if self.image is None:                                # fallback: plain polygon car
            colour = config.COLOR_CRASH if flashing else config.COLOR_CAR
            draw_car_polygon(screen, car, colour)
            return

        source = self.flash_image if flashing else self.image
        rotated = pygame.transform.rotate(source, -car.angle)
        rect = rotated.get_rect(center=(car.x, car.y))
        screen.blit(rotated, rect)


def draw_car_polygon(screen, car, colour):
    corners = car.get_corners()
    pygame.draw.polygon(screen, colour, corners)
    pygame.draw.line(screen, config.COLOR_CHECKPOINT, corners[0], corners[1], 4)

def draw_rays(screen, origin, hit_points):
    for point in hit_points:
        pygame.draw.line(screen, config.COLOR_RAY, origin, point, 1)
        pygame.draw.circle(screen, config.COLOR_RAY_HIT, (int(point[0]), int(point[1])), 4)

def draw_hitbox(screen, car):
    pygame.draw.polygon(screen, config.COLOR_HITBOX, car.get_corners(), 1)


def build_background(track):
    """Draw everything that never moves ONCE: grass, road, kerbs, start line, decorations."""
    background = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

    # 1. Grass: tile the texture across the whole window (or plain colour if missing)
    background.fill(config.COLOR_GRASS)
    grass = load_image(config.GRASS_TILE_PATH)
    if grass is not None:
        tile_w, tile_h = grass.get_size()
        for x in range(0, config.SCREEN_WIDTH, tile_w):
            for y in range(0, config.SCREEN_HEIGHT, tile_h):
                background.blit(grass, (x, y))

    # 2. Road: a ring with a see-through hole, so the grass shows in the middle
    road = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)
    pygame.draw.polygon(road, config.COLOR_ROAD, track.outer_points)
    pygame.draw.polygon(road, (0, 0, 0, 0), track.inner_points)
    background.blit(road, (0, 0))

    # 3. Kerbs: alternate red and white along every wall segment
    for i, (wall_start, wall_end) in enumerate(track.walls):
        colour = config.COLOR_KERB_A if i % 2 == 0 else config.COLOR_KERB_B
        pygame.draw.line(background, colour, wall_start, wall_end, 6)

    # 4. Start/finish line
    start, end = track.checkpoints[0]
    pygame.draw.line(background, config.COLOR_WALL, start, end, 5)

    # 5. Decorations (purely visual: not walls, never touched by the physics)
    tree = load_image(config.TREE_PATH)
    if tree is not None:
        tree = pygame.transform.smoothscale(tree, (60, 60))
        for position in track.trees:
            background.blit(tree, tree.get_rect(center=position))
    tires = load_image(config.TIRES_PATH)
    if tires is not None:
        tires = pygame.transform.smoothscale(tires, (36, 36))
        margin = 28
        for pos in [(margin, margin), (config.SCREEN_WIDTH - margin, margin),
                    (margin, config.SCREEN_HEIGHT - margin),
                    (config.SCREEN_WIDTH - margin, config.SCREEN_HEIGHT - margin)]:
            background.blit(tires, tires.get_rect(center=pos))

    return background


def draw_checkpoints(screen, track, next_checkpoint, label_font):
    for i, (gate_start, gate_end) in enumerate(track.checkpoints):
        if i == next_checkpoint:
            colour = config.COLOR_NEXT_GATE
        elif i == 0:
            colour = config.COLOR_WALL
        else:
            colour = config.COLOR_CHECKPOINT
        pygame.draw.line(screen, colour, gate_start, gate_end, 2)
        label = label_font.render(str(i), True, config.COLOR_HUD_TEXT)
        screen.blit(label, gate_end)


def draw_text_centered(screen, font, text, colour, center):
    shadow = font.render(text, True, (0, 0, 0))
    screen.blit(shadow, shadow.get_rect(center=(center[0] + 2, center[1] + 2)))
    image = font.render(text, True, colour)
    screen.blit(image, image.get_rect(center=center))


def format_time(steps):
    if steps is None:
        return "--.--"
    return f"{steps_to_seconds(steps):.2f} s"


def draw_hud(screen, race, font, big_font, small_font, message, controls=None):
    cx, cy = race.track.hud_center
    panel = pygame.Surface((380, 165), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 110))                      # see-through dark box behind the text
    screen.blit(panel, panel.get_rect(center=(cx, cy - 30)))
    speed_kmh = int(abs(race.car.speed) / config.CAR_MAX_SPEED * 200)
    gear = " (R)" if race.car.speed < 0 else ""

    lines = [
        f"Laps: {race.laps_completed}",
        f"Time: {format_time(race.lap_steps)}",
        f"Last: {format_time(race.last_lap_steps)}    Best: {format_time(race.best_lap_steps)}",
        f"Checkpoints: {race.checkpoints_passed()} / {race.track.num_gates()}",
        f"Speed: {speed_kmh} km/h{gear}",
    ]
    top = cy - 90
    for row, text in enumerate(lines):
        draw_text_centered(screen, font, text, config.COLOR_HUD_TEXT, (cx, top + row * 30))

    if message:
        draw_text_centered(screen, big_font, message, config.COLOR_MESSAGE, (cx, top + 170))

    if controls is None:
        controls = "Arrows: drive (hold Down to reverse)   C: checkpoints   H: hitbox   R: restart   Esc: quit"
    draw_text_centered(screen, small_font, controls, config.COLOR_HUD_TEXT,
                       (config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT - 20))


# =====================================================================
#  Play loop (runs only with:  python -m game.race)
# =====================================================================

def main(track_name):
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption(f"AI Racer - {track_name}")
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 30)
    big_font = pygame.font.Font(None, 48)
    small_font = pygame.font.Font(None, 22)

    race = Race(track_name=track_name)
    background = build_background(race.track)                  # after set_mode!
    car_sprite = CarSprite(config.CAR_SPRITE_PATH, config.CAR_HUE_SHIFT)

    show_checkpoints = False
    show_hitbox = False
    message = ""
    message_timer = 0
    crash_flash = 0

    running = True
    while running:
        # --- Events (single key presses) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    race.reset()
                    message, message_timer = "", 0
                elif event.key == pygame.K_c:
                    show_checkpoints = not show_checkpoints
                elif event.key == pygame.K_h:
                    show_hitbox = not show_hitbox

        # --- Held keys -> controls ---
        keys = pygame.key.get_pressed()
        throttle = keys[pygame.K_UP]
        brake = keys[pygame.K_DOWN]
        steer = 0
        if keys[pygame.K_LEFT]:
            steer -= 1
        if keys[pygame.K_RIGHT]:
            steer += 1

        # --- One physics step ---
        events = race.step(throttle, brake, steer)

        # --- React to what happened ---
        if events["crashed"]:
            crash_flash = 10
            if race.wall_mode == "reset":
                race.reset()
                message, message_timer = "CRASHED!", 60
        if events["lap"]:
            text = f"LAP {format_time(race.last_lap_steps)}"
            if race.last_lap_steps == race.best_lap_steps:
                text += "  NEW BEST!"
            message, message_timer = text, 120
        if events["timeout"]:
            race.reset()
            message, message_timer = "TIME UP!", 90

        if message_timer > 0:
            message_timer -= 1
        else:
            message = ""
        if crash_flash > 0:
            crash_flash -= 1

        # --- Draw ---
        screen.blit(background, (0, 0))
        if show_checkpoints:
            draw_checkpoints(screen, race.track, race.next_checkpoint, small_font)
        car_sprite.draw(screen, race.car, crash_flash > 0)
        if show_hitbox:
            draw_hitbox(screen, race.car)
        draw_hud(screen, race, font, big_font, small_font, message)

        pygame.display.flip()
        clock.tick(config.FPS)

    pygame.quit()


if __name__ == "__main__":
    import argparse
    from game.tracks import track_names
    parser = argparse.ArgumentParser(description="Play AI Racer yourself.")
    parser.add_argument("--track", default=config.DEFAULT_TRACK, choices=track_names(),
                        help="which track to drive (add _mirror to drive it the other way round)")
    parser.add_argument("--list", action="store_true", help="list all tracks and exit")
    args = parser.parse_args()
    if args.list:
        print("Tracks:", ", ".join(track_names()))
    else:
        main(args.track)