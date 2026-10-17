import math
import numpy as np
import config
from game.race import Race


class RacingEnv:
    # Each action = (throttle, brake, steer)
    ACTIONS = [
        (True,  False,  0),   # 0: accelerate
        (True,  False, -1),   # 1: accelerate + left
        (True,  False,  1),   # 2: accelerate + right
        (False, True,   0),   # 3: brake
        (False, False,  0),   # 4: coast
    ]
    ACTION_NAMES = ["accelerate", "accel + left", "accel + right", "brake", "coast"]

    def __init__(self, render=False, wall_mode="reset", random_start=False, seed=None):
        self.race = Race(wall_mode=wall_mode, allow_reverse=False)
        self.random_start = random_start
        self.rng = np.random.default_rng(seed)
        self.n_actions = len(self.ACTIONS)
        self.obs_size = len(config.RAY_ANGLES) + 2      # rays + speed + heading

        self.render_enabled = render
        self.screen = None                              # window is created on first render
        self.closed = False

        self.ray_hit_points = []
        self.episode_reward = 0.0
        self.last_reward = 0.0

    # ------------------------------------------------------------------
    #  The two core functions
    # ------------------------------------------------------------------
    def random_start_position(self):
        """A random gate plus a small random nudge to position and angle."""
        gate = int(self.rng.integers(config.NUM_CHECKPOINTS))
        nudge = (self.rng.uniform(-config.START_JITTER_POS, config.START_JITTER_POS),
                 self.rng.uniform(-config.START_JITTER_POS, config.START_JITTER_POS),
                 self.rng.uniform(-config.START_JITTER_ANGLE, config.START_JITTER_ANGLE))
        return gate, nudge

    def reset(self, start_gate=None, nudge=None):
        if start_gate is None:
            if self.random_start:
                start_gate, nudge = self.random_start_position()
            else:
                start_gate = 0
        self.race.reset(start_gate, nudge or (0.0, 0.0, 0.0))
        if self.race.track.check_collision(self.race.car.get_corners()):
            self.race.reset(start_gate)                     # nudge hit a wall: start un-nudged
        self.steps_since_progress = 0
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.last_reward = 0.0

        observation = self._get_observation()
        if self.render_enabled:
            self.render()
        return observation

    def step(self, action):
        throttle, brake, steer = self.ACTIONS[action]
        reward = 0.0
        terminated = False
        truncated = False
        reason = ""

        for _ in range(config.FRAME_SKIP):
            events = self.race.step(throttle, brake, steer)
            self.episode_steps += 1
            self.steps_since_progress += 1

            if events["checkpoint"]:
                reward += config.REWARD_CHECKPOINT
                self.steps_since_progress = 0
            if events["lap"]:
                reward += config.REWARD_LAP

            if events["crashed"]:
                reward += config.REWARD_CRASH
                terminated, reason = True, "crashed"
                break
            if self.race.laps_completed >= config.EPISODE_LAPS:
                terminated, reason = True, "finished"
                break
            if events["timeout"]:
                truncated, reason = True, "lap time limit"
                break
            if self.steps_since_progress >= config.NO_PROGRESS_LIMIT * config.STEPS_PER_SECOND:
                truncated, reason = True, "no progress"
                break

        reward += config.REWARD_STEP

        self.episode_reward += reward
        self.last_reward = reward
        observation = self._get_observation()

        info = {
            "laps": self.race.laps_completed,
            "checkpoints": self.race.checkpoints_passed(),
            "lap_steps": self.race.lap_steps,
            "best_lap_steps": self.race.best_lap_steps,
            "episode_steps": self.episode_steps,
            "episode_reward": self.episode_reward,
            "reason": reason,
        }

        if self.render_enabled:
            self.render()
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    #  What the car senses
    # ------------------------------------------------------------------
    def _get_observation(self):
        car = self.race.car
        track = self.race.track
        origin = (car.x, car.y)

        # 1-7: distance sensors
        distances = []
        self.ray_hit_points = []
        for relative_angle in config.RAY_ANGLES:
            angle = car.angle + relative_angle
            distance = track.cast_ray(origin, angle, config.RAY_LENGTH)
            distances.append(distance / config.RAY_LENGTH)

            rad = math.radians(angle)
            self.ray_hit_points.append((car.x + distance * math.cos(rad),
                                        car.y + distance * math.sin(rad)))

        # 8: speed
        speed = car.speed / config.CAR_MAX_SPEED

        # 9: direction to the next checkpoint (-1 = far left, 0 = dead ahead, +1 = far right)
        gx, gy = track.gate_midpoint(self.race.next_checkpoint)
        target_angle = math.degrees(math.atan2(gy - car.y, gx - car.x))
        difference = (target_angle - car.angle + 180) % 360 - 180
        heading = difference / 180

        return np.array(distances + [speed, heading], dtype=np.float32)

    # ------------------------------------------------------------------
    #  Watching (optional)
    # ------------------------------------------------------------------
    def render(self):
        import pygame
        from game import race as drawing

        if self.closed:
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            pygame.display.set_caption("AI Racer - Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 30)
            self.big_font = pygame.font.Font(None, 48)
            self.small_font = pygame.font.Font(None, 22)
            self.background = drawing.build_background(self.race.track)
            self.car_sprite = drawing.CarSprite(config.CAR_SPRITE_PATH, config.CAR_HUE_SHIFT)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        car = self.race.car
        self.screen.blit(self.background, (0, 0))
        drawing.draw_rays(self.screen, (car.x, car.y), self.ray_hit_points)
        self.car_sprite.draw(self.screen, car, False)
        drawing.draw_hud(self.screen, self.race, self.font, self.big_font, self.small_font, "")

        observation = self._get_observation()
        rays_text = "Rays: " + " ".join(f"{value:.2f}" for value in observation[:-2])
        extra_text = (f"Speed: {observation[-2]:.2f}   Heading: {observation[-1]:+.2f}   "
                      f"Reward: {self.last_reward:+.2f}   Episode: {self.episode_reward:+.2f}")
        cx, cy = config.TRACK_CENTER
        drawing.draw_text_centered(self.screen, self.small_font, rays_text, config.COLOR_HUD_TEXT, (cx, cy + 115))
        drawing.draw_text_centered(self.screen, self.small_font, extra_text, config.COLOR_HUD_TEXT, (cx, cy + 140))

        pygame.display.flip()
        self.clock.tick(config.FPS // config.FRAME_SKIP)

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
        self.closed = True