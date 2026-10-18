import math
import numpy as np
import config
from game.race import Race
from game.track import cross


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
    MIRROR_ACTION = [0, 2, 1, 3, 4]      # in a mirror, left and right swap; everything else stays

    def __init__(self, render=False, wall_mode="reset", random_start=False, seed=None,
                 track_name=None, tracks=None, lookahead=None, hard_starts=False):
        # One Race per track, built once. With several tracks, each episode picks one at random.
        names = tracks or [track_name or config.DEFAULT_TRACK]
        self.races = {name: Race(wall_mode=wall_mode, allow_reverse=False, track_name=name) for name in names}
        self.race = self.races[names[0]]
        self.random_start = random_start
        self.rng = np.random.default_rng(seed)
        self.n_actions = len(self.ACTIONS)
        self.set_lookahead(config.LOOKAHEAD_GATES if lookahead is None else lookahead)

        # Where the car crashes, per track and gate, so training can practise those spots
        self.hard_starts = hard_starts
        self.crash_counts = {name: np.zeros(race.track.num_gates()) for name, race in self.races.items()}
        
                # How many nudged starts were checked, and how many had to fall back to the plain start
        self.starts_checked = 0
        self.unfair_starts = 0
        self.render_enabled = render
        self.screen = None                              # window is created on first render
        self.background_track = None                    # which track the background shows
        self.closed = False

        self.ray_hit_points = []
        self.episode_reward = 0.0
        self.last_reward = 0.0

    # ------------------------------------------------------------------
    #  The two core functions
    # ------------------------------------------------------------------
    def set_lookahead(self, gates):
        """Which gates ahead the car is told about. Older models were trained with none."""
        self.lookahead = list(gates)
        self.obs_size = len(config.RAY_ANGLES) + 2 + len(self.lookahead)   # rays + speed + headings

    @staticmethod
    def mirror_observation(observation):
        """What the car would see on the mirror-image track: rays swap sides, directions flip."""
        n_rays = len(config.RAY_ANGLES)
        mirrored = observation.copy()
        mirrored[:n_rays] = observation[:n_rays][::-1]        # rightmost ray becomes leftmost
        mirrored[n_rays + 1:] = -observation[n_rays + 1:]      # left of the car becomes right
        return mirrored                                        # (speed, at n_rays, stays the same)

    def random_start_position(self):
        """A random gate plus a small random nudge to position and angle."""
        n = self.race.track.num_gates()
        counts = self.crash_counts[self.race.track.name]
        if self.hard_starts and counts.sum() > 0 and self.rng.random() < config.HARD_START_FRACTION:
            crash_gate = int(self.rng.choice(n, p=counts / counts.sum()))   # busy crash spots more often
            back = int(self.rng.integers(*config.HARD_START_BACK))
            gate = (crash_gate - back) % n                                  # a run-up before the corner
        else:
            gate = int(self.rng.integers(n))
        nudge = (self.rng.uniform(-config.START_JITTER_POS, config.START_JITTER_POS),
                 self.rng.uniform(-config.START_JITTER_POS, config.START_JITTER_POS),
                 self.rng.uniform(-config.START_JITTER_ANGLE, config.START_JITTER_ANGLE))
        return gate, nudge


    def is_fair_start(self, gate, nudge):
        """Could a good driver win from this start? It must (1) not touch a wall, (2) not be past
        the first gate it has to cross, and (3) get through the next gates with SOME steering."""
        race, track = self.race, self.race.track
        race.reset(gate, nudge)
        car = race.car
        if track.check_collision(car.get_corners()):
            return False
        a, b = track.checkpoints[race.next_checkpoint]
        car_side = cross(a, b, (car.x, car.y)) > 0
        gate_side = cross(a, b, track.gate_midpoint(gate)) > 0
        if car_side != gate_side:
            return False                                    # already past it: could never score it

        probe_steps = int(config.FAIR_START_PROBE_SECONDS * config.STEPS_PER_SECOND)
        for steer in (-1, 0, 1):                            # dry runs: hold left, straight, right
            race.reset(gate, nudge)
            for _ in range(probe_steps):
                events = race.step(True, False, steer)
                if events["crashed"]:
                    break
                if race.checkpoints_passed() >= config.FAIR_START_GATES:
                    return True                             # a winning line exists
        return False

    
    def reset(self, start_gate=None, nudge=None, track=None):
        if track is not None:
            self.race = self.races[track]                   # a specific track was asked for
        elif len(self.races) > 1:
            names = list(self.races)
            self.race = self.races[names[int(self.rng.integers(len(names)))]]
        if start_gate is None:
            if self.random_start:
                start_gate, nudge = self.random_start_position()
            else:
                start_gate = 0
        nudge = tuple(nudge) if nudge else (0.0, 0.0, 0.0)
        if nudge != (0.0, 0.0, 0.0):
            self.starts_checked += 1
            if not self.is_fair_start(start_gate, nudge):
                nudge = (0.0, 0.0, 0.0)                     # unfair nudge: use the plain start
                self.unfair_starts += 1
        self.race.reset(start_gate, nudge)                  # (also clears the dry runs)
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
                for counts in self.crash_counts.values():
                    counts *= config.CRASH_MEMORY                           # old crashes fade
                self.crash_counts[self.race.track.name][self.race.next_checkpoint] += 1
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

        # 9+: direction to the next checkpoint, then to gates further ahead
        #     (-1 = far left, 0 = dead ahead, +1 = far right)
        headings = []
        for ahead in [0] + self.lookahead:
            gate = (self.race.next_checkpoint + ahead) % track.num_gates()
            gx, gy = track.gate_midpoint(gate)
            target_angle = math.degrees(math.atan2(gy - car.y, gx - car.x))
            difference = (target_angle - car.angle + 180) % 360 - 180
            headings.append(difference / 180)

        return np.array(distances + [speed] + headings, dtype=np.float32)

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
            self.car_sprite = drawing.CarSprite(config.CAR_SPRITE_PATH, config.CAR_HUE_SHIFT)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        if self.background_track is not self.race.track:
            self.background = drawing.build_background(self.race.track)    # the track changed
            self.background_track = self.race.track

        car = self.race.car
        self.screen.blit(self.background, (0, 0))
        drawing.draw_rays(self.screen, (car.x, car.y), self.ray_hit_points)
        self.car_sprite.draw(self.screen, car, False)
        drawing.draw_hud(self.screen, self.race, self.font, self.big_font, self.small_font, "")

        observation = self._get_observation()
        n_rays = len(config.RAY_ANGLES)
        rays_text = "Rays: " + " ".join(f"{value:.2f}" for value in observation[:n_rays])
        headings = " ".join(f"{value:+.2f}" for value in observation[n_rays + 1:])
        extra_text = (f"Speed: {observation[n_rays]:.2f}   Headings: {headings}   "
                      f"Reward: {self.last_reward:+.2f}   Episode: {self.episode_reward:+.2f}")
        cx, cy = self.race.track.hud_center
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