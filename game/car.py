import math
import config


class Car:
    def __init__(self, x=0.0, y=0.0, angle=0.0, allow_reverse=False):
        self.length = config.CAR_LENGTH
        self.width = config.CAR_WIDTH
        self.allow_reverse = allow_reverse
        self.reset(x, y, angle)

    def reset(self, x, y, angle):
        self.x = float(x)
        self.y = float(y)
        self.angle = float(angle)      # degrees, 0 = facing right
        self.speed = 0.0               # negative = reversing

    def update(self, throttle, brake, steer):
        # 1. Speed
        if throttle:
            self.speed += config.CAR_ACCELERATION
        if brake:
            if self.speed > 0:
                self.speed = max(0.0, self.speed - config.CAR_BRAKE)   # brake down to a stop
            elif self.allow_reverse:
                self.speed -= config.CAR_REVERSE_ACCEL                 # already stopped: reverse
        if not throttle and not brake:
            if self.speed > 0:
                self.speed = max(0.0, self.speed - config.CAR_FRICTION)
            elif self.speed < 0:
                self.speed = min(0.0, self.speed + config.CAR_FRICTION)

        min_speed = -config.CAR_MAX_REVERSE_SPEED if self.allow_reverse else 0.0
        self.speed = max(min_speed, min(self.speed, config.CAR_MAX_SPEED))

        # 2. Turning (only possible while moving; flips when going backwards)
        speed_factor = self.speed / config.CAR_MAX_SPEED
        self.angle += steer * config.CAR_TURN_RATE * speed_factor
        self.angle %= 360

        # 3. Moving (forwards, or backwards if speed is negative)
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)

    def get_corners(self):
        half_l = self.length / 2
        half_w = self.width / 2

        # Corner offsets as if the car faced right (angle 0)
        offsets = [
            ( half_l, -half_w),   # front-left
            ( half_l,  half_w),   # front-right
            (-half_l,  half_w),   # back-right
            (-half_l, -half_w),   # back-left
        ]

        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        corners = []
        for lx, ly in offsets:
            wx = self.x + lx * cos_a - ly * sin_a
            wy = self.y + lx * sin_a + ly * cos_a
            corners.append((wx, wy))
        return corners

    def get_front(self):
        rad = math.radians(self.angle)
        return (self.x + (self.length / 2) * math.cos(rad),
                self.y + (self.length / 2) * math.sin(rad))