import math
import config


def cross(o, a, b):
    """2D cross product of vectors OA and OB.
    > 0: B is to one side of line OA, < 0: the other side, 0: on the line."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(p1, p2, p3, p4):
    """True if segment p1-p2 crosses segment p3-p4."""
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def loop_segments(points):
    """Join a list of points into a closed loop of segments."""
    n = len(points)
    return [(points[i], points[(i + 1) % n]) for i in range(n)]


def ray_cast(origin, angle_degrees, max_length, walls):
    """Shoot a ray from origin in a direction. Return the distance to the
    nearest wall it hits, or max_length if it hits nothing."""
    px, py = origin
    rad = math.radians(angle_degrees)
    rx = math.cos(rad) * max_length          # the ray as a vector r
    ry = math.sin(rad) * max_length

    closest_t = 1.0                          # 1.0 = the full ray length
    for (qx, qy), (ex, ey) in walls:
        sx = ex - qx                         # the wall as a vector s
        sy = ey - qy
        denom = rx * sy - ry * sx            # cross(r, s)
        if denom == 0:
            continue                         # parallel: they never meet
        dx = qx - px
        dy = qy - py
        t = (dx * sy - dy * sx) / denom      # how far along the ray (0..1)
        u = (dx * ry - dy * rx) / denom      # how far along the wall (0..1)
        if 0 <= t < closest_t and 0 <= u <= 1:
            closest_t = t
    return closest_t * max_length


class Track:
    def __init__(self):
        cx, cy = config.TRACK_CENTER
        a = config.TRACK_RADIUS_X
        b = config.TRACK_RADIUS_Y
        half_road = config.ROAD_WIDTH / 2
        n = config.NUM_CHECKPOINTS

        self.outer_points = []
        self.inner_points = []
        for i in range(n):
            t = -math.pi / 2 + 2 * math.pi * i / n
            cos_t = math.cos(t)
            sin_t = math.sin(t)
            self.outer_points.append((cx + (a + half_road) * cos_t,
                                      cy - (b + half_road) * sin_t))
            self.inner_points.append((cx + (a - half_road) * cos_t,
                                      cy - (b - half_road) * sin_t))

        self.walls = loop_segments(self.outer_points) + loop_segments(self.inner_points)

        self.checkpoints = [(self.inner_points[i], self.outer_points[i]) for i in range(n)]

        self._compute_start()

    def gate_midpoint(self, index):
        (x1, y1), (x2, y2) = self.checkpoints[index]
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def start_pose(self, gate):
        """Where a car starts if its race begins at this gate: (x, y, angle)."""
        x0, y0 = self.gate_midpoint(gate)
        x1, y1 = self.gate_midpoint((gate + 1) % len(self.checkpoints))

        dx = x1 - x0
        dy = y1 - y0
        angle = math.degrees(math.atan2(dy, dx))

        distance = math.hypot(dx, dy)
        forward = config.CAR_LENGTH / 2 + 5
        return x0 + dx / distance * forward, y0 + dy / distance * forward, angle

    def _compute_start(self):
        self.start_x, self.start_y, self.start_angle = self.start_pose(0)

    def check_collision(self, corners):
        for edge_start, edge_end in loop_segments(corners):
            for wall_start, wall_end in self.walls:
                if segments_intersect(edge_start, edge_end, wall_start, wall_end):
                    return True
        return False

    def crossed_checkpoint(self, old_pos, new_pos, index):
        gate_start, gate_end = self.checkpoints[index]
        return segments_intersect(old_pos, new_pos, gate_start, gate_end)

    def cast_ray(self, origin, angle_degrees, max_length):
        return ray_cast(origin, angle_degrees, max_length, self.walls)