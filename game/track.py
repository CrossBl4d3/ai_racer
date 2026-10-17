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

    def _gate_midpoint(self, index):
        (x1, y1), (x2, y2) = self.checkpoints[index]
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _compute_start(self):
        x0, y0 = self._gate_midpoint(0)
        x1, y1 = self._gate_midpoint(1)

        dx = x1 - x0
        dy = y1 - y0
        self.start_angle = math.degrees(math.atan2(dy, dx))

        distance = math.hypot(dx, dy)
        forward = config.CAR_LENGTH / 2 + 5
        self.start_x = x0 + dx / distance * forward
        self.start_y = y0 + dy / distance * forward

    def check_collision(self, corners):
        for edge_start, edge_end in loop_segments(corners):
            for wall_start, wall_end in self.walls:
                if segments_intersect(edge_start, edge_end, wall_start, wall_end):
                    return True
        return False

    def crossed_checkpoint(self, old_pos, new_pos, index):
        gate_start, gate_end = self.checkpoints[index]
        return segments_intersect(old_pos, new_pos, gate_start, gate_end)