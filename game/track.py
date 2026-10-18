import math
import config
from game.tracks import get_definition


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


def polygon_area(points):
    """Signed area (shoelace formula). Its size tells us which wall is the outer one."""
    return 0.5 * sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in loop_segments(points))


# ----------------------------------------------------------------------
#  Building a centre line from control points
# ----------------------------------------------------------------------

def catmull_rom_loop(points, samples_per_segment=40):
    """A smooth closed curve that passes through every control point."""
    n = len(points)
    curve = []
    for i in range(n):
        p0, p1, p2, p3 = points[i - 1], points[i], points[(i + 1) % n], points[(i + 2) % n]
        for step in range(samples_per_segment):
            t = step / samples_per_segment
            t2, t3 = t * t, t * t * t
            x = 0.5 * (2 * p1[0] + (-p0[0] + p2[0]) * t
                       + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                       + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * (2 * p1[1] + (-p0[1] + p2[1]) * t
                       + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                       + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            curve.append((x, y))
    return curve


def resample_evenly(curve, spacing):
    """Pick points along a closed curve that are (almost exactly) `spacing` apart."""
    lengths = [0.0]                                  # running distance along the curve
    for (x1, y1), (x2, y2) in loop_segments(curve):
        lengths.append(lengths[-1] + math.hypot(x2 - x1, y2 - y1))
    total = lengths[-1]
    count = max(8, round(total / spacing))
    step = total / count

    result, j = [], 0
    closed = curve + [curve[0]]
    for k in range(count):
        target = k * step
        while lengths[j + 1] < target:
            j += 1
        fraction = (target - lengths[j]) / (lengths[j + 1] - lengths[j])
        (x1, y1), (x2, y2) = closed[j], closed[j + 1]
        result.append((x1 + (x2 - x1) * fraction, y1 + (y2 - y1) * fraction))
    return result


def offset_walls(centre, width):
    """Push each centre point sideways by half the width, both ways (along the normal)."""
    n = len(centre)
    left, right = [], []
    for i in range(n):
        (px, py), (nx_, ny_) = centre[i - 1], centre[(i + 1) % n]
        tx, ty = nx_ - px, ny_ - py                  # tangent: direction of the road
        length = math.hypot(tx, ty)
        normal_x, normal_y = -ty / length, tx / length   # perpendicular to the tangent
        cx, cy = centre[i]
        half = width / 2
        left.append((cx + normal_x * half, cy + normal_y * half))
        right.append((cx - normal_x * half, cy - normal_y * half))
    return left, right


def circle_radius(a, b, c):
    """Radius of the circle through three points (how sharply the road bends there)."""
    area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))   # 2 x triangle area
    if area2 < 1e-9:
        return float("inf")                                                      # straight
    return math.dist(a, b) * math.dist(b, c) * math.dist(c, a) / (2 * area2)


def car_turning_radius():
    """The car's turning circle. Turning speed grows with speed, so this is the SAME at every speed."""
    degrees_per_pixel = config.CAR_TURN_RATE / config.CAR_MAX_SPEED
    return 1 / math.radians(degrees_per_pixel)


def check_track(inner, outer, checkpoints, width):
    """Refuse broken tracks: walls or gates that cross each other, points off screen,
    or corners too tight for the car's turning circle."""
    problems = []
    mids = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in checkpoints]
    n = len(mids)
    usable = width / 2 - config.CAR_WIDTH / 2         # how far the car can swing wide
    for i in range(n):
        radius = circle_radius(mids[i - 2], mids[i], mids[(i + 2) % n])
        if radius + usable < 1.1 * car_turning_radius():
            problems.append(f"corner at gate {i} is too tight for the car "
                            f"(bend radius {radius:.0f}px, the car needs about {car_turning_radius() - usable:.0f}px)")
            break
    for name, wall in (("inner", inner), ("outer", outer)):
        segments = loop_segments(wall)
        n = len(segments)
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue                          # neighbours in the loop
                if segments_intersect(*segments[i], *segments[j]):
                    problems.append(f"{name} wall crosses itself near gate {i} (a corner is too tight)")
                    break
    for a in loop_segments(inner):
        for b in loop_segments(outer):
            if segments_intersect(*a, *b):
                problems.append("inner and outer walls cross (the road overlaps itself)")
                break
    n = len(checkpoints)
    for i in range(n):
        for j in range(i + 2, n):
            if not (i == 0 and j == n - 1) and segments_intersect(*checkpoints[i], *checkpoints[j]):
                problems.append(f"gates {i} and {j} cross (a corner is too tight)")
    for x, y in inner + outer:
        if not (0 <= x <= config.SCREEN_WIDTH and 0 <= y <= config.SCREEN_HEIGHT):
            problems.append("part of the track is off screen")
            break
    return sorted(set(problems))


class Track:
    def __init__(self, name=None):
        self.name = name or config.DEFAULT_TRACK
        definition, mirrored = get_definition(self.name)

        if definition["type"] == "ellipse":
            inner, outer = self._ellipse_walls()
        else:
            centre = resample_evenly(catmull_rom_loop(definition["points"]), config.GATE_SPACING)
            left, right = offset_walls(centre, definition["width"])
            # whichever wall encloses the bigger area is the outer one
            if abs(polygon_area(left)) > abs(polygon_area(right)):
                outer, inner = left, right
            else:
                outer, inner = right, left

        hud = definition["hud"]
        q_panel = definition.get("q_panel", (hud[0] + 10, hud[1] + 100))    # default: under the HUD
        trees = definition.get("trees", [])
        if mirrored:                                   # flip left-to-right
            flip = lambda p: (config.SCREEN_WIDTH - p[0], p[1])
            inner, outer = [flip(p) for p in inner], [flip(p) for p in outer]
            hud, q_panel, trees = flip(hud), flip(q_panel), [flip(p) for p in trees]

        self.inner_points, self.outer_points = inner, outer
        self.hud_center, self.q_panel_center, self.trees = hud, q_panel, trees
        self.walls = loop_segments(self.outer_points) + loop_segments(self.inner_points)
        self.wall_boxes = [(min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1]))
                           for a, b in self.walls]     # each wall's bounding box, for a quick test
        self.checkpoints = [(inner[i], outer[i]) for i in range(len(inner))]

        width = definition.get("width", config.ROAD_WIDTH)
        problems = check_track(inner, outer, self.checkpoints, width)
        if problems:
            raise ValueError(f"Track '{self.name}' is broken:\n  " + "\n  ".join(problems))

        self._compute_start()

    def _ellipse_walls(self):
        cx, cy = config.TRACK_CENTER
        a = config.TRACK_RADIUS_X
        b = config.TRACK_RADIUS_Y
        half_road = config.ROAD_WIDTH / 2
        n = config.NUM_CHECKPOINTS
        outer, inner = [], []
        for i in range(n):
            t = -math.pi / 2 + 2 * math.pi * i / n
            cos_t, sin_t = math.cos(t), math.sin(t)
            outer.append((cx + (a + half_road) * cos_t, cy - (b + half_road) * sin_t))
            inner.append((cx + (a - half_road) * cos_t, cy - (b - half_road) * sin_t))
        return inner, outer

    def num_gates(self):
        return len(self.checkpoints)

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
        xs = [x for x, y in corners]
        ys = [y for x, y in corners]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        edges = loop_segments(corners)
        for (wall_start, wall_end), (x1, x2, y1, y2) in zip(self.walls, self.wall_boxes):
            if x2 < left or x1 > right or y2 < top or y1 > bottom:
                continue                        # this wall is nowhere near the car: skip it
            for edge_start, edge_end in edges:
                if segments_intersect(edge_start, edge_end, wall_start, wall_end):
                    return True
        return False

    def crossed_checkpoint(self, old_pos, new_pos, index):
        gate_start, gate_end = self.checkpoints[index]
        return segments_intersect(old_pos, new_pos, gate_start, gate_end)

    def cast_ray(self, origin, angle_degrees, max_length):
        return ray_cast(origin, angle_degrees, max_length, self.walls)