import math
import random
import statistics
import time
import heapq
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np

# 0=free, 1=obstacle

def inflate_grid(grid, radius):
    h = len(grid)
    w = len(grid[0])
    inflated = [[0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if grid[y][x] == 1:
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if 0 <= y + dy < h and 0 <= x + dx < w:
                            if dx * dx + dy * dy <= radius * radius:
                                inflated[y + dy][x + dx] = 1
    return inflated


def print_grid(grid, path=None, start=None, goal=None):
    chars = {0: ' ', 1: '#'}
    h = len(grid)
    w = len(grid[0])
    grid_chars = [[chars[val] for val in row] for row in grid]
    if path:
        for x, y in path:
            if (x, y) != start and (x, y) != goal:
                grid_chars[y][x] = '.'
    if start:
        sx, sy = start
        grid_chars[sy][sx] = 'S'
    if goal:
        gx, gy = goal
        grid_chars[gy][gx] = 'G'
    print('\n'.join(''.join(row) for row in grid_chars))


class GridBenchmark:
    def __init__(self, grid, start, goal, robot):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.robot = robot
        self.inflated = inflate_grid(grid, robot.inflation_radius)

    def is_free(self, x, y):
        return 0 <= x < len(self.grid[0]) and 0 <= y < len(self.grid) and self.inflated[y][x] == 0

    def collision_line(self, p0, p1):
        x0, y0 = p0
        x1, y1 = p1
        steps = int(max(abs(x1 - x0), abs(y1 - y0)) * 2) + 1
        for i in range(steps + 1):
            t = i / max(1, steps)
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            ix, iy = int(round(x)), int(round(y))
            if not self.is_free(ix, iy):
                return True
        return False

    def path_clearance(self, path):
        if not path:
            return 0.0
        min_clear = float('inf')
        for x, y in path:
            clear = self.local_clearance(x, y)
            min_clear = min(min_clear, clear)
        return min_clear

    def local_clearance(self, x, y):
        # compute clearance w.r.t. original grid (not inflated)
        max_r = min(len(self.grid), len(self.grid[0]))
        distance = 0
        while True:
            distance += 1
            for dx in range(-distance, distance + 1):
                for dy in range(-distance, distance + 1):
                    tx, ty = x + dx, y + dy
                    if 0 <= tx < len(self.grid[0]) and 0 <= ty < len(self.grid):
                        if self.grid[ty][tx] == 1:
                            return distance - 1
                    else:
                        return distance - 1
            if distance > max_r:
                break
        return distance

    def evaluate_path(self, path, planner_name, elapsed):
        success = self.validate_path(path)
        length = 0.0
        if success and len(path) > 1:
            for (x0, y0), (x1, y1) in zip(path, path[1:]):
                length += math.hypot(x1 - x0, y1 - y0)
        min_clear = self.path_clearance(path) if success else 0.0
        # narrow_success only meaningful when a path was found
        narrow_success = success and (min_clear >= self.robot.safe_margin)
        collision = not success
        return {
            'planner': planner_name,
            'success': success,
            'path_length': round(length, 3),
            'min_clearance': round(min_clear, 3),
            'narrow_success': narrow_success,
            'collision': collision,
            'planning_time': round(elapsed, 4),
            'path': path if success else None,
        }

    def run_planner(self, planner):
        t0 = time.perf_counter()
        path = planner.plan(self)
        return self.evaluate_path(path, planner.name, time.perf_counter() - t0)

    def validate_path(self, path):
        if not path or path[0] != self.start or path[-1] != self.goal:
            return False
        if not self.is_free(*self.start) or not self.is_free(*self.goal):
            return False
        for point in path:
            if not self.is_free(*point):
                return False
        return all(not self.collision_line(a, b) for a, b in zip(path, path[1:]))


class RobotFootprint:
    def __init__(self, body_width=3, body_length=4, leg_margin=1, sensor_margin=1, safe_margin=1):
        self.body_width = body_width
        self.body_length = body_length
        self.leg_margin = leg_margin
        self.sensor_margin = sensor_margin
        self.safe_margin = safe_margin
        self.eff_width = body_width + 2 * (leg_margin + sensor_margin + safe_margin)
        self.inflation_radius = int(math.ceil(self.eff_width / 2))


class PlannerBase:
    def __init__(self, name):
        self.name = name

    def plan(self, benchmark):
        raise NotImplementedError


class AStarPlanner(PlannerBase):
    def __init__(self):
        super().__init__('A*')

    def plan(self, benchmark):
        start = benchmark.start
        goal = benchmark.goal
        if not benchmark.is_free(*start) or not benchmark.is_free(*goal):
            return None
        open_set = []
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start] = 0
        open_set.append((self.heuristic(start, goal), start))
        closed = set()
        while open_set:
            open_set.sort(key=lambda x: x[0])
            _, current = open_set.pop(0)
            if current == goal:
                return self.reconstruct(came_from, current)
            closed.add(current)
            for nxt in self.neighbors(current, benchmark):
                if nxt in closed:
                    continue
                tentative_g = g_score[current] + self.move_cost(current, nxt)
                if tentative_g < g_score[nxt]:
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    f = tentative_g + self.heuristic(nxt, goal)
                    open_set.append((f, nxt))
        return None

    def neighbors(self, current, benchmark):
        x, y = current
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1,-1), (1,-1), (-1,1), (1,1)]
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if benchmark.is_free(nx, ny) and self.diagonal_is_clear(current, (nx, ny), benchmark):
                yield (nx, ny)

    def diagonal_is_clear(self, current, nxt, benchmark):
        x, y = current
        nx, ny = nxt
        if x == nx or y == ny:
            return True
        return benchmark.is_free(nx, y) and benchmark.is_free(x, ny)

    def move_cost(self, a, b):
        return math.hypot(b[0]-a[0], b[1]-a[1])

    def heuristic(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))


class DijkstraPlanner(AStarPlanner):
    def __init__(self):
        super().__init__()
        self.name = 'Dijkstra'

    def plan(self, benchmark):
        start = benchmark.start
        goal = benchmark.goal
        if not benchmark.is_free(*start) or not benchmark.is_free(*goal):
            return None
        open_set = []
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start] = 0
        open_set.append((0, start))
        closed = set()
        while open_set:
            open_set.sort(key=lambda x: x[0])
            _, current = open_set.pop(0)
            if current == goal:
                return self.reconstruct(came_from, current)
            closed.add(current)
            for nxt in self.neighbors(current, benchmark):
                if nxt in closed:
                    continue
                tentative_g = g_score[current] + self.move_cost(current, nxt)
                if tentative_g < g_score[nxt]:
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    open_set.append((tentative_g, nxt))
        return None


class RRTPlanner(PlannerBase):
    def __init__(self, max_iter=500, step_size=3):
        super().__init__('RRT')
        self.max_iter = max_iter
        self.step_size = step_size

    def plan(self, benchmark):
        start = benchmark.start
        goal = benchmark.goal
        if not benchmark.is_free(*start) or not benchmark.is_free(*goal):
            return None
        nodes = {start: None}
        for _ in range(self.max_iter):
            x_rand = self.random_point(benchmark)
            x_near = min(nodes.keys(), key=lambda p: self.dist(p, x_rand))
            x_new = self.steer(x_near, x_rand)
            # skip if duplicate node or collision
            if x_new in nodes or benchmark.collision_line(x_near, x_new):
                continue
            nodes[x_new] = x_near
            if self.dist(x_new, goal) <= self.step_size and not benchmark.collision_line(x_new, goal):
                if x_new == goal:
                    return self.reconstruct(nodes, goal)
                nodes[goal] = x_new
                return self.reconstruct(nodes, goal)
        return None

    def random_point(self, benchmark):
        w = len(benchmark.grid[0])
        h = len(benchmark.grid)
        return (random.randrange(w), random.randrange(h))

    def steer(self, source, target):
        dx = target[0] - source[0]
        dy = target[1] - source[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return source
        scale = min(self.step_size / length, 1)
        return (int(round(source[0] + dx * scale)), int(round(source[1] + dy * scale)))

    def dist(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def reconstruct(self, nodes, current):
        path = [current]
        visited = set()
        # protect against cycles in the parent map
        while True:
            parent = nodes.get(current)
            if parent is None:
                break
            if parent in visited:
                # cycle detected, stop reconstruction
                break
            visited.add(parent)
            current = parent
            path.append(current)
            # safety: don't iterate more times than number of nodes
            if len(path) > len(nodes) + 5:
                break
        return list(reversed(path))


class RRTStarPlanner(RRTPlanner):
    def __init__(self, max_iter=500, step_size=3, radius=6):
        super().__init__(max_iter, step_size)
        self.name = 'RRT*'
        self.radius = radius

    def plan(self, benchmark):
        start = benchmark.start
        goal = benchmark.goal
        if not benchmark.is_free(*start) or not benchmark.is_free(*goal):
            return None
        nodes = {start: None}
        cost = {start: 0}
        for _ in range(self.max_iter):
            x_rand = self.random_point(benchmark)
            x_near = min(nodes.keys(), key=lambda p: self.dist(p, x_rand))
            x_new = self.steer(x_near, x_rand)
            # skip if duplicate node or collision
            if x_new in nodes or benchmark.collision_line(x_near, x_new):
                continue
            neighbors = [p for p in nodes if self.dist(p, x_new) <= self.radius]
            best_parent = x_near
            best_cost = cost[x_near] + self.dist(x_near, x_new)
            for p in neighbors:
                c = cost[p] + self.dist(p, x_new)
                if c < best_cost and not benchmark.collision_line(p, x_new):
                    best_parent = p
                    best_cost = c
            nodes[x_new] = best_parent
            cost[x_new] = best_cost
            for p in neighbors:
                c = best_cost + self.dist(x_new, p)
                if c < cost[p] and not benchmark.collision_line(x_new, p):
                    nodes[p] = x_new
                    cost[p] = c
        goal_candidates = [
            point for point in nodes
            if self.dist(point, goal) <= self.step_size and not benchmark.collision_line(point, goal)
        ]
        if not goal_candidates:
            return None
        best_parent = min(goal_candidates, key=lambda point: cost[point] + self.dist(point, goal))
        if best_parent != goal:
            nodes[goal] = best_parent
        return self.reconstruct(nodes, goal)


class HybridAStarPlanner(PlannerBase):
    def __init__(self, angle_steps=8, step_size=3, max_iter=3000):
        super().__init__('Hybrid-A*')
        self.angle_steps = angle_steps
        self.step_size = step_size
        self.max_iter = max_iter

    def plan(self, benchmark):
        start = benchmark.start
        goal = benchmark.goal
        if not benchmark.is_free(*start) or not benchmark.is_free(*goal):
            return None
        start_state = (start[0], start[1], 0)
        goal_radius = 2
        open_set = []
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_state] = 0
        open_set.append((self.heuristic(start_state, goal), start_state))
        closed = set()
        while open_set and len(closed) < self.max_iter:
            open_set.sort(key=lambda x: x[0])
            _, current = open_set.pop(0)
            if self.reached_goal(current, goal, goal_radius) and not benchmark.collision_line((current[0], current[1]), goal):
                path = self.reconstruct(came_from, current)
                if path[-1] != goal:
                    path.append(goal)
                return path
            if current in closed:
                continue
            closed.add(current)
            for nxt in self.neighbors(current, benchmark):
                if nxt in closed:
                    continue
                tentative_g = g_score[current] + self.step_size
                if tentative_g < g_score[nxt]:
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    open_set.append((tentative_g + self.heuristic(nxt, goal), nxt))
        return None

    def neighbors(self, state, benchmark):
        x, y, theta = state
        directions = [-1, 0, 1]
        for dtheta in directions:
            new_theta = (theta + dtheta) % self.angle_steps
            angle = 2 * math.pi * new_theta / self.angle_steps
            nx = int(round(x + math.cos(angle) * self.step_size))
            ny = int(round(y + math.sin(angle) * self.step_size))
            if benchmark.is_free(nx, ny) and not benchmark.collision_line((x,y), (nx, ny)):
                yield (nx, ny, new_theta)

    def heuristic(self, state, goal):
        x, y, _ = state
        return math.hypot(x - goal[0], y - goal[1])

    def reached_goal(self, state, goal, radius):
        x, y, _ = state
        return math.hypot(x - goal[0], y - goal[1]) <= radius

    def reconstruct(self, came_from, current):
        path = [(current[0], current[1])]
        while current in came_from:
            current = came_from[current]
            path.append((current[0], current[1]))
        return list(reversed(path))


class OursPlanner(PlannerBase):
    def __init__(self, use_clearance_penalty=True, clearance_penalty=5.0, memory_penalty=100.0):
        super().__init__('Ours')
        self.memory = {}
        self.use_clearance_penalty = use_clearance_penalty
        self.clearance_penalty = clearance_penalty
        self.memory_penalty = memory_penalty

    def plan(self, benchmark):
        start = benchmark.start
        goal = benchmark.goal
        if not benchmark.is_free(*start) or not benchmark.is_free(*goal):
            return None
        open_set = []
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start] = 0
        open_set.append((self.heuristic(start, goal), start))
        closed = set()
        while open_set:
            open_set.sort(key=lambda x: x[0])
            _, current = open_set.pop(0)
            if current == goal:
                return self.reconstruct(came_from, current)
            closed.add(current)
            for nxt in self.neighbors(current, benchmark):
                if nxt in closed:
                    continue
                cost = self.move_cost(current, nxt)
                narrow_penalty = self.narrow_penalty(nxt, benchmark) if self.use_clearance_penalty else 0.0
                memory_penalty = self.memory.get(nxt, 0) * self.memory_penalty
                tentative_g = g_score[current] + cost + narrow_penalty + memory_penalty
                if tentative_g < g_score[nxt]:
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    f = tentative_g + self.heuristic(nxt, goal)
                    open_set.append((f, nxt))
        return None

    def remember_failed_region(self, region):
        for point in region:
            self.memory[point] = self.memory.get(point, 0) + 1

    def decay_memory(self, amount=1.0):
        for point in list(self.memory):
            self.memory[point] = max(0.0, self.memory[point] - amount)
            if self.memory[point] == 0:
                del self.memory[point]

    def neighbors(self, current, benchmark):
        x, y = current
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1,-1), (1,-1), (-1,1), (1,1)]
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if benchmark.is_free(nx, ny) and self.diagonal_is_clear(current, (nx, ny), benchmark):
                yield (nx, ny)

    def diagonal_is_clear(self, current, nxt, benchmark):
        x, y = current
        nx, ny = nxt
        if x == nx or y == ny:
            return True
        return benchmark.is_free(nx, y) and benchmark.is_free(x, ny)

    def move_cost(self, a, b):
        return math.hypot(b[0]-a[0], b[1]-a[1])

    def narrow_penalty(self, point, benchmark):
        clearance = benchmark.local_clearance(point[0], point[1])
        if clearance < benchmark.robot.safe_margin:
            return 100.0
        if clearance < benchmark.robot.inflation_radius + 1:
            return self.clearance_penalty
        return 0.0

    def heuristic(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))


def build_scene(lines):
    grid = []
    start = None
    goal = None
    widths = {len(line.rstrip('\n')) for line in lines}
    if len(widths) != 1:
        raise ValueError(f'scene rows must have equal width, got {sorted(widths)}')
    max_width = widths.pop()
    for y, line in enumerate(lines):
        row = []
        text = line.rstrip('\n')
        for x in range(max_width):
            ch = text[x] if x < len(text) else ' '
            if ch == '#':
                row.append(1)
            elif ch == 'S':
                row.append(0)
                start = (x, y)
            elif ch == 'G':
                row.append(0)
                goal = (x, y)
            else:
                row.append(0)
        grid.append(row)
    return grid, start, goal


def make_single_passage_scene(passage_width, width=60, height=22):
    grid = [[' '] * width for _ in range(height)]
    for x in range(width):
        grid[0][x] = '#'
        grid[-1][x] = '#'
    for y in range(height):
        grid[y][0] = '#'
        grid[y][-1] = '#'
    opening_x = width // 2 - passage_width // 2
    for y in range(9, 13):
        for x in range(1, width - 1):
            if not opening_x <= x < opening_x + passage_width:
                grid[y][x] = '#'
    grid[5][7] = 'S'
    grid[16][width - 8] = 'G'
    return [''.join(row) for row in grid]


def make_wide_scene(width=60, height=22):
    grid = [[' '] * width for _ in range(height)]
    for x in range(width):
        grid[0][x] = '#'
        grid[-1][x] = '#'
    for y in range(height):
        grid[y][0] = '#'
        grid[y][-1] = '#'
    grid[5][7] = 'S'
    grid[16][width - 8] = 'G'
    return [''.join(row) for row in grid]


def make_multi_passage_scene(width=60, height=22):
    grid = [[' '] * width for _ in range(height)]
    for x in range(width):
        grid[0][x] = '#'
        grid[-1][x] = '#'
    for y in range(height):
        grid[y][0] = '#'
        grid[y][-1] = '#'
    openings = [(12, 5), (28, 7), (45, 9)]
    for y in range(9, 13):
        for x in range(1, width - 1):
            if not any(start <= x < start + opening_width for start, opening_width in openings):
                grid[y][x] = '#'
    grid[5][7] = 'S'
    grid[16][width - 8] = 'G'
    return [''.join(row) for row in grid]


def make_repeated_passage_scenes(
        start=(5, 7), goal=(44, 7), deceptive_opening_y=7,
        safe_opening_start=16, barrier_x=24, width=50, height=23,
        short_passage_blocked=True):
    observed = [[' '] * width for _ in range(height)]
    for x in range(width):
        observed[0][x] = '#'
        observed[-1][x] = '#'
    for y in range(height):
        observed[y][0] = '#'
        observed[y][-1] = '#'
    safe_opening = range(safe_opening_start, safe_opening_start + 3)
    for y in range(1, height - 1):
        if y != deceptive_opening_y and y not in safe_opening:
            observed[y][barrier_x] = '#'
    observed[start[1]][start[0]] = 'S'
    observed[goal[1]][goal[0]] = 'G'

    truth = [row[:] for row in observed]
    if short_passage_blocked:
        truth[deceptive_opening_y][barrier_x] = '#'
    failed_region = {
        (x, y)
        for x in range(barrier_x - 4, barrier_x + 5)
        for y in range(deceptive_opening_y - 2, deceptive_opening_y + 3)
    }
    return [''.join(row) for row in observed], [''.join(row) for row in truth], failed_region


def occupancy_patch(grid, center, radius=4):
    center_x, center_y = center
    return tuple(
        grid[y][x]
        if 0 <= y < len(grid) and 0 <= x < len(grid[0])
        else 1
        for y in range(center_y - radius, center_y + radius + 1)
        for x in range(center_x - radius, center_x + radius + 1)
    )


def patch_similarity(left, right):
    if not left or len(left) != len(right):
        return 0.0
    return sum(a == b for a, b in zip(left, right)) / len(left)


def find_local_passage_anchors(grid, max_opening_width=4):
    height = len(grid)
    width = len(grid[0])
    anchors = []
    for x in range(1, width - 1):
        y = 1
        while y < height - 1:
            if grid[y][x] != 0:
                y += 1
                continue
            opening_start = y
            while y < height - 1 and grid[y][x] == 0:
                y += 1
            opening_width = y - opening_start
            if (
                    opening_width <= max_opening_width
                    and opening_start > 1
                    and y < height - 1
                    and grid[opening_start - 1][x] == 1
                    and grid[y][x] == 1):
                center = (x, opening_start + (opening_width - 1) // 2)
                anchors.append({
                    'center': center,
                    'direction': 'horizontal',
                    'opening_width': opening_width,
                    'patch': occupancy_patch(grid, center),
                })
    for y in range(1, height - 1):
        x = 1
        while x < width - 1:
            if grid[y][x] != 0:
                x += 1
                continue
            opening_start = x
            while x < width - 1 and grid[y][x] == 0:
                x += 1
            opening_width = x - opening_start
            if (
                    opening_width <= max_opening_width
                    and opening_start > 1
                    and x < width - 1
                    and grid[y][opening_start - 1] == 1
                    and grid[y][x] == 1):
                center = (opening_start + (opening_width - 1) // 2, y)
                anchors.append({
                    'center': center,
                    'direction': 'vertical',
                    'opening_width': opening_width,
                    'patch': occupancy_patch(grid, center),
                })
    return anchors


class SemanticAnchorMemory:
    def __init__(self, similarity_threshold=0.82):
        self.similarity_threshold = similarity_threshold
        self.entries = []

    def remember_failure(self, grid, failed_region, memory_type='static-dead-end'):
        region_center = (
            round(statistics.fmean(x for x, _ in failed_region)),
            round(statistics.fmean(y for _, y in failed_region)),
        )
        candidates = [
            anchor for anchor in find_local_passage_anchors(grid)
            if anchor['center'] in failed_region
        ]
        if not candidates:
            return None
        anchor = min(
            candidates,
            key=lambda candidate: (
                candidate['opening_width'],
                math.hypot(
                    candidate['center'][0] - region_center[0],
                    candidate['center'][1] - region_center[1])))
        center_x, center_y = anchor['center']
        entry = {
            **anchor,
            'memory_type': memory_type,
            'confidence': 1.0,
            'relative_region': {
                (x - center_x, y - center_y) for x, y in failed_region
            },
        }
        self.entries.append(entry)
        return entry

    def recall(self, grid):
        projected_cells = set()
        matches = []
        candidates = find_local_passage_anchors(grid)
        for entry in self.entries:
            compatible = [
                (
                    patch_similarity(entry['patch'], candidate['patch']),
                    candidate,
                )
                for candidate in candidates
                if candidate['direction'] == entry['direction']
                and candidate['opening_width'] == entry['opening_width']
            ]
            if not compatible:
                continue
            similarity, candidate = max(compatible, key=lambda item: item[0])
            if similarity < self.similarity_threshold:
                continue
            center_x, center_y = candidate['center']
            region = {
                (center_x + dx, center_y + dy)
                for dx, dy in entry['relative_region']
            }
            projected_cells.update(region)
            matches.append({
                'SourceCenter': entry['center'],
                'TargetCenter': candidate['center'],
                'Direction': candidate['direction'],
                'OpeningWidth': candidate['opening_width'],
                'Similarity': round(similarity, 4),
                'ProjectedCells': len(region),
            })
        return projected_cells, matches


def mean_ci95(values):
    if not values:
        return 0.0, 0.0, 0.0
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean, mean
    margin = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - margin, mean + margin


def wilson_ci95(successes, trials):
    if trials == 0:
        return 0.0, 0.0
    z = 1.96
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt((rate * (1 - rate) + z * z / (4 * trials)) / trials) / denominator
    return center - margin, center + margin


def write_dict_rows(csv_path, rows):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_repeated_passage_experiment(tasks=3, max_attempts=2):
    observed_lines, truth_lines, failed_region = make_repeated_passage_scenes()
    robot = RobotFootprint(body_width=0, body_length=0, leg_margin=0, sensor_margin=0, safe_margin=0)
    observed_grid, start, goal = build_scene(observed_lines)
    truth_grid, _, _ = build_scene(truth_lines)
    rows = []

    variants = [
        ('A* repeated baseline', AStarPlanner(), False),
        ('Memory-aware replanning', OursPlanner(use_clearance_penalty=False), True),
    ]
    for variant_name, planner, uses_memory in variants:
        for task in range(1, tasks + 1):
            for attempt in range(1, max_attempts + 1):
                planning_benchmark = GridBenchmark(observed_grid, start, goal, robot)
                truth_benchmark = GridBenchmark(truth_grid, start, goal, robot)
                summary = planning_benchmark.run_planner(planner)
                valid_in_truth = truth_benchmark.validate_path(summary['path'])
                used_short_passage = bool(summary['path']) and any(point in failed_region for point in summary['path'])
                rows.append({
                    'Variant': variant_name,
                    'Task': task,
                    'Attempt': attempt,
                    'Planned': summary['success'],
                    'Executable': valid_in_truth,
                    'UsedShortPassage': used_short_passage,
                    'Length': summary['path_length'],
                    'RememberedCells': len(getattr(planner, 'memory', {})),
                })
                if valid_in_truth:
                    break
                if uses_memory and used_short_passage:
                    planner.remember_failed_region(failed_region)

    write_dict_rows('results_repeated_passage.csv', rows)
    return rows


def run_sampling_statistics(seeds=100):
    scenes = {
        'w7': make_single_passage_scene(7),
        'w6': make_single_passage_scene(6),
        'w5': make_single_passage_scene(5),
        'w4': make_single_passage_scene(4),
        'wide': make_wide_scene(),
        'multi': make_multi_passage_scene(),
    }
    phases = {
        'point': RobotFootprint(body_width=0, body_length=0, leg_margin=0, sensor_margin=0, safe_margin=0),
        'small': RobotFootprint(body_width=1, body_length=2, leg_margin=0, sensor_margin=0, safe_margin=0),
        'body': RobotFootprint(body_width=2, body_length=2, leg_margin=0, sensor_margin=0, safe_margin=1),
    }
    planner_factories = {
        'RRT': RRTPlanner,
        'RRT*': RRTStarPlanner,
    }
    rows = []
    for phase_name, robot in phases.items():
        for scene_name, lines in scenes.items():
            grid, start, goal = build_scene(lines)
            benchmark = GridBenchmark(grid, start, goal, robot)
            for planner_name, planner_factory in planner_factories.items():
                for seed in range(seeds):
                    random.seed(seed)
                    summary = benchmark.run_planner(planner_factory())
                    rows.append({
                        'Phase': phase_name,
                        'Scene': scene_name,
                        'Planner': planner_name,
                        'Seed': seed,
                        'Success': summary['success'],
                        'Length': summary['path_length'],
                        'Clearance': summary['min_clearance'],
                        'Time': summary['planning_time'],
                    })
    write_dict_rows('results_sampling_trials.csv', rows)

    summary_rows = []
    for phase_name in phases:
        for scene_name in scenes:
            for planner_name in planner_factories:
                selected = [
                    row for row in rows
                    if row['Phase'] == phase_name and row['Scene'] == scene_name and row['Planner'] == planner_name
                ]
                successful = [row for row in selected if row['Success']]
                ci_low, ci_high = wilson_ci95(len(successful), len(selected))
                mean_length, length_low, length_high = mean_ci95([row['Length'] for row in successful])
                mean_time, time_low, time_high = mean_ci95([row['Time'] for row in selected])
                summary_rows.append({
                    'Phase': phase_name,
                    'Scene': scene_name,
                    'Planner': planner_name,
                    'Trials': len(selected),
                    'Successes': len(successful),
                    'SuccessRate': round(len(successful) / len(selected), 4),
                    'SuccessCI95Low': round(ci_low, 4),
                    'SuccessCI95High': round(ci_high, 4),
                    'MeanLengthSuccess': round(mean_length, 4),
                    'LengthCI95Low': round(length_low, 4),
                    'LengthCI95High': round(length_high, 4),
                    'MeanTime': round(mean_time, 6),
                    'TimeCI95Low': round(time_low, 6),
                    'TimeCI95High': round(time_high, 6),
                })
    write_dict_rows('results_sampling_summary.csv', summary_rows)
    return summary_rows


def run_memory_statistics(batches=30, tasks=5, max_attempts=2):
    rng = random.Random(0)
    robot = RobotFootprint(body_width=0, body_length=0, leg_margin=0, sensor_margin=0, safe_margin=0)
    task_sets = []
    for batch in range(batches):
        tasks_in_batch = []
        for task in range(1, tasks + 1):
            start = (5, rng.choice([5, 6, 7, 8, 9]))
            goal = (44, rng.choice([5, 6, 7, 8, 9]))
            safe_opening_start = rng.choice([15, 16, 17])
            tasks_in_batch.append((task, start, goal, safe_opening_start))
        task_sets.append((batch, tasks_in_batch))

    rows = []
    variants = [
        ('A* repeated baseline', AStarPlanner, False),
        ('Memory-aware replanning', lambda: OursPlanner(use_clearance_penalty=False), True),
    ]
    for variant_name, planner_factory, uses_memory in variants:
        for batch, tasks_in_batch in task_sets:
            planner = planner_factory()
            for task, start, goal, safe_opening_start in tasks_in_batch:
                observed_lines, truth_lines, failed_region = make_repeated_passage_scenes(
                    start=start, goal=goal, safe_opening_start=safe_opening_start)
                observed_grid, _, _ = build_scene(observed_lines)
                truth_grid, _, _ = build_scene(truth_lines)
                for attempt in range(1, max_attempts + 1):
                    planning_benchmark = GridBenchmark(observed_grid, start, goal, robot)
                    truth_benchmark = GridBenchmark(truth_grid, start, goal, robot)
                    plan_summary = planning_benchmark.run_planner(planner)
                    valid_in_truth = truth_benchmark.validate_path(plan_summary['path'])
                    used_short_passage = bool(plan_summary['path']) and any(
                        point in failed_region for point in plan_summary['path'])
                    rows.append({
                        'Variant': variant_name,
                        'Batch': batch,
                        'Task': task,
                        'Attempt': attempt,
                        'Start': str(start),
                        'Goal': str(goal),
                        'SafeOpeningStart': safe_opening_start,
                        'Planned': plan_summary['success'],
                        'Executable': valid_in_truth,
                        'UsedShortPassage': used_short_passage,
                        'Length': plan_summary['path_length'],
                        'RememberedCells': len(getattr(planner, 'memory', {})),
                    })
                    if valid_in_truth:
                        break
                    if uses_memory and used_short_passage:
                        planner.remember_failed_region(failed_region)
    write_dict_rows('results_memory_trials.csv', rows)

    summary_rows = []
    total_tasks = batches * tasks
    for variant_name, _, _ in variants:
        selected = [row for row in rows if row['Variant'] == variant_name]
        executable = [row for row in selected if row['Executable']]
        successes = len(executable)
        ci_low, ci_high = wilson_ci95(successes, total_tasks)
        summary_rows.append({
            'Variant': variant_name,
            'Batches': batches,
            'Tasks': total_tasks,
            'Attempts': len(selected),
            'ExecutableTasks': successes,
            'ExecutableRate': round(successes / total_tasks, 4),
            'ExecutableCI95Low': round(ci_low, 4),
            'ExecutableCI95High': round(ci_high, 4),
            'FailedShortPassageAttempts': sum(row['UsedShortPassage'] and not row['Executable'] for row in selected),
            'MeanAttemptsPerTask': round(len(selected) / total_tasks, 4),
            'MeanExecutableLength': round(statistics.fmean(row['Length'] for row in executable), 4) if executable else 0.0,
        })
    write_dict_rows('results_memory_summary.csv', summary_rows)
    return summary_rows


def run_ablation_statistics(batches=30, tasks=5, max_attempts=2):
    rng = random.Random(0)
    robot = RobotFootprint(body_width=0, body_length=0, leg_margin=0, sensor_margin=0, safe_margin=0)
    task_sets = []
    for batch in range(batches):
        tasks_in_batch = []
        for task in range(1, tasks + 1):
            tasks_in_batch.append((
                task,
                (5, rng.choice([5, 6, 7, 8, 9])),
                (44, rng.choice([5, 6, 7, 8, 9])),
                rng.choice([15, 16, 17]),
            ))
        task_sets.append((batch, tasks_in_batch))

    variants = [
        ('A* baseline', AStarPlanner, False),
        ('Clearance-aware A*', lambda: OursPlanner(clearance_penalty=0.5), False),
        ('Memory-aware', lambda: OursPlanner(clearance_penalty=0.5), True),
    ]
    rows = []
    for variant_name, planner_factory, uses_memory in variants:
        for batch, tasks_in_batch in task_sets:
            planner = planner_factory()
            for task, start, goal, safe_opening_start in tasks_in_batch:
                observed_lines, truth_lines, failed_region = make_repeated_passage_scenes(
                    start=start, goal=goal, safe_opening_start=safe_opening_start)
                observed_grid, _, _ = build_scene(observed_lines)
                truth_grid, _, _ = build_scene(truth_lines)
                for attempt in range(1, max_attempts + 1):
                    planning_benchmark = GridBenchmark(observed_grid, start, goal, robot)
                    truth_benchmark = GridBenchmark(truth_grid, start, goal, robot)
                    plan_summary = planning_benchmark.run_planner(planner)
                    valid_in_truth = truth_benchmark.validate_path(plan_summary['path'])
                    used_short_passage = bool(plan_summary['path']) and any(
                        point in failed_region for point in plan_summary['path'])
                    rows.append({
                        'Variant': variant_name,
                        'Batch': batch,
                        'Task': task,
                        'Attempt': attempt,
                        'Start': str(start),
                        'Goal': str(goal),
                        'SafeOpeningStart': safe_opening_start,
                        'Executable': valid_in_truth,
                        'UsedShortPassage': used_short_passage,
                        'Length': plan_summary['path_length'],
                        'RememberedCells': len(getattr(planner, 'memory', {})),
                    })
                    if valid_in_truth:
                        break
                    if uses_memory and used_short_passage:
                        planner.remember_failed_region(failed_region)
    write_dict_rows('results_ablation_trials.csv', rows)

    summary_rows = []
    total_tasks = batches * tasks
    for variant_name, _, _ in variants:
        selected = [row for row in rows if row['Variant'] == variant_name]
        executable = [row for row in selected if row['Executable']]
        ci_low, ci_high = wilson_ci95(len(executable), total_tasks)
        summary_rows.append({
            'Variant': variant_name,
            'Tasks': total_tasks,
            'Attempts': len(selected),
            'ExecutableTasks': len(executable),
            'ExecutableRate': round(len(executable) / total_tasks, 4),
            'ExecutableCI95Low': round(ci_low, 4),
            'ExecutableCI95High': round(ci_high, 4),
            'FailedShortPassageAttempts': sum(
                row['UsedShortPassage'] and not row['Executable'] for row in selected),
            'MeanAttemptsPerTask': round(len(selected) / total_tasks, 4),
        })
    write_dict_rows('results_ablation_summary.csv', summary_rows)
    return summary_rows


def run_environment_recovery_experiment(recovery_tasks=5):
    robot = RobotFootprint(body_width=0, body_length=0, leg_margin=0, sensor_margin=0, safe_margin=0)
    planner = OursPlanner(clearance_penalty=0.5)
    observed_lines, blocked_lines, failed_region = make_repeated_passage_scenes()
    _, reopened_lines, _ = make_repeated_passage_scenes(short_passage_blocked=False)
    observed_grid, start, goal = build_scene(observed_lines)
    blocked_grid, _, _ = build_scene(blocked_lines)
    reopened_grid, _, _ = build_scene(reopened_lines)
    rows = []

    for stage, truth_grid in [('learn-failure', blocked_grid), ('avoid-known-failure', blocked_grid)]:
        planning_benchmark = GridBenchmark(observed_grid, start, goal, robot)
        truth_benchmark = GridBenchmark(truth_grid, start, goal, robot)
        plan_summary = planning_benchmark.run_planner(planner)
        executable = truth_benchmark.validate_path(plan_summary['path'])
        used_short_passage = any(point in failed_region for point in plan_summary['path'])
        rows.append({
            'Stage': stage,
            'RecoveryTask': 0,
            'Executable': executable,
            'UsedShortPassage': used_short_passage,
            'Length': plan_summary['path_length'],
            'MemoryStrength': round(max(planner.memory.values(), default=0.0), 2),
        })
        if not executable and used_short_passage:
            planner.remember_failed_region(failed_region)

    for task in range(1, recovery_tasks + 1):
        planner.decay_memory(amount=0.25)
        planning_benchmark = GridBenchmark(observed_grid, start, goal, robot)
        truth_benchmark = GridBenchmark(reopened_grid, start, goal, robot)
        plan_summary = planning_benchmark.run_planner(planner)
        rows.append({
            'Stage': 'reopened',
            'RecoveryTask': task,
            'Executable': truth_benchmark.validate_path(plan_summary['path']),
            'UsedShortPassage': any(point in failed_region for point in plan_summary['path']),
            'Length': plan_summary['path_length'],
            'MemoryStrength': round(max(planner.memory.values(), default=0.0), 2),
        })
    write_dict_rows('results_environment_recovery.csv', rows)
    return rows


def run_unrelated_maps_experiment():
    robot = RobotFootprint(body_width=0, body_length=0, leg_margin=0, sensor_margin=0, safe_margin=0)
    planner = OursPlanner(clearance_penalty=0.5)
    rows = []
    map_configs = [
        ('map-a', 5, 24),
        ('map-b', 12, 24),
        ('map-c', 19, 5),
    ]
    for map_name, deceptive_opening_y, safe_opening_start in map_configs:
        start = (5, deceptive_opening_y)
        goal = (44, deceptive_opening_y)
        observed_lines, truth_lines, failed_region = make_repeated_passage_scenes(
            start=start, goal=goal, deceptive_opening_y=deceptive_opening_y,
            safe_opening_start=safe_opening_start, height=29)
        observed_grid, _, _ = build_scene(observed_lines)
        truth_grid, _, _ = build_scene(truth_lines)
        for attempt in range(1, 3):
            planning_benchmark = GridBenchmark(observed_grid, start, goal, robot)
            truth_benchmark = GridBenchmark(truth_grid, start, goal, robot)
            plan_summary = planning_benchmark.run_planner(planner)
            executable = truth_benchmark.validate_path(plan_summary['path'])
            used_current_short_passage = any(point in failed_region for point in plan_summary['path'])
            rows.append({
                'Map': map_name,
                'Attempt': attempt,
                'Executable': executable,
                'UsedCurrentShortPassage': used_current_short_passage,
                'Length': plan_summary['path_length'],
                'RememberedCellsBeforeUpdate': len(planner.memory),
            })
            if executable:
                break
            if used_current_short_passage:
                planner.remember_failed_region(failed_region)
    write_dict_rows('results_unrelated_maps.csv', rows)
    return rows


def add_map_noise(lines, obstacle_cells):
    grid = [list(line) for line in lines]
    for x, y in obstacle_cells:
        if 0 <= y < len(grid) and 0 <= x < len(grid[0]) and grid[y][x] == ' ':
            grid[y][x] = '#'
    return [''.join(row) for row in grid]


def run_semantic_anchor_migration_experiment():
    robot = RobotFootprint(body_width=0, body_length=0, leg_margin=0, sensor_margin=0, safe_margin=0)
    source_observed_lines, _, source_failed_region = make_repeated_passage_scenes()
    source_grid, _, _ = build_scene(source_observed_lines)
    semantic_memory = SemanticAnchorMemory(similarity_threshold=0.8)
    source_entry = semantic_memory.remember_failure(source_grid, source_failed_region)
    absolute_memory = set(source_failed_region)
    target_configs = [
        ('translated-x', 33, 7, 16, []),
        ('translated-xy', 20, 11, 17, []),
        ('translated-xy-noisy', 31, 13, 18, [(30, 10), (32, 16)]),
    ]
    rows = []
    for map_name, barrier_x, deceptive_y, safe_opening_start, noise_cells in target_configs:
        start = (5, deceptive_y)
        goal = (44, deceptive_y)
        observed_lines, truth_lines, failed_region = make_repeated_passage_scenes(
            start=start, goal=goal, deceptive_opening_y=deceptive_y,
            safe_opening_start=safe_opening_start, barrier_x=barrier_x,
            width=50, height=27)
        observed_lines = add_map_noise(observed_lines, noise_cells)
        truth_lines = add_map_noise(truth_lines, noise_cells)
        observed_grid, _, _ = build_scene(observed_lines)
        truth_grid, _, _ = build_scene(truth_lines)
        projected_cells, matches = semantic_memory.recall(observed_grid)
        variants = [
            ('No transferred memory', set()),
            ('Absolute-cell transfer', absolute_memory),
            ('Semantic-anchor transfer', projected_cells),
        ]
        for variant, remembered_cells in variants:
            planner = OursPlanner(use_clearance_penalty=False)
            planner.remember_failed_region(remembered_cells)
            planning_benchmark = GridBenchmark(observed_grid, start, goal, robot)
            truth_benchmark = GridBenchmark(truth_grid, start, goal, robot)
            summary = planning_benchmark.run_planner(planner)
            rows.append({
                'Map': map_name,
                'Variant': variant,
                'SourceAnchor': str(source_entry['center']),
                'MatchedAnchor': str(matches[0]['TargetCenter']) if matches else '',
                'MatchSimilarity': matches[0]['Similarity'] if matches else 0.0,
                'TransferredCells': len(remembered_cells),
                'Planned': summary['success'],
                'Executable': truth_benchmark.validate_path(summary['path']),
                'UsedTargetFailedPassage': bool(summary['path']) and any(
                    point in failed_region for point in summary['path']),
                'Length': summary['path_length'],
            })
    write_dict_rows('results_semantic_anchor_migration.csv', rows)
    return rows


def run_semantic_anchor_migration_statistics(seeds=100):
    robot = RobotFootprint(body_width=0, body_length=0, leg_margin=0, sensor_margin=0, safe_margin=0)
    source_observed_lines, _, source_failed_region = make_repeated_passage_scenes()
    source_grid, _, _ = build_scene(source_observed_lines)
    semantic_memory = SemanticAnchorMemory(similarity_threshold=0.8)
    semantic_memory.remember_failure(source_grid, source_failed_region)
    absolute_memory = set(source_failed_region)
    variants = [
        'No transferred memory',
        'Absolute-cell transfer',
        'Semantic-anchor transfer',
    ]
    rows = []
    for seed in range(seeds):
        rng = random.Random(1709 * seed + 41)
        barrier_x = rng.choice([14, 15, 16, 33, 34, 35])
        deceptive_y = rng.choice([6, 9, 12, 15, 18])
        safe_opening_start = (
            rng.choice([18, 19, 20])
            if deceptive_y <= 12 else rng.choice([4, 5, 6])
        )
        start = (5, deceptive_y)
        goal = (49, deceptive_y)
        observed_lines, truth_lines, failed_region = make_repeated_passage_scenes(
            start=start, goal=goal, deceptive_opening_y=deceptive_y,
            safe_opening_start=safe_opening_start, barrier_x=barrier_x,
            width=55, height=29)
        noise_cells = [
            (barrier_x - 1, deceptive_y - 3),
            (barrier_x + 1, deceptive_y + 3),
        ][:rng.randint(0, 2)]
        observed_lines = add_map_noise(observed_lines, noise_cells)
        truth_lines = add_map_noise(truth_lines, noise_cells)
        observed_grid, _, _ = build_scene(observed_lines)
        truth_grid, _, _ = build_scene(truth_lines)
        projected_cells, matches = semantic_memory.recall(observed_grid)
        memory_cells = {
            'No transferred memory': set(),
            'Absolute-cell transfer': absolute_memory,
            'Semantic-anchor transfer': projected_cells,
        }
        for variant in variants:
            planner = OursPlanner(use_clearance_penalty=False)
            planner.remember_failed_region(memory_cells[variant])
            planning_benchmark = GridBenchmark(observed_grid, start, goal, robot)
            truth_benchmark = GridBenchmark(truth_grid, start, goal, robot)
            summary = planning_benchmark.run_planner(planner)
            rows.append({
                'Variant': variant,
                'Seed': seed,
                'BarrierX': barrier_x,
                'DeceptiveY': deceptive_y,
                'NoiseCells': len(noise_cells),
                'MatchedAnchor': str(matches[0]['TargetCenter']) if matches else '',
                'MatchSimilarity': matches[0]['Similarity'] if matches else 0.0,
                'TransferredCells': len(memory_cells[variant]),
                'Executable': truth_benchmark.validate_path(summary['path']),
                'UsedTargetFailedPassage': bool(summary['path']) and any(
                    point in failed_region for point in summary['path']),
                'Length': summary['path_length'],
            })
    write_dict_rows('results_semantic_anchor_migration_trials.csv', rows)

    summary_rows = []
    for variant in variants:
        selected = [row for row in rows if row['Variant'] == variant]
        successes = sum(row['Executable'] for row in selected)
        ci_low, ci_high = wilson_ci95(successes, len(selected))
        summary_rows.append({
            'Variant': variant,
            'Trials': len(selected),
            'ExecutableTasks': successes,
            'ExecutableRate': round(successes / len(selected), 4),
            'ExecutableCI95Low': round(ci_low, 4),
            'ExecutableCI95High': round(ci_high, 4),
            'FailedPassageSelections': sum(
                row['UsedTargetFailedPassage'] for row in selected),
            'MeanTransferredCells': round(statistics.fmean(
                row['TransferredCells'] for row in selected), 4),
            'MeanMatchSimilarity': round(statistics.fmean(
                row['MatchSimilarity'] for row in selected), 4),
        })
    write_dict_rows('results_semantic_anchor_migration_summary.csv', summary_rows)
    return summary_rows


def run_semantic_anchor_specificity_experiment():
    source_observed_lines, _, source_failed_region = make_repeated_passage_scenes()
    source_grid, _, _ = build_scene(source_observed_lines)
    semantic_memory = SemanticAnchorMemory(similarity_threshold=0.8)
    semantic_memory.remember_failure(source_grid, source_failed_region)

    wider_lines, _, _ = make_repeated_passage_scenes(
        deceptive_opening_y=10, safe_opening_start=18,
        barrier_x=28, width=55, height=29)
    wider_grid = [list(line) for line in wider_lines]
    wider_grid[11][28] = ' '
    wider_lines = [''.join(row) for row in wider_grid]
    cases = [
        ('different-opening-width', build_scene(wider_lines)[0]),
        ('different-direction', build_scene(make_single_passage_scene(4))[0]),
        ('open-map', build_scene(make_wide_scene())[0]),
    ]
    rows = []
    for case, grid in cases:
        projected_cells, matches = semantic_memory.recall(grid)
        rows.append({
            'Case': case,
            'Matches': len(matches),
            'ProjectedCells': len(projected_cells),
            'Rejected': not matches,
        })
    write_dict_rows('results_semantic_anchor_specificity.csv', rows)
    return rows


def rectangle_corners(x, y, theta, robot):
    half_length = (robot.body_length + 2 * robot.safe_margin) / 2
    half_width = (robot.body_width + 2 * robot.safe_margin) / 2
    if half_length == 0 and half_width == 0:
        return [(x, y)]
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    corners = []
    for local_x, local_y in [
            (-half_length, -half_width), (half_length, -half_width),
            (half_length, half_width), (-half_length, half_width)]:
        corners.append((
            x + local_x * cos_theta - local_y * sin_theta,
            y + local_x * sin_theta + local_y * cos_theta,
        ))
    return corners


def polygons_overlap(poly_a, poly_b):
    if len(poly_a) == 1:
        x, y = poly_a[0]
        return poly_b[0][0] <= x <= poly_b[2][0] and poly_b[0][1] <= y <= poly_b[2][1]
    axes = [(1, 0), (0, 1)]
    for index in range(len(poly_a)):
        x0, y0 = poly_a[index]
        x1, y1 = poly_a[(index + 1) % len(poly_a)]
        edge_x, edge_y = x1 - x0, y1 - y0
        length = math.hypot(edge_x, edge_y)
        axes.append((-edge_y / length, edge_x / length))
    for axis_x, axis_y in axes:
        projection_a = [x * axis_x + y * axis_y for x, y in poly_a]
        projection_b = [x * axis_x + y * axis_y for x, y in poly_b]
        if max(projection_a) < min(projection_b) or max(projection_b) < min(projection_a):
            return False
    return True


def rectangle_pose_collision(grid, pose, robot, dynamic_cells=None):
    x, y, theta = pose
    footprint = rectangle_corners(x, y, theta, robot)
    min_x = math.floor(min(point[0] for point in footprint) - 0.5)
    max_x = math.ceil(max(point[0] for point in footprint) + 0.5)
    min_y = math.floor(min(point[1] for point in footprint) - 0.5)
    max_y = math.ceil(max(point[1] for point in footprint) + 0.5)
    dynamic_cells = dynamic_cells or set()
    for cell_y in range(min_y, max_y + 1):
        for cell_x in range(min_x, max_x + 1):
            occupied = (
                not (0 <= cell_x < len(grid[0]) and 0 <= cell_y < len(grid))
                or grid[cell_y][cell_x] == 1
                or (cell_x, cell_y) in dynamic_cells
            )
            if not occupied:
                continue
            cell = [
                (cell_x - 0.5, cell_y - 0.5), (cell_x + 0.5, cell_y - 0.5),
                (cell_x + 0.5, cell_y + 0.5), (cell_x - 0.5, cell_y + 0.5),
            ]
            if polygons_overlap(footprint, cell):
                return True
    return False


def validate_oriented_path(grid, path, robot, dynamic_obstacles=None):
    if not path:
        return False, None
    pose_index = 0
    for start, goal in zip(path, path[1:]):
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        theta = math.atan2(dy, dx)
        steps = max(1, int(math.ceil(math.hypot(dx, dy) * 2)))
        for step in range(steps + 1):
            ratio = step / steps
            pose = (start[0] + dx * ratio, start[1] + dy * ratio, theta)
            dynamic_cells = dynamic_obstacles(pose_index) if dynamic_obstacles else set()
            if rectangle_pose_collision(grid, pose, robot, dynamic_cells):
                return False, pose
            pose_index += 1
    return True, None


def observe_local_grid(truth_grid, belief_grid, center, radius, rng, noise_rate=0.0):
    cx, cy = center
    for y in range(max(0, cy - radius), min(len(truth_grid), cy + radius + 1)):
        for x in range(max(0, cx - radius), min(len(truth_grid[0]), cx + radius + 1)):
            if math.hypot(x - cx, y - cy) > radius:
                continue
            value = truth_grid[y][x]
            if rng.random() < noise_rate:
                value = 1 - value
            belief_grid[y][x] = value


def optimistic_grid(belief_grid):
    return [[0 if cell == -1 else cell for cell in row] for row in belief_grid]


def merge_observations(prior_grid, belief_grid):
    return [
        [
            prior_grid[y][x] if belief_grid[y][x] == -1 else belief_grid[y][x]
            for x in range(len(prior_grid[0]))
        ]
        for y in range(len(prior_grid))
    ]


def make_realistic_scene(width=64, height=30):
    observed = [[' '] * width for _ in range(height)]
    for x in range(width):
        observed[0][x] = '#'
        observed[-1][x] = '#'
    for y in range(height):
        observed[y][0] = '#'
        observed[y][-1] = '#'
    for y in range(1, height - 1):
        if not 6 <= y <= 10 and not 16 <= y <= 26:
            observed[y][31] = '#'
    observed[8][6] = 'S'
    observed[8][57] = 'G'
    truth = [row[:] for row in observed]
    for y in range(6, 11):
        truth[y][31] = '#'
    failed_region = {(x, y) for x in range(27, 36) for y in range(5, 12)}
    return [''.join(row) for row in observed], [''.join(row) for row in truth], failed_region


def run_realistic_suite():
    observed_lines, truth_lines, failed_region = make_realistic_scene()
    observed_grid, start, goal = build_scene(observed_lines)
    truth_grid, _, _ = build_scene(truth_lines)
    rectangle_robot = RobotFootprint(body_width=2, body_length=4, leg_margin=0, sensor_margin=0, safe_margin=0.25)
    belief_grid = [[-1] * len(truth_grid[0]) for _ in truth_grid]
    rng = random.Random(7)
    observe_local_grid(truth_grid, belief_grid, start, radius=7, rng=rng, noise_rate=0.02)
    local_grid = merge_observations(observed_grid, belief_grid)
    planning_benchmark = GridBenchmark(local_grid, start, goal, rectangle_robot)
    baseline_path = AStarPlanner().plan(planning_benchmark)
    baseline_static, baseline_collision = validate_oriented_path(truth_grid, baseline_path, rectangle_robot)

    if baseline_collision:
        collision_cell = (round(baseline_collision[0]), round(baseline_collision[1]))
        observe_local_grid(truth_grid, belief_grid, collision_cell, radius=5, rng=rng, noise_rate=0.02)
    local_grid = merge_observations(observed_grid, belief_grid)
    planning_benchmark = GridBenchmark(local_grid, start, goal, rectangle_robot)
    memory_planner = OursPlanner(use_clearance_penalty=False)
    memory_planner.remember_failed_region(failed_region)
    memory_path = memory_planner.plan(planning_benchmark)
    memory_static, _ = validate_oriented_path(truth_grid, memory_path, rectangle_robot)

    def moving_obstacle(pose_index):
        if 75 <= pose_index <= 140:
            return {(39, 16), (40, 16)}
        return set()

    memory_dynamic, dynamic_collision = validate_oriented_path(
        truth_grid, memory_path, rectangle_robot, moving_obstacle)

    observed_cells = sum(cell != -1 for row in belief_grid for cell in row)
    noisy_cells = sum(
        cell != -1 and cell != truth_grid[y][x]
        for y, row in enumerate(belief_grid)
        for x, cell in enumerate(row)
    )

    rows = [
        {
            'Variant': 'A* optimistic local map',
            'Planned': bool(baseline_path),
            'RectangleStaticExecutable': baseline_static,
            'DynamicExecutable': baseline_static,
            'PathLength': round(path_length(baseline_path), 3),
            'ObservedCells': observed_cells,
            'NoisyObservedCells': noisy_cells,
        },
        {
            'Variant': 'Memory-aware rectangular footprint',
            'Planned': bool(memory_path),
            'RectangleStaticExecutable': memory_static,
            'DynamicExecutable': memory_dynamic,
            'PathLength': round(path_length(memory_path), 3),
            'ObservedCells': observed_cells,
            'NoisyObservedCells': noisy_cells,
        },
    ]
    write_dict_rows('results_realistic_suite.csv', rows)
    return {
        'rows': rows,
        'observed_grid': observed_grid,
        'truth_grid': truth_grid,
        'start': start,
        'goal': goal,
        'baseline_path': baseline_path,
        'memory_path': memory_path,
        'rectangle_robot': rectangle_robot,
        'dynamic_collision': dynamic_collision,
    }


class ShortTermMemory:
    def __init__(self, ttl=120):
        self.ttl = ttl
        self.cells = {}

    def remember(self, region):
        for cell in region:
            self.cells[cell] = self.ttl

    def decay(self, amount=1):
        for cell in list(self.cells):
            self.cells[cell] -= amount
            if self.cells[cell] <= 0:
                del self.cells[cell]

    def active_cells(self):
        return set(self.cells)


def make_dynamic_replanning_scene(width=64, height=34):
    grid = [[' '] * width for _ in range(height)]
    for x in range(width):
        grid[0][x] = '#'
        grid[-1][x] = '#'
    for y in range(height):
        grid[y][0] = '#'
        grid[y][-1] = '#'
    for x in range(20, 45):
        for y in range(10, 23):
            grid[y][x] = '#'
    grid[7][5] = 'S'
    grid[7][58] = 'G'
    return [''.join(row) for row in grid]


def dynamic_event(seed, mission):
    rng = random.Random(seed * 1009 + mission * 97)
    x = rng.choice([26, 30, 34, 38])
    expected_arrival = x - 5
    start_time = max(1, expected_arrival - rng.randint(5, 8))
    duration = rng.randint(2, 45)
    cells = {(x + dx, y) for dx in range(-1, 2) for y in range(4, 10)}
    closure = {(x, y) for y in range(1, 11)}
    memory_region = {(x + dx, y) for dx in range(-5, 6) for y in range(1, 12)}
    return {
        'x': x,
        'start': start_time,
        'end': start_time + duration,
        'duration': duration,
        'cells': cells,
        'closure': closure,
        'memory_region': memory_region,
    }


def plan_dynamic_route(grid, start, goal, robot, memory=None, transient_cells=None):
    benchmark = GridBenchmark(grid, start, goal, robot)
    for x, y in transient_cells or set():
        if 0 <= x < len(benchmark.inflated[0]) and 0 <= y < len(benchmark.inflated):
            benchmark.inflated[y][x] = 1
    planner = OursPlanner(use_clearance_penalty=False, memory_penalty=100.0)
    if memory:
        planner.remember_failed_region(memory.active_cells())
    return planner.plan(benchmark)


class SpaceTimeAStarPlanner:
    def __init__(
            self, max_time=120, dynamic_penalty=500.0,
            memory_penalty=80.0, wait_penalty=0.35,
            dynamic_memory_penalty=4.0, max_expansions=50000):
        self.name = 'Space-Time A*'
        self.max_time = max_time
        self.dynamic_penalty = dynamic_penalty
        self.memory_penalty = memory_penalty
        self.wait_penalty = wait_penalty
        self.dynamic_memory_penalty = dynamic_memory_penalty
        self.max_expansions = max_expansions
        self.last_stats = {
            'ExpandedStates': 0,
            'GeneratedStates': 0,
            'PlanningTime': 0.0,
            'ReachedGoal': False,
        }
        self.moves = [
            (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (1, -1), (-1, 1), (1, 1),
        ]

    def plan(
            self, grid, start, goal, robot, dynamic_cells_at_time,
            static_memory_cells=None, dynamic_memory_cells=None,
            start_time=0, benchmark=None):
        planning_start = time.perf_counter()
        self.last_stats = {
            'ExpandedStates': 0,
            'GeneratedStates': 0,
            'PlanningTime': 0.0,
            'ReachedGoal': False,
        }
        benchmark = benchmark or GridBenchmark(grid, start, goal, robot)
        static_memory_cells = static_memory_cells or set()
        dynamic_memory_cells = dynamic_memory_cells or set()
        if not benchmark.is_free(*start) or not benchmark.is_free(*goal):
            self.last_stats['PlanningTime'] = time.perf_counter() - planning_start
            return None
        start_state = (start[0], start[1], start_time)
        open_set = [(self.heuristic(start, goal), 0, start_state)]
        push_count = 1
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_state] = 0.0
        closed = set()
        while open_set:
            _, _, current = heapq.heappop(open_set)
            x, y, t = current
            if (x, y) == goal:
                self.last_stats['ReachedGoal'] = True
                self.last_stats['PlanningTime'] = time.perf_counter() - planning_start
                return self.reconstruct(came_from, current)
            if current in closed or t >= start_time + self.max_time:
                continue
            closed.add(current)
            self.last_stats['ExpandedStates'] += 1
            if self.last_stats['ExpandedStates'] >= self.max_expansions:
                break
            for dx, dy in self.moves:
                nx, ny = x + dx, y + dy
                nt = t + 1
                if not benchmark.is_free(nx, ny):
                    continue
                if dx and dy and (
                        not benchmark.is_free(nx, y)
                        or not benchmark.is_free(x, ny)):
                    continue
                if (nx, ny) in dynamic_cells_at_time(nt):
                    continue
                step_cost = math.hypot(dx, dy) if dx or dy else self.wait_penalty
                memory_cost = 0.0
                if (nx, ny) in static_memory_cells:
                    memory_cost += self.memory_penalty
                if (nx, ny) in dynamic_memory_cells:
                    memory_cost += self.dynamic_memory_penalty
                nxt = (nx, ny, nt)
                tentative_g = g_score[current] + step_cost + memory_cost
                if tentative_g < g_score[nxt]:
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    f = tentative_g + self.heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f, push_count, nxt))
                    push_count += 1
                    self.last_stats['GeneratedStates'] += 1
        self.last_stats['PlanningTime'] = time.perf_counter() - planning_start
        return None

    def heuristic(self, point, goal):
        return math.hypot(point[0] - goal[0], point[1] - goal[1])

    def reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))


def space_time_path_metrics(path):
    if not path:
        return {
            'Success': False,
            'WaitSteps': 0,
            'TimeSteps': 0,
            'PathLength': 0.0,
        }
    waits = 0
    distance = 0.0
    for current, nxt in zip(path, path[1:]):
        if current[:2] == nxt[:2]:
            waits += 1
        distance += math.hypot(nxt[0] - current[0], nxt[1] - current[1])
    return {
        'Success': True,
        'WaitSteps': waits,
        'TimeSteps': path[-1][2],
        'PathLength': round(distance, 3),
    }


def make_dynamic_cells_function(obstacles, padding=0.5):
    def dynamic_cells_at_time(time_step):
        cells = set()
        for obstacle in obstacles:
            cells.update(obstacle.occupied_cells_at(time_step, padding=padding))
        return cells
    return dynamic_cells_at_time


def validate_space_time_path(path, grid, robot, dynamic_cells_at_time):
    if not path:
        return False
    benchmark = GridBenchmark(grid, path[0][:2], path[-1][:2], robot)
    for x, y, time_step in path:
        if not benchmark.is_free(x, y):
            return False
        if (x, y) in dynamic_cells_at_time(time_step):
            return False
    return True


def point_to_obstacle_distance(point, obstacle, time_step):
    obstacle_x, obstacle_y = obstacle.position_at(time_step)
    return math.hypot(point[0] - obstacle_x, point[1] - obstacle_y) - obstacle.radius


def predicted_cells_from_trajectory(predictions, default_radius=1.6):
    cells_by_time = defaultdict(set)
    risk_cells = set()
    for prediction in predictions:
        time_step, predicted_x, predicted_y, radius = prediction_values(
            prediction, default_radius=default_radius)
        cells = cells_near_position((predicted_x, predicted_y), radius=radius)
        cells_by_time[time_step].update(cells)
        risk_cells.update(cells)
    return cells_by_time, risk_cells


def simulate_space_time_prediction_mission(
        grid, start, goal, robot, obstacle, predictor,
        static_memory_cells=None, dynamic_memory=None, max_steps=180,
        replan_interval=4):
    planner = SpaceTimeAStarPlanner(max_time=80)
    planning_benchmark = GridBenchmark(grid, start, goal, robot)
    current = start
    time_step = 0
    distance = 0.0
    waits = 0
    replans = 0
    prediction_errors = []
    expanded_states = 0
    generated_states = 0
    planning_times = []
    collision_count = 0
    min_dynamic_clearance = float('inf')
    route = None
    executed_path = [current]
    static_memory_cells = static_memory_cells or set()
    dynamic_memory = dynamic_memory or ShortTermMemory(ttl=16)
    while current != goal and time_step < max_steps:
        dynamic_memory.decay()
        true_position = obstacle.position_at(time_step)
        predictor.observe(true_position, time_step)
        predictions = predictor.predict()
        if predictions:
            next_true = obstacle.position_at(time_step + 1)
            _, predicted_x, predicted_y, _ = prediction_values(
                predictions[0], default_radius=1.6)
            prediction_errors.append(math.hypot(
                predicted_x - next_true[0], predicted_y - next_true[1]))
        predicted_cells_by_time, predicted_risk_cells = predicted_cells_from_trajectory(
            predictions)
        if predicted_risk_cells:
            dynamic_memory.remember(predicted_risk_cells)

        def dynamic_cells_at_time(query_time):
            if query_time == time_step:
                return obstacle.occupied_cells_at(query_time, padding=0.5)
            return predicted_cells_by_time.get(query_time, set())

        needs_replan = (
            route is None
            or len(route) < 2
            or route[0][:2] != current
            or route[0][2] != time_step
            or time_step % replan_interval == 0
        )
        if not needs_replan and len(route) >= 2:
            candidate_next = route[1]
            if candidate_next[:2] in dynamic_cells_at_time(candidate_next[2]):
                needs_replan = True
        if needs_replan:
            route = planner.plan(
                grid, current, goal, robot, dynamic_cells_at_time,
                static_memory_cells=static_memory_cells,
                dynamic_memory_cells=dynamic_memory.active_cells(),
                start_time=time_step, benchmark=planning_benchmark)
            if route is None and dynamic_memory.active_cells():
                expanded_states += planner.last_stats['ExpandedStates']
                generated_states += planner.last_stats['GeneratedStates']
                planning_times.append(planner.last_stats['PlanningTime'])
                route = planner.plan(
                    grid, current, goal, robot, dynamic_cells_at_time,
                    static_memory_cells=static_memory_cells,
                    dynamic_memory_cells=set(),
                    start_time=time_step, benchmark=planning_benchmark)
            replans += 1
            expanded_states += planner.last_stats['ExpandedStates']
            generated_states += planner.last_stats['GeneratedStates']
            planning_times.append(planner.last_stats['PlanningTime'])
        if not route or len(route) < 2:
            waits += 1
            time_step += 1
            continue
        nxt = route[1]
        actual_next_cells = obstacle.occupied_cells_at(nxt[2], padding=0.5)
        min_dynamic_clearance = min(
            min_dynamic_clearance,
            point_to_obstacle_distance(nxt[:2], obstacle, nxt[2]))
        if nxt[:2] in actual_next_cells:
            collision_count += 1
            return {
                'Success': False,
                'Collision': True,
                'CollisionCount': collision_count,
                'Replans': replans,
                'WaitSteps': waits,
                'TimeSteps': time_step,
                'PathLength': round(distance, 3),
                'PredictionMAE': round(statistics.fmean(prediction_errors), 4) if prediction_errors else 0.0,
                'ExpandedStates': expanded_states,
                'GeneratedStates': generated_states,
                'MeanPlanningTime': round(statistics.fmean(planning_times), 6) if planning_times else 0.0,
                'MaxPlanningTime': round(max(planning_times), 6) if planning_times else 0.0,
                'MinDynamicClearance': round(min_dynamic_clearance, 4),
                'TotalCost': round(distance + waits * 3 + replans * 2 + max_steps, 3),
                'ExecutedPath': executed_path,
            }
        if nxt[:2] == current:
            waits += 1
        distance += math.hypot(nxt[0] - current[0], nxt[1] - current[1])
        current = nxt[:2]
        executed_path.append(current)
        time_step = nxt[2]
        route = route[1:]
    success = current == goal
    total_cost = distance + waits * 3 + replans * 2 + (0 if success else max_steps)
    return {
        'Success': success,
        'Collision': False,
        'CollisionCount': collision_count,
        'Replans': replans,
        'WaitSteps': waits,
        'TimeSteps': time_step,
        'PathLength': round(distance, 3),
        'PredictionMAE': round(statistics.fmean(prediction_errors), 4) if prediction_errors else 0.0,
        'ExpandedStates': expanded_states,
        'GeneratedStates': generated_states,
        'MeanPlanningTime': round(statistics.fmean(planning_times), 6) if planning_times else 0.0,
        'MaxPlanningTime': round(max(planning_times), 6) if planning_times else 0.0,
        'MinDynamicClearance': round(min_dynamic_clearance, 4) if min_dynamic_clearance < float('inf') else 0.0,
        'TotalCost': round(total_cost, 3),
        'ExecutedPath': executed_path,
    }


def simulate_dynamic_mission(
        strategy, grid, start, goal, planning_robot, rectangle_robot, event,
        memory=None, wait_threshold=5, max_steps=240, max_replans=8):
    path = plan_dynamic_route(grid, start, goal, planning_robot, memory=memory)
    if not path:
        return {'Success': False, 'Replans': 0, 'WaitSteps': 0, 'PathLength': 0.0, 'TotalCost': max_steps}

    current = start
    route = path
    time_step = 0
    replans = 0
    wait_steps = 0
    distance = 0.0
    while current != goal and time_step < max_steps:
        if memory:
            memory.decay()
        active_cells = event['cells'] if event['start'] <= time_step < event['end'] else set()
        if len(route) < 2 or route[0] != current:
            route = plan_dynamic_route(grid, current, goal, planning_robot, memory=memory)
            if not route:
                break
        nxt = route[1]
        collision_ahead = False
        for index in range(min(4, len(route) - 1)):
            segment_clear, _ = validate_oriented_path(
                grid, [route[index], route[index + 1]], rectangle_robot,
                lambda _: active_cells)
            if not segment_clear:
                collision_ahead = True
                break
        if not collision_ahead:
            distance += math.hypot(nxt[0] - current[0], nxt[1] - current[1])
            current = nxt
            route = route[1:]
            time_step += 1
            continue

        if strategy == 'Static plan':
            break

        remaining = max(0, event['end'] - time_step)
        should_wait = strategy == 'Wait-only' or (strategy == 'Memory-aware wait-or-reroute' and remaining <= wait_threshold)
        if should_wait:
            wait_steps += 1
            time_step += 1
            continue

        replans += 1
        if replans > max_replans:
            break
        if strategy == 'Memory-aware wait-or-reroute' and memory:
            memory.remember(event['memory_region'])
        route = plan_dynamic_route(
            grid, current, goal, planning_robot, memory=memory,
            transient_cells=event['closure'])
        if not route:
            break

    success = current == goal
    total_cost = distance + wait_steps * 3 + replans * 5 + (0 if success else max_steps)
    return {
        'Success': success,
        'Replans': replans,
        'WaitSteps': wait_steps,
        'PathLength': round(distance, 3),
        'TotalCost': round(total_cost, 3),
    }


def run_dynamic_replanning_statistics(seeds=100, missions_per_seed=3):
    lines = make_dynamic_replanning_scene()
    grid, start, goal = build_scene(lines)
    planning_robot = RobotFootprint(body_width=2, body_length=4, leg_margin=0, sensor_margin=0, safe_margin=0.25)
    rectangle_robot = RobotFootprint(body_width=2, body_length=4, leg_margin=0, sensor_margin=0, safe_margin=0.25)
    strategies = ['Static plan', 'Wait-only', 'Reactive reroute', 'Memory-aware wait-or-reroute']
    rows = []
    for strategy in strategies:
        for seed in range(seeds):
            memory = ShortTermMemory(ttl=120) if strategy == 'Memory-aware wait-or-reroute' else None
            for mission in range(missions_per_seed):
                event = dynamic_event(seed, mission)
                result = simulate_dynamic_mission(
                    strategy, grid, start, goal, planning_robot, rectangle_robot,
                    event, memory=memory)
                rows.append({
                    'Strategy': strategy,
                    'Seed': seed,
                    'Mission': mission,
                    'ObstacleX': event['x'],
                    'ObstacleStart': event['start'],
                    'ObstacleDuration': event['duration'],
                    **result,
                })
    write_dict_rows('results_dynamic_trials.csv', rows)

    summary_rows = []
    for strategy in strategies:
        selected = [row for row in rows if row['Strategy'] == strategy]
        successes = sum(row['Success'] for row in selected)
        ci_low, ci_high = wilson_ci95(successes, len(selected))
        summary_rows.append({
            'Strategy': strategy,
            'Trials': len(selected),
            'Successes': successes,
            'SuccessRate': round(successes / len(selected), 4),
            'SuccessCI95Low': round(ci_low, 4),
            'SuccessCI95High': round(ci_high, 4),
            'MeanReplans': round(statistics.fmean(row['Replans'] for row in selected), 4),
            'MeanWaitSteps': round(statistics.fmean(row['WaitSteps'] for row in selected), 4),
            'MeanPathLength': round(statistics.fmean(row['PathLength'] for row in selected), 4),
            'MeanTotalCost': round(statistics.fmean(row['TotalCost'] for row in selected), 4),
        })
    write_dict_rows('results_dynamic_summary.csv', summary_rows)
    return summary_rows


class MovingObstacle:
    def __init__(
            self, start_x, start_y, velocity_x, velocity_y,
            acceleration_x=0.0, acceleration_y=0.0, radius=1.0):
        self.start_x = start_x
        self.start_y = start_y
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.acceleration_x = acceleration_x
        self.acceleration_y = acceleration_y
        self.radius = radius

    def position_at(self, time_step):
        return (
            self.start_x + self.velocity_x * time_step + 0.5 * self.acceleration_x * time_step * time_step,
            self.start_y + self.velocity_y * time_step + 0.5 * self.acceleration_y * time_step * time_step,
        )

    def occupied_cells_at(self, time_step, padding=0.0):
        x, y = self.position_at(time_step)
        radius = self.radius + padding
        cells = set()
        for cell_y in range(math.floor(y - radius), math.ceil(y + radius) + 1):
            for cell_x in range(math.floor(x - radius), math.ceil(x + radius) + 1):
                if math.hypot(cell_x - x, cell_y - y) <= radius:
                    cells.add((cell_x, cell_y))
        return cells


class PiecewiseMovingObstacle:
    def __init__(self, position_fn, radius=1.0):
        self.position_fn = position_fn
        self.radius = radius

    def position_at(self, time_step):
        return self.position_fn(time_step)

    def occupied_cells_at(self, time_step, padding=0.0):
        return cells_near_position(self.position_at(time_step), self.radius + padding)


class OnlineTrajectoryPredictor:
    def __init__(self, horizon=10, position_noise=0.75, velocity_noise=0.12, rng=None):
        self.horizon = horizon
        self.position_noise = position_noise
        self.velocity_noise = velocity_noise
        self.rng = rng or random.Random(0)
        self.observations = []

    def observe(self, position, time_step):
        noisy = (
            position[0] + self.rng.gauss(0, self.position_noise),
            position[1] + self.rng.gauss(0, self.position_noise),
        )
        self.observations.append((time_step, noisy))
        self.observations = self.observations[-4:]
        return noisy

    def estimate_velocity(self):
        if len(self.observations) < 2:
            return 0.0, 0.0
        start_time, start = self.observations[0]
        end_time, end = self.observations[-1]
        dt = max(1, end_time - start_time)
        return (
            (end[0] - start[0]) / dt + self.rng.gauss(0, self.velocity_noise),
            (end[1] - start[1]) / dt + self.rng.gauss(0, self.velocity_noise),
        )

    def predict(self):
        if len(self.observations) < 2:
            return []
        time_step, position = self.observations[-1]
        velocity = self.estimate_velocity()
        return [
            (
                time_step + delta,
                position[0] + velocity[0] * delta,
                position[1] + velocity[1] * delta,
            )
            for delta in range(1, self.horizon + 1)
        ]


class KalmanTrajectoryPredictor:
    def __init__(
            self, horizon=10, position_noise=0.75, process_noise=0.08,
            uncertainty_scale=None, base_risk_radius=1.0, max_risk_radius=2.4, rng=None):
        self.horizon = horizon
        self.position_noise = position_noise
        self.process_noise = process_noise
        self.uncertainty_scale = uncertainty_scale
        self.base_risk_radius = base_risk_radius
        self.max_risk_radius = max_risk_radius
        self.rng = rng or random.Random(0)
        self.state = None
        self.covariance = np.eye(6) * 10.0
        self.last_time = None
        self.observation_count = 0
        self.previous_measurement = None

    def transition_matrix(self, dt):
        return np.array([
            [1, 0, dt, 0, 0.5 * dt * dt, 0],
            [0, 1, 0, dt, 0, 0.5 * dt * dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)

    def observe(self, position, time_step):
        measurement = np.array([
            position[0] + self.rng.gauss(0, self.position_noise),
            position[1] + self.rng.gauss(0, self.position_noise),
        ], dtype=float)
        if self.state is None:
            self.state = np.array([measurement[0], measurement[1], 0, 0, 0, 0], dtype=float)
            self.last_time = time_step
            self.observation_count = 1
            self.previous_measurement = measurement
            return tuple(measurement)

        dt = max(1, time_step - self.last_time)
        if self.observation_count == 1:
            velocity = (measurement - self.previous_measurement) / dt
            self.state = np.array([
                measurement[0], measurement[1], velocity[0], velocity[1], 0, 0,
            ], dtype=float)
            self.covariance = np.diag([1, 1, 2, 2, 0.5, 0.5]).astype(float)
            self.last_time = time_step
            self.observation_count = 2
            self.previous_measurement = measurement
            return tuple(measurement)
        transition = self.transition_matrix(dt)
        process_covariance = np.eye(6) * self.process_noise
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_covariance
        observation = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ], dtype=float)
        observation_covariance = np.eye(2) * max(1e-6, self.position_noise ** 2)
        innovation = measurement - observation @ self.state
        innovation_covariance = observation @ self.covariance @ observation.T + observation_covariance
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(6) - gain @ observation) @ self.covariance
        self.last_time = time_step
        self.observation_count += 1
        self.previous_measurement = measurement
        return tuple(measurement)

    def predict(self):
        if self.observation_count < 2:
            return []
        predictions = []
        predicted_state = self.state.copy()
        predicted_covariance = self.covariance.copy()
        transition = self.transition_matrix(1)
        for delta in range(1, self.horizon + 1):
            predicted_state = transition @ predicted_state
            predicted_covariance = (
                transition @ predicted_covariance @ transition.T
                + np.eye(6) * self.process_noise
            )
            if self.uncertainty_scale is None:
                predictions.append((self.last_time + delta, predicted_state[0], predicted_state[1]))
                continue
            position_covariance = predicted_covariance[:2, :2]
            sigma = math.sqrt(max(np.linalg.eigvalsh(position_covariance)))
            radius = min(
                self.max_risk_radius,
                self.base_risk_radius + self.uncertainty_scale * sigma,
            )
            predictions.append((
                self.last_time + delta, predicted_state[0], predicted_state[1], radius,
            ))
        return predictions


def prediction_values(prediction, default_radius):
    if len(prediction) == 4:
        time_step, x, y, radius = prediction
        return time_step, x, y, radius
    time_step, x, y = prediction
    return time_step, x, y, default_radius


def cells_near_position(position, radius):
    x, y = position
    cells = set()
    for cell_y in range(math.floor(y - radius), math.ceil(y + radius) + 1):
        for cell_x in range(math.floor(x - radius), math.ceil(x + radius) + 1):
            if math.hypot(cell_x - x, cell_y - y) <= radius:
                cells.add((cell_x, cell_y))
    return cells


def moving_obstacle_scenario(seed, mission):
    rng = random.Random(seed * 1613 + mission * 211)
    crossing_x = rng.choice([27, 31, 35, 39])
    speed = rng.uniform(0.42, 0.72)
    acceleration = rng.uniform(-0.012, 0.012)
    crossing_time = crossing_x - 5 + rng.randint(-2, 3)
    start_y = 7 - speed * crossing_time - 0.5 * acceleration * crossing_time * crossing_time
    obstacle = MovingObstacle(
        crossing_x, start_y, 0.0, speed, acceleration_y=acceleration, radius=1.0)
    return obstacle


def nonlinear_moving_obstacle_scenario(seed, mission, trajectory_type):
    rng = random.Random(seed * 2017 + mission * 223 + len(trajectory_type) * 29)
    crossing_x = rng.choice([27, 31, 35, 39])
    speed = rng.uniform(0.42, 0.68)
    crossing_time = crossing_x - 5 + rng.randint(-2, 3)
    if trajectory_type == 'turn':
        turn_time = max(2, crossing_time - rng.randint(4, 7))
        horizontal_speed = rng.choice([-0.18, 0.18])
        start_y = 7 - speed * crossing_time

        def position_fn(time_step):
            x = crossing_x
            y = start_y + speed * time_step
            if time_step > turn_time:
                x += horizontal_speed * (time_step - turn_time)
            return x, y

    elif trajectory_type == 'sudden-acceleration':
        acceleration_time = max(2, crossing_time - rng.randint(4, 7))
        faster_speed = speed * rng.uniform(1.5, 2.1)
        start_y = 7 - speed * acceleration_time - faster_speed * (crossing_time - acceleration_time)

        def position_fn(time_step):
            if time_step <= acceleration_time:
                return crossing_x, start_y + speed * time_step
            return crossing_x, (
                start_y + speed * acceleration_time
                + faster_speed * (time_step - acceleration_time)
            )

    elif trajectory_type == 'stop-and-go':
        stop_start = max(2, crossing_time - rng.randint(5, 8))
        stop_duration = rng.randint(3, 7)
        start_y = 7 - speed * (crossing_time - stop_duration)

        def position_fn(time_step):
            if time_step <= stop_start:
                return crossing_x, start_y + speed * time_step
            if time_step <= stop_start + stop_duration:
                return crossing_x, start_y + speed * stop_start
            return crossing_x, (
                start_y + speed * stop_start
                + speed * (time_step - stop_start - stop_duration)
            )

    else:
        raise ValueError(f'unknown trajectory type: {trajectory_type}')
    return PiecewiseMovingObstacle(position_fn, radius=1.0)


def route_prediction_conflicts(
        route, predictions, rectangle_robot, route_delay=0, lookahead=10):
    if not route:
        return []
    conflicts = []
    route_segments = list(zip(route, route[1:]))[:lookahead]
    for route_index, (start, goal) in enumerate(route_segments):
        prediction_index = route_index + route_delay
        if prediction_index >= len(predictions):
            break
        prediction = predictions[prediction_index]
        _, predicted_x, predicted_y, radius = prediction_values(prediction, default_radius=1.6)
        cells = cells_near_position((predicted_x, predicted_y), radius=radius)
        clear, _ = validate_oriented_path(
            [[0] * 64 for _ in range(34)], [start, goal], rectangle_robot,
            lambda _: cells)
        if not clear:
            conflicts.append((prediction_index, prediction, cells))
    return conflicts


def route_conflicts_with_prediction(
        route, predictions, rectangle_robot, route_delay=0, lookahead=10):
    return bool(route_prediction_conflicts(
        route, predictions, rectangle_robot,
        route_delay=route_delay, lookahead=lookahead))


def choose_prediction_response(
        grid, current, goal, route, predictions, planning_robot,
        rectangle_robot, memory=None, max_wait_steps=4):
    conflicts = route_prediction_conflicts(route, predictions, rectangle_robot)
    conflict_cells = set()
    for _, _, cells in conflicts:
        conflict_cells.update(cells)
    if not conflicts:
        return 'continue', route, conflict_cells, 0

    wait_steps = None
    for delay in range(1, max_wait_steps + 1):
        if not route_conflicts_with_prediction(
                route, predictions, rectangle_robot, route_delay=delay):
            wait_steps = delay
            break

    reroute = plan_dynamic_route(
        grid, current, goal, planning_robot, memory=memory,
        transient_cells=conflict_cells)
    reroute_cost = (
        path_length(reroute) + 5 if reroute
        else float('inf')
    )
    wait_cost = (
        path_length(route) + wait_steps * 3 if wait_steps is not None
        else float('inf')
    )
    if wait_cost <= reroute_cost:
        return 'wait', route, conflict_cells, wait_steps
    if reroute:
        return 'reroute', reroute, conflict_cells, 0
    if wait_steps is not None:
        return 'wait', route, conflict_cells, wait_steps
    return 'reroute', route, conflict_cells, 0


def simulate_trajectory_prediction_mission(
        strategy, grid, start, goal, planning_robot, rectangle_robot, obstacle,
        predictor, memory=None, max_steps=240, decision_policy='legacy'):
    route = plan_dynamic_route(grid, start, goal, planning_robot, memory=memory)
    current = start
    time_step = 0
    distance = 0.0
    waits = 0
    replans = 0
    decision_waits = 0
    decision_reroutes = 0
    prediction_errors = []
    risk_radii = []
    while route and current != goal and time_step < max_steps:
        if memory:
            memory.decay()
        true_position = obstacle.position_at(time_step)
        predictor.observe(true_position, time_step)
        predictions = predictor.predict()
        if predictions:
            next_true = obstacle.position_at(time_step + 1)
            _, predicted_x, predicted_y, radius = prediction_values(predictions[0], default_radius=1.6)
            prediction_errors.append(math.hypot(predicted_x - next_true[0], predicted_y - next_true[1]))
            risk_radii.append(radius)
        actual_cells = obstacle.occupied_cells_at(time_step, padding=0.5)
        collision_ahead = False
        for index in range(min(5, len(route) - 1)):
            clear, _ = validate_oriented_path(
                grid, [route[index], route[index + 1]], rectangle_robot,
                lambda _: actual_cells)
            if not clear:
                collision_ahead = True
                break
        predicted_conflict = (
            strategy == 'Prediction-aware memory'
            and route_conflicts_with_prediction(route, predictions, rectangle_robot)
        )
        if collision_ahead or predicted_conflict:
            if strategy == 'Prediction-aware memory' and decision_policy == 'cost-aware':
                response, candidate_route, conflict_cells, wait_steps = choose_prediction_response(
                    grid, current, goal, route, predictions, planning_robot,
                    rectangle_robot, memory=memory)
                if response in {'continue', 'wait'}:
                    waits += 1
                    decision_waits += 1
                    time_step += 1
                    continue
                replans += 1
                decision_reroutes += 1
                if memory:
                    memory.remember(conflict_cells or actual_cells)
                route = candidate_route
                if not route:
                    time_step += 1
                    continue
                continue
            waits += 1
            if waits % 3 != 0 and not collision_ahead:
                time_step += 1
                continue
            replans += 1
            predicted_cells = set()
            for prediction in predictions:
                _, predicted_x, predicted_y, radius = prediction_values(prediction, default_radius=2.0)
                predicted_cells.update(cells_near_position((predicted_x, predicted_y), radius=radius))
            if strategy == 'Prediction-aware memory' and memory:
                memory.remember(predicted_cells)
            route = plan_dynamic_route(
                grid, current, goal, planning_robot, memory=memory,
                transient_cells=predicted_cells or actual_cells)
            if not route:
                time_step += 1
                continue
        if len(route) < 2:
            break
        nxt = route[1]
        distance += math.hypot(nxt[0] - current[0], nxt[1] - current[1])
        current = nxt
        route = route[1:]
        time_step += 1
    success = current == goal
    total_cost = distance + waits * 3 + replans * 5 + (0 if success else max_steps)
    return {
        'Success': success,
        'Replans': replans,
        'WaitSteps': waits,
        'DecisionWaits': decision_waits,
        'DecisionReroutes': decision_reroutes,
        'PathLength': round(distance, 3),
        'PredictionMAE': round(statistics.fmean(prediction_errors), 4) if prediction_errors else 0.0,
        'MeanRiskRadius': round(statistics.fmean(risk_radii), 4) if risk_radii else 0.0,
        'TotalCost': round(total_cost, 3),
    }


def run_trajectory_prediction_statistics(seeds=100, missions_per_seed=3):
    lines = make_dynamic_replanning_scene()
    grid, start, goal = build_scene(lines)
    robot = RobotFootprint(body_width=2, body_length=4, leg_margin=0, sensor_margin=0, safe_margin=0.25)
    strategies = [
        'Reactive current occupancy',
        'Prediction-aware memory',
        'Kalman prediction-aware memory',
        'Kalman uncertainty-aware memory',
    ]
    rows = []
    for strategy in strategies:
        for seed in range(seeds):
            memory = ShortTermMemory(ttl=40) if strategy != 'Reactive current occupancy' else None
            for mission in range(missions_per_seed):
                obstacle = moving_obstacle_scenario(seed, mission)
                predictor_rng = random.Random(seed * 313 + mission * 19)
                if strategy == 'Kalman uncertainty-aware memory':
                    predictor = KalmanTrajectoryPredictor(
                        horizon=10, position_noise=0.75, process_noise=0.03,
                        uncertainty_scale=0.02, base_risk_radius=1.5, max_risk_radius=2.4,
                        rng=predictor_rng)
                elif strategy == 'Kalman prediction-aware memory':
                    predictor = KalmanTrajectoryPredictor(
                        horizon=10, position_noise=0.75, process_noise=0.03,
                        rng=predictor_rng)
                else:
                    predictor = OnlineTrajectoryPredictor(
                        horizon=10, position_noise=0.75, velocity_noise=0.12,
                        rng=predictor_rng)
                result = simulate_trajectory_prediction_mission(
                    'Prediction-aware memory' if strategy != 'Reactive current occupancy' else strategy,
                    grid, start, goal, robot, robot, obstacle,
                    predictor, memory=memory)
                rows.append({
                    'Strategy': strategy,
                    'Seed': seed,
                    'Mission': mission,
                    **result,
                })
    write_dict_rows('results_trajectory_prediction_trials.csv', rows)

    summary_rows = []
    for strategy in strategies:
        selected = [row for row in rows if row['Strategy'] == strategy]
        successes = sum(row['Success'] for row in selected)
        ci_low, ci_high = wilson_ci95(successes, len(selected))
        summary_rows.append({
            'Strategy': strategy,
            'Trials': len(selected),
            'Successes': successes,
            'SuccessRate': round(successes / len(selected), 4),
            'SuccessCI95Low': round(ci_low, 4),
            'SuccessCI95High': round(ci_high, 4),
            'MeanReplans': round(statistics.fmean(row['Replans'] for row in selected), 4),
            'MeanWaitSteps': round(statistics.fmean(row['WaitSteps'] for row in selected), 4),
            'MeanPathLength': round(statistics.fmean(row['PathLength'] for row in selected), 4),
            'MeanPredictionMAE': round(statistics.fmean(row['PredictionMAE'] for row in selected), 4),
            'MeanRiskRadius': round(statistics.fmean(row['MeanRiskRadius'] for row in selected), 4),
            'MeanTotalCost': round(statistics.fmean(row['TotalCost'] for row in selected), 4),
        })
    write_dict_rows('results_trajectory_prediction_summary.csv', summary_rows)
    return summary_rows


def run_prediction_noise_ablation(seeds=20, missions_per_seed=3):
    lines = make_dynamic_replanning_scene()
    grid, start, goal = build_scene(lines)
    robot = RobotFootprint(body_width=2, body_length=4, leg_margin=0, sensor_margin=0, safe_margin=0.25)
    noise_levels = [0.25, 0.75, 1.5, 2.5]
    predictors = ['Constant velocity', 'Kalman acceleration']
    rows = []
    for predictor_name in predictors:
        for position_noise in noise_levels:
            for seed in range(seeds):
                memory = ShortTermMemory(ttl=40)
                for mission in range(missions_per_seed):
                    obstacle = moving_obstacle_scenario(seed, mission)
                    predictor_rng = random.Random(seed * 313 + mission * 19)
                    if predictor_name == 'Kalman acceleration':
                        predictor = KalmanTrajectoryPredictor(
                            horizon=10, position_noise=position_noise, process_noise=0.03,
                            rng=predictor_rng)
                    else:
                        predictor = OnlineTrajectoryPredictor(
                            horizon=10, position_noise=position_noise, velocity_noise=0.12,
                            rng=predictor_rng)
                    result = simulate_trajectory_prediction_mission(
                        'Prediction-aware memory', grid, start, goal, robot, robot,
                        obstacle, predictor, memory=memory)
                    rows.append({
                        'Predictor': predictor_name,
                        'PositionNoise': position_noise,
                        'Seed': seed,
                        'Mission': mission,
                        **result,
                    })
    write_dict_rows('results_prediction_noise_trials.csv', rows)

    summary_rows = []
    for predictor_name in predictors:
        for position_noise in noise_levels:
            selected = [
                row for row in rows
                if row['Predictor'] == predictor_name and row['PositionNoise'] == position_noise
            ]
            successes = sum(row['Success'] for row in selected)
            ci_low, ci_high = wilson_ci95(successes, len(selected))
            summary_rows.append({
                'Predictor': predictor_name,
                'PositionNoise': position_noise,
                'Trials': len(selected),
                'Successes': successes,
                'SuccessRate': round(successes / len(selected), 4),
                'SuccessCI95Low': round(ci_low, 4),
                'SuccessCI95High': round(ci_high, 4),
                'MeanPredictionMAE': round(statistics.fmean(row['PredictionMAE'] for row in selected), 4),
                'MeanRiskRadius': round(statistics.fmean(row['MeanRiskRadius'] for row in selected), 4),
                'MeanReplans': round(statistics.fmean(row['Replans'] for row in selected), 4),
                'MeanWaitSteps': round(statistics.fmean(row['WaitSteps'] for row in selected), 4),
                'MeanTotalCost': round(statistics.fmean(row['TotalCost'] for row in selected), 4),
            })
    write_dict_rows('results_prediction_noise_summary.csv', summary_rows)
    return summary_rows


def run_uncertainty_radius_search(seeds=10, missions_per_seed=3):
    lines = make_dynamic_replanning_scene()
    grid, start, goal = build_scene(lines)
    robot = RobotFootprint(body_width=2, body_length=4, leg_margin=0, sensor_margin=0, safe_margin=0.25)
    noise_levels = [0.75, 1.5, 2.5]
    scales = [0.01, 0.02, 0.05, 0.1, 0.15]
    rows = []
    for position_noise in noise_levels:
        for scale in scales:
            for seed in range(seeds):
                memory = ShortTermMemory(ttl=40)
                for mission in range(missions_per_seed):
                    obstacle = moving_obstacle_scenario(seed, mission)
                    predictor = KalmanTrajectoryPredictor(
                        horizon=10, position_noise=position_noise, process_noise=0.03,
                        uncertainty_scale=scale, base_risk_radius=1.5, max_risk_radius=2.4,
                        rng=random.Random(seed * 313 + mission * 19))
                    result = simulate_trajectory_prediction_mission(
                        'Prediction-aware memory', grid, start, goal, robot, robot,
                        obstacle, predictor, memory=memory)
                    rows.append({
                        'PositionNoise': position_noise,
                        'RiskScale': scale,
                        'Seed': seed,
                        'Mission': mission,
                        **result,
                    })
    write_dict_rows('results_uncertainty_radius_trials.csv', rows)

    summary_rows = []
    for position_noise in noise_levels:
        for scale in scales:
            selected = [
                row for row in rows
                if row['PositionNoise'] == position_noise and row['RiskScale'] == scale
            ]
            successes = sum(row['Success'] for row in selected)
            summary_rows.append({
                'PositionNoise': position_noise,
                'RiskScale': scale,
                'Trials': len(selected),
                'Successes': successes,
                'SuccessRate': round(successes / len(selected), 4),
                'MeanPredictionMAE': round(statistics.fmean(row['PredictionMAE'] for row in selected), 4),
                'MeanRiskRadius': round(statistics.fmean(row['MeanRiskRadius'] for row in selected), 4),
                'MeanReplans': round(statistics.fmean(row['Replans'] for row in selected), 4),
                'MeanWaitSteps': round(statistics.fmean(row['WaitSteps'] for row in selected), 4),
                'MeanTotalCost': round(statistics.fmean(row['TotalCost'] for row in selected), 4),
            })
    write_dict_rows('results_uncertainty_radius_summary.csv', summary_rows)
    return summary_rows


def run_nonlinear_trajectory_statistics(seeds=20, missions_per_seed=3):
    lines = make_dynamic_replanning_scene()
    grid, start, goal = build_scene(lines)
    robot = RobotFootprint(body_width=2, body_length=4, leg_margin=0, sensor_margin=0, safe_margin=0.25)
    trajectory_types = ['turn', 'sudden-acceleration', 'stop-and-go']
    strategies = [
        'Reactive current occupancy',
        'Constant velocity memory',
        'Kalman fixed-radius memory',
        'Kalman uncertainty-aware memory',
    ]
    rows = []
    for trajectory_type in trajectory_types:
        for strategy in strategies:
            for seed in range(seeds):
                memory = ShortTermMemory(ttl=40) if strategy != 'Reactive current occupancy' else None
                for mission in range(missions_per_seed):
                    obstacle = nonlinear_moving_obstacle_scenario(seed, mission, trajectory_type)
                    predictor_rng = random.Random(
                        seed * 313 + mission * 19
                        + trajectory_types.index(trajectory_type) * 100003)
                    if strategy == 'Kalman uncertainty-aware memory':
                        predictor = KalmanTrajectoryPredictor(
                            horizon=10, position_noise=0.75, process_noise=0.03,
                            uncertainty_scale=0.02, base_risk_radius=1.5, max_risk_radius=2.4,
                            rng=predictor_rng)
                    elif strategy == 'Kalman fixed-radius memory':
                        predictor = KalmanTrajectoryPredictor(
                            horizon=10, position_noise=0.75, process_noise=0.03,
                            rng=predictor_rng)
                    else:
                        predictor = OnlineTrajectoryPredictor(
                            horizon=10, position_noise=0.75, velocity_noise=0.12,
                            rng=predictor_rng)
                    result = simulate_trajectory_prediction_mission(
                        'Prediction-aware memory' if strategy != 'Reactive current occupancy' else strategy,
                        grid, start, goal, robot, robot, obstacle,
                        predictor, memory=memory)
                    rows.append({
                        'TrajectoryType': trajectory_type,
                        'Strategy': strategy,
                        'Seed': seed,
                        'Mission': mission,
                        **result,
                    })
    write_dict_rows('results_nonlinear_trajectory_trials.csv', rows)

    summary_rows = []
    for trajectory_type in trajectory_types:
        for strategy in strategies:
            selected = [
                row for row in rows
                if row['TrajectoryType'] == trajectory_type and row['Strategy'] == strategy
            ]
            successes = sum(row['Success'] for row in selected)
            ci_low, ci_high = wilson_ci95(successes, len(selected))
            summary_rows.append({
                'TrajectoryType': trajectory_type,
                'Strategy': strategy,
                'Trials': len(selected),
                'Successes': successes,
                'SuccessRate': round(successes / len(selected), 4),
                'SuccessCI95Low': round(ci_low, 4),
                'SuccessCI95High': round(ci_high, 4),
                'MeanReplans': round(statistics.fmean(row['Replans'] for row in selected), 4),
                'MeanWaitSteps': round(statistics.fmean(row['WaitSteps'] for row in selected), 4),
                'MeanPathLength': round(statistics.fmean(row['PathLength'] for row in selected), 4),
                'MeanPredictionMAE': round(statistics.fmean(row['PredictionMAE'] for row in selected), 4),
                'MeanRiskRadius': round(statistics.fmean(row['MeanRiskRadius'] for row in selected), 4),
                'MeanTotalCost': round(statistics.fmean(row['TotalCost'] for row in selected), 4),
            })
    write_dict_rows('results_nonlinear_trajectory_summary.csv', summary_rows)
    return summary_rows


def run_nonlinear_parameter_calibration(
        calibration_seeds=5, evaluation_seeds=20, missions_per_seed=3):
    lines = make_dynamic_replanning_scene()
    grid, start, goal = build_scene(lines)
    robot = RobotFootprint(body_width=2, body_length=4, leg_margin=0, sensor_margin=0, safe_margin=0.25)
    trajectory_types = ['turn', 'sudden-acceleration', 'stop-and-go']
    process_noise_values = [0.01, 0.03, 0.08]
    base_risk_radius_values = [1.3, 1.5, 1.7]
    uncertainty_scale_values = [0.01, 0.02, 0.05]
    rows = []
    for trajectory_type in trajectory_types:
        for process_noise in process_noise_values:
            for base_risk_radius in base_risk_radius_values:
                for uncertainty_scale in uncertainty_scale_values:
                    for seed in range(calibration_seeds):
                        memory = ShortTermMemory(ttl=40)
                        for mission in range(missions_per_seed):
                            obstacle = nonlinear_moving_obstacle_scenario(seed, mission, trajectory_type)
                            predictor = KalmanTrajectoryPredictor(
                                horizon=10, position_noise=0.75,
                                process_noise=process_noise,
                                uncertainty_scale=uncertainty_scale,
                                base_risk_radius=base_risk_radius,
                                max_risk_radius=2.4,
                                rng=random.Random(
                                    seed * 313 + mission * 19
                                    + trajectory_types.index(trajectory_type) * 100003))
                            result = simulate_trajectory_prediction_mission(
                                'Prediction-aware memory', grid, start, goal,
                                robot, robot, obstacle, predictor, memory=memory)
                            rows.append({
                                'TrajectoryType': trajectory_type,
                                'ProcessNoise': process_noise,
                                'BaseRiskRadius': base_risk_radius,
                                'UncertaintyScale': uncertainty_scale,
                                'Seed': seed,
                                'Mission': mission,
                                **result,
                            })
    write_dict_rows('results_nonlinear_calibration_trials.csv', rows)

    summary_rows = []
    for trajectory_type in trajectory_types:
        for process_noise in process_noise_values:
            for base_risk_radius in base_risk_radius_values:
                for uncertainty_scale in uncertainty_scale_values:
                    selected = [
                        row for row in rows
                        if row['TrajectoryType'] == trajectory_type
                        and row['ProcessNoise'] == process_noise
                        and row['BaseRiskRadius'] == base_risk_radius
                        and row['UncertaintyScale'] == uncertainty_scale
                    ]
                    successes = sum(row['Success'] for row in selected)
                    summary_rows.append({
                        'TrajectoryType': trajectory_type,
                        'ProcessNoise': process_noise,
                        'BaseRiskRadius': base_risk_radius,
                        'UncertaintyScale': uncertainty_scale,
                        'Trials': len(selected),
                        'Successes': successes,
                        'SuccessRate': round(successes / len(selected), 4),
                        'MeanPredictionMAE': round(statistics.fmean(row['PredictionMAE'] for row in selected), 4),
                        'MeanRiskRadius': round(statistics.fmean(row['MeanRiskRadius'] for row in selected), 4),
                        'MeanReplans': round(statistics.fmean(row['Replans'] for row in selected), 4),
                        'MeanWaitSteps': round(statistics.fmean(row['WaitSteps'] for row in selected), 4),
                        'MeanTotalCost': round(statistics.fmean(row['TotalCost'] for row in selected), 4),
                    })
    write_dict_rows('results_nonlinear_calibration_summary.csv', summary_rows)

    best_rows = []
    for trajectory_type in trajectory_types:
        candidates = [
            row for row in summary_rows
            if row['TrajectoryType'] == trajectory_type
        ]
        best_rows.append(min(
            candidates,
            key=lambda row: (
                row['MeanTotalCost'], -row['SuccessRate'],
                row['MeanRiskRadius'], row['ProcessNoise'])))
    write_dict_rows('results_nonlinear_calibration_best.csv', best_rows)

    evaluation_rows = []
    strategies = [
        'Constant velocity memory',
        'Default uncertainty-aware memory',
        'Calibrated uncertainty-aware memory',
    ]
    for trajectory_type in trajectory_types:
        best = next(
            row for row in best_rows
            if row['TrajectoryType'] == trajectory_type)
        for strategy in strategies:
            for seed in range(1000, 1000 + evaluation_seeds):
                memory = ShortTermMemory(ttl=40)
                for mission in range(missions_per_seed):
                    obstacle = nonlinear_moving_obstacle_scenario(seed, mission, trajectory_type)
                    predictor_rng = random.Random(
                        seed * 313 + mission * 19
                        + trajectory_types.index(trajectory_type) * 100003)
                    if strategy == 'Constant velocity memory':
                        predictor = OnlineTrajectoryPredictor(
                            horizon=10, position_noise=0.75,
                            velocity_noise=0.12, rng=predictor_rng)
                    elif strategy == 'Default uncertainty-aware memory':
                        predictor = KalmanTrajectoryPredictor(
                            horizon=10, position_noise=0.75,
                            process_noise=0.03, uncertainty_scale=0.02,
                            base_risk_radius=1.5, max_risk_radius=2.4,
                            rng=predictor_rng)
                    else:
                        predictor = KalmanTrajectoryPredictor(
                            horizon=10, position_noise=0.75,
                            process_noise=best['ProcessNoise'],
                            uncertainty_scale=best['UncertaintyScale'],
                            base_risk_radius=best['BaseRiskRadius'],
                            max_risk_radius=2.4, rng=predictor_rng)
                    result = simulate_trajectory_prediction_mission(
                        'Prediction-aware memory', grid, start, goal,
                        robot, robot, obstacle, predictor, memory=memory)
                    evaluation_rows.append({
                        'TrajectoryType': trajectory_type,
                        'Strategy': strategy,
                        'Seed': seed,
                        'Mission': mission,
                        **result,
                    })
    write_dict_rows('results_nonlinear_calibration_evaluation_trials.csv', evaluation_rows)

    evaluation_summary_rows = []
    for trajectory_type in trajectory_types:
        for strategy in strategies:
            selected = [
                row for row in evaluation_rows
                if row['TrajectoryType'] == trajectory_type and row['Strategy'] == strategy
            ]
            successes = sum(row['Success'] for row in selected)
            ci_low, ci_high = wilson_ci95(successes, len(selected))
            evaluation_summary_rows.append({
                'TrajectoryType': trajectory_type,
                'Strategy': strategy,
                'Trials': len(selected),
                'Successes': successes,
                'SuccessRate': round(successes / len(selected), 4),
                'SuccessCI95Low': round(ci_low, 4),
                'SuccessCI95High': round(ci_high, 4),
                'MeanPredictionMAE': round(statistics.fmean(row['PredictionMAE'] for row in selected), 4),
                'MeanRiskRadius': round(statistics.fmean(row['MeanRiskRadius'] for row in selected), 4),
                'MeanReplans': round(statistics.fmean(row['Replans'] for row in selected), 4),
                'MeanWaitSteps': round(statistics.fmean(row['WaitSteps'] for row in selected), 4),
                'MeanTotalCost': round(statistics.fmean(row['TotalCost'] for row in selected), 4),
            })
    write_dict_rows(
        'results_nonlinear_calibration_evaluation_summary.csv',
        evaluation_summary_rows)
    return best_rows, evaluation_summary_rows


def run_prediction_decision_policy_evaluation(
        evaluation_seeds=20, missions_per_seed=3):
    lines = make_dynamic_replanning_scene()
    grid, start, goal = build_scene(lines)
    robot = RobotFootprint(body_width=2, body_length=4, leg_margin=0, sensor_margin=0, safe_margin=0.25)
    trajectory_types = ['turn', 'sudden-acceleration', 'stop-and-go']
    strategies = [
        'Constant velocity legacy',
        'Constant velocity cost-aware',
        'Calibrated uncertainty legacy',
        'Calibrated uncertainty cost-aware',
    ]
    calibrated_rows = list(csv.DictReader(open(
        'results_nonlinear_calibration_best.csv', newline='')))
    calibrated = {
        row['TrajectoryType']: {
            'ProcessNoise': float(row['ProcessNoise']),
            'BaseRiskRadius': float(row['BaseRiskRadius']),
            'UncertaintyScale': float(row['UncertaintyScale']),
        }
        for row in calibrated_rows
    }
    rows = []
    for trajectory_type in trajectory_types:
        parameters = calibrated[trajectory_type]
        for strategy in strategies:
            for seed in range(2000, 2000 + evaluation_seeds):
                memory = ShortTermMemory(ttl=40)
                for mission in range(missions_per_seed):
                    obstacle = nonlinear_moving_obstacle_scenario(seed, mission, trajectory_type)
                    predictor_rng = random.Random(
                        seed * 313 + mission * 19
                        + trajectory_types.index(trajectory_type) * 100003)
                    if strategy.startswith('Constant velocity'):
                        predictor = OnlineTrajectoryPredictor(
                            horizon=10, position_noise=0.75,
                            velocity_noise=0.12, rng=predictor_rng)
                    else:
                        predictor = KalmanTrajectoryPredictor(
                            horizon=10, position_noise=0.75,
                            process_noise=parameters['ProcessNoise'],
                            uncertainty_scale=parameters['UncertaintyScale'],
                            base_risk_radius=parameters['BaseRiskRadius'],
                            max_risk_radius=2.4, rng=predictor_rng)
                    result = simulate_trajectory_prediction_mission(
                        'Prediction-aware memory', grid, start, goal,
                        robot, robot, obstacle, predictor, memory=memory,
                        decision_policy=(
                            'cost-aware' if strategy.endswith('cost-aware')
                            else 'legacy'))
                    rows.append({
                        'TrajectoryType': trajectory_type,
                        'Strategy': strategy,
                        'Seed': seed,
                        'Mission': mission,
                        **result,
                    })
    write_dict_rows('results_prediction_decision_trials.csv', rows)

    summary_rows = []
    for trajectory_type in trajectory_types:
        for strategy in strategies:
            selected = [
                row for row in rows
                if row['TrajectoryType'] == trajectory_type and row['Strategy'] == strategy
            ]
            successes = sum(row['Success'] for row in selected)
            ci_low, ci_high = wilson_ci95(successes, len(selected))
            summary_rows.append({
                'TrajectoryType': trajectory_type,
                'Strategy': strategy,
                'Trials': len(selected),
                'Successes': successes,
                'SuccessRate': round(successes / len(selected), 4),
                'SuccessCI95Low': round(ci_low, 4),
                'SuccessCI95High': round(ci_high, 4),
                'MeanPredictionMAE': round(statistics.fmean(row['PredictionMAE'] for row in selected), 4),
                'MeanRiskRadius': round(statistics.fmean(row['MeanRiskRadius'] for row in selected), 4),
                'MeanReplans': round(statistics.fmean(row['Replans'] for row in selected), 4),
                'MeanWaitSteps': round(statistics.fmean(row['WaitSteps'] for row in selected), 4),
                'MeanDecisionWaits': round(statistics.fmean(row['DecisionWaits'] for row in selected), 4),
                'MeanDecisionReroutes': round(statistics.fmean(row['DecisionReroutes'] for row in selected), 4),
                'MeanTotalCost': round(statistics.fmean(row['TotalCost'] for row in selected), 4),
            })
    write_dict_rows('results_prediction_decision_summary.csv', summary_rows)
    return summary_rows


def run_space_time_dynamic_statistics(seeds=20, missions_per_seed=3):
    lines = make_dynamic_replanning_scene()
    grid, start, goal = build_scene(lines)
    robot = RobotFootprint(body_width=2, body_length=4, leg_margin=0, sensor_margin=0, safe_margin=0.25)
    trajectory_types = ['turn', 'sudden-acceleration', 'stop-and-go']
    strategies = ['Space-Time CV', 'Space-Time Kalman uncertainty']
    rows = []
    for trajectory_type in trajectory_types:
        for strategy in strategies:
            for seed in range(seeds):
                dynamic_memory = ShortTermMemory(ttl=16)
                for mission in range(missions_per_seed):
                    obstacle = nonlinear_moving_obstacle_scenario(seed, mission, trajectory_type)
                    predictor_rng = random.Random(
                        seed * 521 + mission * 37
                        + trajectory_types.index(trajectory_type) * 100003)
                    if strategy == 'Space-Time Kalman uncertainty':
                        predictor = KalmanTrajectoryPredictor(
                            horizon=16, position_noise=0.75,
                            process_noise=0.03, uncertainty_scale=0.02,
                            base_risk_radius=1.5, max_risk_radius=2.4,
                            rng=predictor_rng)
                    else:
                        predictor = OnlineTrajectoryPredictor(
                            horizon=16, position_noise=0.75,
                            velocity_noise=0.12, rng=predictor_rng)
                    result = simulate_space_time_prediction_mission(
                        grid, start, goal, robot, obstacle, predictor,
                        dynamic_memory=dynamic_memory)
                    rows.append({
                        'TrajectoryType': trajectory_type,
                        'Strategy': strategy,
                        'Seed': seed,
                        'Mission': mission,
                        **{
                            key: value for key, value in result.items()
                            if key != 'ExecutedPath'
                        },
                    })
    write_dict_rows('results_space_time_dynamic_trials.csv', rows)

    summary_rows = []
    for trajectory_type in trajectory_types:
        for strategy in strategies:
            selected = [
                row for row in rows
                if row['TrajectoryType'] == trajectory_type and row['Strategy'] == strategy
            ]
            successes = sum(row['Success'] for row in selected)
            ci_low, ci_high = wilson_ci95(successes, len(selected))
            summary_rows.append({
                'TrajectoryType': trajectory_type,
                'Strategy': strategy,
                'Trials': len(selected),
                'Successes': successes,
                'SuccessRate': round(successes / len(selected), 4),
                'SuccessCI95Low': round(ci_low, 4),
                'SuccessCI95High': round(ci_high, 4),
                'MeanReplans': round(statistics.fmean(row['Replans'] for row in selected), 4),
                'MeanWaitSteps': round(statistics.fmean(row['WaitSteps'] for row in selected), 4),
                'MeanTimeSteps': round(statistics.fmean(row['TimeSteps'] for row in selected), 4),
                'MeanPathLength': round(statistics.fmean(row['PathLength'] for row in selected), 4),
                'MeanPredictionMAE': round(statistics.fmean(row['PredictionMAE'] for row in selected), 4),
                'CollisionRate': round(statistics.fmean(row['Collision'] for row in selected), 4),
                'MeanCollisionCount': round(statistics.fmean(row['CollisionCount'] for row in selected), 4),
                'MeanExpandedStates': round(statistics.fmean(row['ExpandedStates'] for row in selected), 4),
                'MeanGeneratedStates': round(statistics.fmean(row['GeneratedStates'] for row in selected), 4),
                'MeanPlanningTime': round(statistics.fmean(row['MeanPlanningTime'] for row in selected), 6),
                'MaxPlanningTime': round(max(row['MaxPlanningTime'] for row in selected), 6),
                'MeanMinDynamicClearance': round(statistics.fmean(row['MinDynamicClearance'] for row in selected), 4),
                'MeanTotalCost': round(statistics.fmean(row['TotalCost'] for row in selected), 4),
            })
    write_dict_rows('results_space_time_dynamic_summary.csv', summary_rows)
    return summary_rows


def run_space_time_semantic_dynamic_experiment():
    robot = RobotFootprint(body_width=0, body_length=0, leg_margin=0, sensor_margin=0, safe_margin=0)
    source_observed_lines, _, source_failed_region = make_repeated_passage_scenes()
    source_grid, _, _ = build_scene(source_observed_lines)
    semantic_memory = SemanticAnchorMemory(similarity_threshold=0.8)
    semantic_memory.remember_failure(source_grid, source_failed_region)

    start = (5, 13)
    goal = (44, 13)
    observed_lines, truth_lines, target_failed_region = make_repeated_passage_scenes(
        start=start, goal=goal, deceptive_opening_y=13,
        safe_opening_start=18, barrier_x=31, width=50, height=27)
    observed_grid, _, _ = build_scene(observed_lines)
    truth_grid, _, _ = build_scene(truth_lines)
    projected_cells, matches = semantic_memory.recall(observed_grid)
    obstacle = MovingObstacle(
        31, 7.0, 0.0, 0.34, radius=1.8)
    variants = [
        ('Space-Time without semantic memory', set()),
        ('Space-Time with semantic-anchor memory', projected_cells),
    ]
    rows = []
    for variant, static_memory_cells in variants:
        predictor = KalmanTrajectoryPredictor(
            horizon=16, position_noise=0.35, process_noise=0.02,
            uncertainty_scale=0.02, base_risk_radius=1.3,
            max_risk_radius=2.0, rng=random.Random(42))
        result = simulate_space_time_prediction_mission(
            observed_grid, start, goal, robot, obstacle, predictor,
            static_memory_cells=static_memory_cells,
            dynamic_memory=ShortTermMemory(ttl=16), max_steps=160,
            replan_interval=1)
        executed_path = result.pop('ExecutedPath')
        rows.append({
            'Variant': variant,
            'MatchedAnchor': str(matches[0]['TargetCenter']) if matches else '',
            'MatchSimilarity': matches[0]['Similarity'] if matches else 0.0,
            'TransferredCells': len(static_memory_cells),
            'TruthExecutable': GridBenchmark(
                truth_grid, start, goal, robot).validate_path(executed_path),
            'UsedTargetFailedPassage': any(
                point in target_failed_region for point in executed_path),
            **result,
        })
    write_dict_rows('results_space_time_semantic_dynamic.csv', rows)
    return rows


def path_length(path):
    if not path:
        return 0.0
    return sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(path, path[1:]))


def generate_visualizations(realistic_result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)

    sampling_rows = list(csv.DictReader(open('results_sampling_summary.csv', newline='')))
    scenes = ['w7', 'w6', 'w5', 'w4', 'wide']
    fig, ax = plt.subplots(figsize=(7, 4))
    for planner, offset, color in [('RRT', -0.12, '#4c78a8'), ('RRT*', 0.12, '#f58518')]:
        rows = {
            row['Scene']: row for row in sampling_rows
            if row['Phase'] == 'body' and row['Planner'] == planner
        }
        values = [float(rows[scene]['SuccessRate']) for scene in scenes]
        errors = [
            [
                value - float(rows[scene]['SuccessCI95Low'])
                for scene, value in zip(scenes, values)
            ],
            [
                float(rows[scene]['SuccessCI95High']) - value
                for scene, value in zip(scenes, values)
            ],
        ]
        ax.errorbar(
            [index + offset for index in range(len(scenes))], values, yerr=errors,
            marker='o', capsize=3, label=planner, color=color)
    ax.set_xticks(range(len(scenes)), scenes)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Success rate')
    ax.set_title('Body-aware narrow-passage reliability (100 seeds)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'sampling_success_rates.png', dpi=180)
    plt.close(fig)

    ablation_rows = list(csv.DictReader(open('results_ablation_summary.csv', newline='')))
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [row['Variant'] for row in ablation_rows]
    values = [float(row['FailedShortPassageAttempts']) for row in ablation_rows]
    ax.bar(labels, values, color=['#9d9d9d', '#72b7b2', '#54a24b'])
    ax.set_ylabel('Failed short-passage attempts')
    ax.set_title('Failure-memory ablation (150 tasks)')
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / 'memory_ablation.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, path, title in [
            (axes[0], realistic_result['baseline_path'], 'Optimistic local-map plan'),
            (axes[1], realistic_result['memory_path'], 'Memory-aware rectangular-footprint plan')]:
        grid = realistic_result['truth_grid']
        ax.imshow(grid, cmap='Greys', origin='upper', vmin=0, vmax=1)
        if path:
            ax.plot([point[0] for point in path], [point[1] for point in path], color='#e45756', linewidth=2)
            for point, nxt in zip(path[::max(1, len(path) // 8)], path[1::max(1, len(path) // 8)]):
                theta = math.atan2(nxt[1] - point[1], nxt[0] - point[0])
                corners = rectangle_corners(point[0], point[1], theta, realistic_result['rectangle_robot'])
                ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor='#4c78a8', linewidth=0.8))
        ax.scatter(*realistic_result['start'], color='#54a24b', label='start')
        ax.scatter(*realistic_result['goal'], color='#f58518', label='goal')
        ax.set_title(title)
        ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(output_dir / 'realistic_paths.png', dpi=180)
    plt.close(fig)

    dynamic_rows = list(csv.DictReader(open('results_dynamic_summary.csv', newline='')))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [row['Strategy'] for row in dynamic_rows]
    success_rates = [float(row['SuccessRate']) for row in dynamic_rows]
    costs = [float(row['MeanTotalCost']) for row in dynamic_rows]
    colors = ['#9d9d9d', '#72b7b2', '#f58518', '#54a24b']
    axes[0].bar(labels, success_rates, color=colors)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel('Success rate')
    axes[0].set_title('Dynamic replanning success (300 missions)')
    axes[1].bar(labels, costs, color=colors)
    axes[1].set_ylabel('Mean weighted task cost')
    axes[1].set_title('Distance + 3 x wait + 5 x replan')
    for ax in axes:
        ax.tick_params(axis='x', rotation=18)
    fig.tight_layout()
    fig.savefig(output_dir / 'dynamic_replanning.png', dpi=180)
    plt.close(fig)

    prediction_rows = list(csv.DictReader(open('results_trajectory_prediction_summary.csv', newline='')))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [row['Strategy'] for row in prediction_rows]
    success_rates = [float(row['SuccessRate']) for row in prediction_rows]
    costs = [float(row['MeanTotalCost']) for row in prediction_rows]
    colors = ['#9d9d9d', '#f58518', '#4c78a8', '#54a24b']
    axes[0].bar(labels, success_rates, color=colors)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel('Success rate')
    axes[0].set_title('Online trajectory prediction (300 missions)')
    axes[1].bar(labels, costs, color=colors)
    axes[1].set_ylabel('Mean weighted task cost')
    axes[1].set_title('Prediction-aware replanning cost')
    for ax in axes:
        ax.tick_params(axis='x', rotation=12)
    fig.tight_layout()
    fig.savefig(output_dir / 'trajectory_prediction.png', dpi=180)
    plt.close(fig)

    noise_rows = list(csv.DictReader(open('results_prediction_noise_summary.csv', newline='')))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for predictor_name, color in [('Constant velocity', '#f58518'), ('Kalman acceleration', '#54a24b')]:
        selected = [row for row in noise_rows if row['Predictor'] == predictor_name]
        levels = [float(row['PositionNoise']) for row in selected]
        rates = [float(row['SuccessRate']) for row in selected]
        errors = [float(row['MeanPredictionMAE']) for row in selected]
        axes[0].plot(levels, rates, marker='o', label=predictor_name, color=color)
        axes[1].plot(levels, errors, marker='o', label=predictor_name, color=color)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel('Position noise std (cells)')
    axes[0].set_ylabel('Success rate')
    axes[0].set_title('Noise robustness')
    axes[1].set_xlabel('Position noise std (cells)')
    axes[1].set_ylabel('One-step prediction MAE (cells)')
    axes[1].set_title('Prediction error')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'prediction_noise_ablation.png', dpi=180)
    plt.close(fig)

    uncertainty_rows = list(csv.DictReader(open('results_uncertainty_radius_summary.csv', newline='')))
    fig, ax = plt.subplots(figsize=(7, 4))
    for noise, color in [(0.75, '#4c78a8'), (1.5, '#f58518'), (2.5, '#54a24b')]:
        selected = [row for row in uncertainty_rows if float(row['PositionNoise']) == noise]
        scales = [float(row['RiskScale']) for row in selected]
        costs = [float(row['MeanTotalCost']) for row in selected]
        ax.plot(scales, costs, marker='o', label=f'noise={noise}', color=color)
    ax.set_xlabel('Uncertainty scale k')
    ax.set_ylabel('Mean weighted task cost')
    ax.set_title('Uncertainty-aware risk-radius search')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'uncertainty_radius_search.png', dpi=180)
    plt.close(fig)

    nonlinear_rows = list(csv.DictReader(open('results_nonlinear_trajectory_summary.csv', newline='')))
    trajectory_types = ['turn', 'sudden-acceleration', 'stop-and-go']
    strategies = [
        'Reactive current occupancy',
        'Constant velocity memory',
        'Kalman fixed-radius memory',
        'Kalman uncertainty-aware memory',
    ]
    colors = ['#9d9d9d', '#f58518', '#4c78a8', '#54a24b']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    width = 0.18
    for strategy_index, (strategy, color) in enumerate(zip(strategies, colors)):
        selected = {
            row['TrajectoryType']: row for row in nonlinear_rows
            if row['Strategy'] == strategy
        }
        positions = [
            index + (strategy_index - 1.5) * width
            for index in range(len(trajectory_types))
        ]
        axes[0].bar(
            positions,
            [float(selected[trajectory_type]['SuccessRate']) for trajectory_type in trajectory_types],
            width=width, label=strategy, color=color)
        axes[1].bar(
            positions,
            [float(selected[trajectory_type]['MeanTotalCost']) for trajectory_type in trajectory_types],
            width=width, label=strategy, color=color)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel('Success rate')
    axes[0].set_title('Nonlinear trajectory success')
    axes[1].set_ylabel('Mean weighted task cost')
    axes[1].set_title('Nonlinear trajectory task cost')
    for ax in axes:
        ax.set_xticks(range(len(trajectory_types)), trajectory_types)
        ax.tick_params(axis='x', rotation=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output_dir / 'nonlinear_trajectory_comparison.png', dpi=180)
    plt.close(fig)

    calibration_path = Path('results_nonlinear_calibration_evaluation_summary.csv')
    if calibration_path.exists():
        calibration_rows = list(csv.DictReader(open(calibration_path, newline='')))
        strategies = [
            'Constant velocity memory',
            'Default uncertainty-aware memory',
            'Calibrated uncertainty-aware memory',
        ]
        colors = ['#f58518', '#54a24b', '#4c78a8']
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        width = 0.24
        for strategy_index, (strategy, color) in enumerate(zip(strategies, colors)):
            selected = {
                row['TrajectoryType']: row for row in calibration_rows
                if row['Strategy'] == strategy
            }
            positions = [
                index + (strategy_index - 1) * width
                for index in range(len(trajectory_types))
            ]
            axes[0].bar(
                positions,
                [float(selected[trajectory_type]['SuccessRate']) for trajectory_type in trajectory_types],
                width=width, label=strategy, color=color)
            axes[1].bar(
                positions,
                [float(selected[trajectory_type]['MeanTotalCost']) for trajectory_type in trajectory_types],
                width=width, label=strategy, color=color)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel('Success rate')
        axes[0].set_title('Holdout success after trajectory-specific calibration')
        axes[1].set_ylabel('Mean weighted task cost')
        axes[1].set_title('Holdout cost after trajectory-specific calibration')
        for ax in axes:
            ax.set_xticks(range(len(trajectory_types)), trajectory_types)
            ax.tick_params(axis='x', rotation=12)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(output_dir / 'nonlinear_calibration_holdout.png', dpi=180)
        plt.close(fig)

    decision_path = Path('results_prediction_decision_summary.csv')
    if decision_path.exists():
        decision_rows = list(csv.DictReader(open(decision_path, newline='')))
        strategies = [
            'Constant velocity legacy',
            'Constant velocity cost-aware',
            'Calibrated uncertainty legacy',
            'Calibrated uncertainty cost-aware',
        ]
        colors = ['#f58518', '#ffbf79', '#4c78a8', '#72b7b2']
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        width = 0.18
        for strategy_index, (strategy, color) in enumerate(zip(strategies, colors)):
            selected = {
                row['TrajectoryType']: row for row in decision_rows
                if row['Strategy'] == strategy
            }
            positions = [
                index + (strategy_index - 1.5) * width
                for index in range(len(trajectory_types))
            ]
            axes[0].bar(
                positions,
                [float(selected[trajectory_type]['SuccessRate']) for trajectory_type in trajectory_types],
                width=width, label=strategy, color=color)
            axes[1].bar(
                positions,
                [float(selected[trajectory_type]['MeanTotalCost']) for trajectory_type in trajectory_types],
                width=width, label=strategy, color=color)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel('Success rate')
        axes[0].set_title('Cost-aware wait-or-reroute success')
        axes[1].set_ylabel('Mean weighted task cost')
        axes[1].set_title('Cost-aware wait-or-reroute task cost')
        for ax in axes:
            ax.set_xticks(range(len(trajectory_types)), trajectory_types)
            ax.tick_params(axis='x', rotation=12)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(output_dir / 'prediction_decision_policy.png', dpi=180)
        plt.close(fig)

    space_time_path = Path('results_space_time_dynamic_summary.csv')
    if space_time_path.exists():
        space_time_rows = list(csv.DictReader(open(space_time_path, newline='')))
        trajectory_types = ['turn', 'sudden-acceleration', 'stop-and-go']
        strategies = ['Space-Time CV', 'Space-Time Kalman uncertainty']
        colors = ['#f58518', '#4c78a8']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        width = 0.32
        for strategy_index, (strategy, color) in enumerate(zip(strategies, colors)):
            selected = {
                row['TrajectoryType']: row for row in space_time_rows
                if row['Strategy'] == strategy
            }
            positions = [
                index + (strategy_index - 0.5) * width
                for index in range(len(trajectory_types))
            ]
            axes[0].bar(
                positions,
                [float(selected[trajectory_type]['SuccessRate']) for trajectory_type in trajectory_types],
                width=width, label=strategy, color=color)
            axes[1].bar(
                positions,
                [float(selected[trajectory_type]['MeanTotalCost']) for trajectory_type in trajectory_types],
                width=width, label=strategy, color=color)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel('Success rate')
        axes[0].set_title('Space-Time A* dynamic avoidance')
        axes[1].set_ylabel('Mean weighted task cost')
        axes[1].set_title('Space-Time A* task cost')
        for ax in axes:
            ax.set_xticks(range(len(trajectory_types)), trajectory_types)
            ax.tick_params(axis='x', rotation=12)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(output_dir / 'space_time_dynamic_avoidance.png', dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        width = 0.32
        for strategy_index, (strategy, color) in enumerate(zip(strategies, colors)):
            selected = {
                row['TrajectoryType']: row for row in space_time_rows
                if row['Strategy'] == strategy
            }
            positions = [
                index + (strategy_index - 0.5) * width
                for index in range(len(trajectory_types))
            ]
            axes[0].bar(
                positions,
                [float(selected[trajectory_type]['CollisionRate']) for trajectory_type in trajectory_types],
                width=width, label=strategy, color=color)
            axes[1].bar(
                positions,
                [float(selected[trajectory_type]['MeanExpandedStates']) for trajectory_type in trajectory_types],
                width=width, label=strategy, color=color)
            axes[2].bar(
                positions,
                [float(selected[trajectory_type]['MeanPlanningTime']) for trajectory_type in trajectory_types],
                width=width, label=strategy, color=color)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel('Collision rate')
        axes[0].set_title('Dynamic collision safety')
        axes[1].set_ylabel('Mean expanded states')
        axes[1].set_title('Search effort per mission')
        axes[2].set_ylabel('Mean planning time per replan (s)')
        axes[2].set_title('Runtime per replan')
        for ax in axes:
            ax.set_xticks(range(len(trajectory_types)), trajectory_types)
            ax.tick_params(axis='x', rotation=12)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(output_dir / 'space_time_runtime_safety.png', dpi=180)
        plt.close(fig)

    semantic_dynamic_path = Path('results_space_time_semantic_dynamic.csv')
    if semantic_dynamic_path.exists():
        semantic_dynamic_rows = list(csv.DictReader(open(semantic_dynamic_path, newline='')))
        labels = [row['Variant'] for row in semantic_dynamic_rows]
        truth_executable = [
            1.0 if row['TruthExecutable'] == 'True' else 0.0
            for row in semantic_dynamic_rows
        ]
        costs = [float(row['TotalCost']) for row in semantic_dynamic_rows]
        colors = ['#9d9d9d', '#54a24b']
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(labels, truth_executable, color=colors)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel('Truth-executable')
        axes[0].set_title('Semantic memory in dynamic space-time planning')
        axes[1].bar(labels, costs, color=colors)
        axes[1].set_ylabel('Weighted task cost')
        axes[1].set_title('Avoid failed passage and moving obstacle')
        for ax in axes:
            ax.tick_params(axis='x', rotation=12)
        fig.tight_layout()
        fig.savefig(output_dir / 'space_time_semantic_dynamic.png', dpi=180)
        plt.close(fig)

    migration_path = Path('results_semantic_anchor_migration_summary.csv')
    if migration_path.exists():
        migration_rows = list(csv.DictReader(open(migration_path, newline='')))
        labels = [row['Variant'] for row in migration_rows]
        success_rates = [float(row['ExecutableRate']) for row in migration_rows]
        failed_selections = [float(row['FailedPassageSelections']) for row in migration_rows]
        colors = ['#9d9d9d', '#f58518', '#54a24b']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(labels, success_rates, color=colors)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel('Executable rate')
        axes[0].set_title('Zero-shot memory transfer (100 perturbed maps)')
        axes[1].bar(labels, failed_selections, color=colors)
        axes[1].set_ylabel('Failed-passage selections')
        axes[1].set_title('Transferred memory avoids repeated failures')
        for ax in axes:
            ax.tick_params(axis='x', rotation=12)
        fig.tight_layout()
        fig.savefig(output_dir / 'semantic_anchor_migration.png', dpi=180)
        plt.close(fig)


def run_all():
    scenes = {
        'Scene 1 (easy-w7)': make_single_passage_scene(7),
        'Scene 2 (medium-w6)': make_single_passage_scene(6),
        'Scene 3 (hard-w5)': make_single_passage_scene(5),
        'Scene 4 (narrow-w4)': make_single_passage_scene(4),
        'Scene 5 (wide-easy)': make_wide_scene(),
        'Scene 6 (multi-passage)': make_multi_passage_scene(),
    }

    random.seed(0)
    planners = [AStarPlanner(), DijkstraPlanner(), RRTPlanner(), RRTStarPlanner(), HybridAStarPlanner(), OursPlanner()]

    phases = [
        ('point', RobotFootprint(body_width=0, body_length=0, leg_margin=0, sensor_margin=0, safe_margin=0), 'results_point_robot.csv'),
        ('small', RobotFootprint(body_width=1, body_length=2, leg_margin=0, sensor_margin=0, safe_margin=0), 'results_small_robot.csv'),
        ('body', RobotFootprint(body_width=2, body_length=2, leg_margin=0, sensor_margin=0, safe_margin=1), 'results_body_aware.csv'),
    ]
    results = []
    for phase_name, robot_cfg, csv_path in phases:
        csv_rows = []
        print(f"\n===== Phase: {phase_name} robot (inflation={robot_cfg.inflation_radius}) =====")
        for scene_name, lines in scenes.items():
            grid, start, goal = build_scene(lines)
            benchmark = GridBenchmark(grid, start, goal, robot_cfg)
            print(f'\n=== {scene_name} ({phase_name} robot) ===')
            # debug: print start/goal and whether they're free after inflation
            print('start:', start, 'goal:', goal)
            try:
                print('start_free:', benchmark.is_free(*start))
                print('goal_free:', benchmark.is_free(*goal))
            except Exception:
                print('start_free/goal_free: invalid start/goal')
            print('inflation_radius:', robot_cfg.inflation_radius)
            print_grid(benchmark.inflated, start=start, goal=goal)
            for planner in planners:
                summary = benchmark.run_planner(planner)
                results.append((scene_name, phase_name, summary))
                csv_rows.append((scene_name, summary['planner'], summary['success'], summary['path_length'], summary['min_clearance'], summary['narrow_success'], summary['planning_time']))
                print(f"{planner.name}: success={summary['success']} length={summary['path_length']} "
                      f"clearance={summary['min_clearance']} narrow_ok={summary['narrow_success']} "
                      f"time={summary['planning_time']}")

        # write CSV for this phase
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scene', 'Planner', 'Success', 'Length', 'Clearance', 'NarrowOK', 'Time'])
            for row in csv_rows:
                writer.writerow(row)
        print(f"Saved {phase_name}-robot results to {csv_path}")
    print('\nSummary Table:')
    print('Phase,Scene,Planner,Success,Length,Clearance,NarrowOK,Time')
    for scene, phase, summary in results:
        print(f"{phase},{scene},{summary['planner']},{summary['success']},{summary['path_length']},{summary['min_clearance']},{summary['narrow_success']},{summary['planning_time']}")

    print('\nRepeated Passage Experiment:')
    print('Variant,Task,Attempt,Planned,Executable,UsedShortPassage,Length,RememberedCells')
    for row in run_repeated_passage_experiment():
        print(','.join(str(value) for value in row.values()))

    print('\nSampling Statistics (100 seeds):')
    print('Phase,Scene,Planner,Trials,Successes,SuccessRate,SuccessCI95Low,SuccessCI95High')
    for row in run_sampling_statistics():
        print(','.join(str(row[key]) for key in [
            'Phase', 'Scene', 'Planner', 'Trials', 'Successes',
            'SuccessRate', 'SuccessCI95Low', 'SuccessCI95High']))

    print('\nMemory Statistics (30 batches x 5 perturbed tasks):')
    print('Variant,Tasks,Attempts,ExecutableTasks,ExecutableRate,FailedShortPassageAttempts,MeanAttemptsPerTask')
    for row in run_memory_statistics():
        print(','.join(str(row[key]) for key in [
            'Variant', 'Tasks', 'Attempts', 'ExecutableTasks', 'ExecutableRate',
            'FailedShortPassageAttempts', 'MeanAttemptsPerTask']))

    print('\nAblation Statistics:')
    print('Variant,Tasks,Attempts,ExecutableTasks,ExecutableRate,FailedShortPassageAttempts,MeanAttemptsPerTask')
    for row in run_ablation_statistics():
        print(','.join(str(row[key]) for key in [
            'Variant', 'Tasks', 'Attempts', 'ExecutableTasks', 'ExecutableRate',
            'FailedShortPassageAttempts', 'MeanAttemptsPerTask']))

    print('\nEnvironment Recovery:')
    print('Stage,RecoveryTask,Executable,UsedShortPassage,Length,MemoryStrength')
    for row in run_environment_recovery_experiment():
        print(','.join(str(value) for value in row.values()))

    print('\nUnrelated Maps:')
    print('Map,Attempt,Executable,UsedCurrentShortPassage,Length,RememberedCellsBeforeUpdate')
    for row in run_unrelated_maps_experiment():
        print(','.join(str(value) for value in row.values()))

    print('\nSemantic Anchor Migration:')
    print('Map,Variant,SourceAnchor,MatchedAnchor,MatchSimilarity,TransferredCells,Planned,Executable,UsedTargetFailedPassage,Length')
    for row in run_semantic_anchor_migration_experiment():
        print(','.join(str(value) for value in row.values()))

    print('\nSemantic Anchor Migration Statistics (100 perturbed maps):')
    print('Variant,Trials,ExecutableTasks,ExecutableRate,ExecutableCI95Low,ExecutableCI95High,FailedPassageSelections,MeanTransferredCells,MeanMatchSimilarity')
    for row in run_semantic_anchor_migration_statistics():
        print(','.join(str(value) for value in row.values()))

    print('\nSemantic Anchor Specificity:')
    print('Case,Matches,ProjectedCells,Rejected')
    for row in run_semantic_anchor_specificity_experiment():
        print(','.join(str(value) for value in row.values()))

    print('\nRealistic Suite:')
    print('Variant,Planned,RectangleStaticExecutable,DynamicExecutable,PathLength,ObservedCells,NoisyObservedCells')
    realistic_result = run_realistic_suite()
    for row in realistic_result['rows']:
        print(','.join(str(value) for value in row.values()))

    print('\nDynamic Replanning Statistics (100 seeds x 3 missions):')
    print('Strategy,Trials,Successes,SuccessRate,MeanReplans,MeanWaitSteps,MeanPathLength,MeanTotalCost')
    for row in run_dynamic_replanning_statistics():
        print(','.join(str(row[key]) for key in [
            'Strategy', 'Trials', 'Successes', 'SuccessRate', 'MeanReplans',
            'MeanWaitSteps', 'MeanPathLength', 'MeanTotalCost']))

    print('\nTrajectory Prediction Statistics (100 seeds x 3 missions):')
    print('Strategy,Trials,Successes,SuccessRate,MeanReplans,MeanWaitSteps,MeanPathLength,MeanPredictionMAE,MeanRiskRadius,MeanTotalCost')
    for row in run_trajectory_prediction_statistics():
        print(','.join(str(row[key]) for key in [
            'Strategy', 'Trials', 'Successes', 'SuccessRate', 'MeanReplans',
            'MeanWaitSteps', 'MeanPathLength', 'MeanPredictionMAE',
            'MeanRiskRadius', 'MeanTotalCost']))

    print('\nPrediction Noise Ablation (20 seeds x 3 missions):')
    print('Predictor,PositionNoise,Trials,Successes,SuccessRate,MeanPredictionMAE,MeanReplans,MeanWaitSteps,MeanTotalCost')
    for row in run_prediction_noise_ablation():
        print(','.join(str(row[key]) for key in [
            'Predictor', 'PositionNoise', 'Trials', 'Successes', 'SuccessRate',
            'MeanPredictionMAE', 'MeanReplans', 'MeanWaitSteps', 'MeanTotalCost']))

    print('\nUncertainty Radius Search (10 seeds x 3 missions):')
    print('PositionNoise,RiskScale,Trials,Successes,SuccessRate,MeanPredictionMAE,MeanRiskRadius,MeanReplans,MeanWaitSteps,MeanTotalCost')
    for row in run_uncertainty_radius_search():
        print(','.join(str(row[key]) for key in [
            'PositionNoise', 'RiskScale', 'Trials', 'Successes', 'SuccessRate',
            'MeanPredictionMAE', 'MeanRiskRadius', 'MeanReplans',
            'MeanWaitSteps', 'MeanTotalCost']))

    print('\nNonlinear Trajectory Statistics (20 seeds x 3 missions):')
    print('TrajectoryType,Strategy,Trials,Successes,SuccessRate,MeanPredictionMAE,MeanRiskRadius,MeanReplans,MeanWaitSteps,MeanTotalCost')
    for row in run_nonlinear_trajectory_statistics():
        print(','.join(str(row[key]) for key in [
            'TrajectoryType', 'Strategy', 'Trials', 'Successes', 'SuccessRate',
            'MeanPredictionMAE', 'MeanRiskRadius', 'MeanReplans',
            'MeanWaitSteps', 'MeanTotalCost']))

    print('\nNonlinear Parameter Calibration (5 training seeds, 20 holdout seeds):')
    best_rows, evaluation_rows = run_nonlinear_parameter_calibration()
    print('TrajectoryType,ProcessNoise,BaseRiskRadius,UncertaintyScale,SuccessRate,MeanTotalCost')
    for row in best_rows:
        print(','.join(str(row[key]) for key in [
            'TrajectoryType', 'ProcessNoise', 'BaseRiskRadius',
            'UncertaintyScale', 'SuccessRate', 'MeanTotalCost']))
    print('TrajectoryType,Strategy,Trials,Successes,SuccessRate,MeanPredictionMAE,MeanRiskRadius,MeanTotalCost')
    for row in evaluation_rows:
        print(','.join(str(row[key]) for key in [
            'TrajectoryType', 'Strategy', 'Trials', 'Successes',
            'SuccessRate', 'MeanPredictionMAE', 'MeanRiskRadius',
            'MeanTotalCost']))

    print('\nPrediction Decision Policy Evaluation (20 holdout seeds x 3 missions):')
    print('TrajectoryType,Strategy,Trials,Successes,SuccessRate,MeanReplans,MeanWaitSteps,MeanDecisionWaits,MeanDecisionReroutes,MeanTotalCost')
    for row in run_prediction_decision_policy_evaluation():
        print(','.join(str(row[key]) for key in [
            'TrajectoryType', 'Strategy', 'Trials', 'Successes',
            'SuccessRate', 'MeanReplans', 'MeanWaitSteps',
            'MeanDecisionWaits', 'MeanDecisionReroutes', 'MeanTotalCost']))

    print('\nSpace-Time Dynamic Avoidance Statistics (20 seeds x 3 missions):')
    print('TrajectoryType,Strategy,Trials,Successes,SuccessRate,CollisionRate,MeanReplans,MeanWaitSteps,MeanTimeSteps,MeanExpandedStates,MeanPlanningTime,MaxPlanningTime,MeanMinDynamicClearance,MeanTotalCost')
    for row in run_space_time_dynamic_statistics():
        print(','.join(str(row[key]) for key in [
            'TrajectoryType', 'Strategy', 'Trials', 'Successes',
            'SuccessRate', 'CollisionRate', 'MeanReplans', 'MeanWaitSteps',
            'MeanTimeSteps', 'MeanExpandedStates', 'MeanPlanningTime',
            'MaxPlanningTime', 'MeanMinDynamicClearance', 'MeanTotalCost']))

    print('\nSpace-Time Semantic Dynamic Experiment:')
    print('Variant,MatchedAnchor,MatchSimilarity,TransferredCells,TruthExecutable,UsedTargetFailedPassage,Success,Replans,WaitSteps,TimeSteps,PathLength,PredictionMAE,TotalCost')
    for row in run_space_time_semantic_dynamic_experiment():
        print(','.join(str(value) for value in row.values()))

    generate_visualizations(realistic_result)
    print('Saved figures to figures/')


if __name__ == '__main__':
    run_all()
