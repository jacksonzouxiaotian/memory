#!/usr/bin/env python3

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


class DatasetTask:
    def __init__(self, start, goal, task_id=0):
        self.start = start
        self.goal = goal
        self.task_id = task_id


class DatasetMovingObstacle:
    def __init__(self, obstacle_id, samples):
        self.obstacle_id = obstacle_id
        self.samples = sorted(samples, key=lambda sample: sample[0])
        if not self.samples:
            raise ValueError(f"dynamic obstacle {obstacle_id} has no samples")
        self.radius = max(sample[3] for sample in self.samples)

    def position_at(self, time_step):
        if time_step <= self.samples[0][0]:
            return self.samples[0][1], self.samples[0][2]
        if time_step >= self.samples[-1][0]:
            return self.samples[-1][1], self.samples[-1][2]
        for left, right in zip(self.samples, self.samples[1:]):
            left_t, left_x, left_y, _ = left
            right_t, right_x, right_y, _ = right
            if left_t <= time_step <= right_t:
                ratio = (time_step - left_t) / max(1e-9, right_t - left_t)
                return (
                    left_x + (right_x - left_x) * ratio,
                    left_y + (right_y - left_y) * ratio,
                )
        return self.samples[-1][1], self.samples[-1][2]

    def occupied_cells_at(self, time_step, padding=0.0):
        x, y = self.position_at(time_step)
        radius = self.radius + padding
        cells = set()
        for cell_y in range(int(np.floor(y - radius)), int(np.ceil(y + radius)) + 1):
            for cell_x in range(int(np.floor(x - radius)), int(np.ceil(x + radius)) + 1):
                if ((cell_x - x) ** 2 + (cell_y - y) ** 2) ** 0.5 <= radius:
                    cells.add((cell_x, cell_y))
        return cells


class DatasetBenchmark:
    def __init__(self, root, grid, tasks, dynamic_obstacles):
        self.root = Path(root)
        self.grid = grid
        self.tasks = tasks
        self.dynamic_obstacles = dynamic_obstacles


def load_dataset(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset directory does not exist: {root}")
    grid = load_map(root)
    tasks = load_tasks(root / "tasks.csv")
    dynamic_path = root / "dynamic_obstacles.csv"
    dynamic_obstacles = (
        load_dynamic_obstacles(dynamic_path) if dynamic_path.exists() else []
    )
    return DatasetBenchmark(root, grid, tasks, dynamic_obstacles)


def load_map(root):
    npy_path = root / "map.npy"
    pgm_path = root / "map.pgm"
    if npy_path.exists():
        return load_npy_map(npy_path)
    if pgm_path.exists():
        return load_pgm_map(pgm_path)
    raise FileNotFoundError(f"expected {npy_path} or {pgm_path}")


def load_npy_map(path):
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"map.npy must be 2D, got shape {array.shape}")
    if array.dtype == np.bool_:
        occupied = array
    else:
        occupied = array > 0
    return occupied.astype(int).tolist()


def load_pgm_map(path, free_threshold=250):
    with Path(path).open("rb") as f:
        magic = _next_pgm_token(f)
        if magic not in (b"P2", b"P5"):
            raise ValueError(f"unsupported PGM magic {magic!r}; expected P2 or P5")
        width = int(_next_pgm_token(f))
        height = int(_next_pgm_token(f))
        max_value = int(_next_pgm_token(f))
        if max_value <= 0 or max_value > 65535:
            raise ValueError(f"unsupported PGM max value: {max_value}")
        if magic == b"P2":
            values = [int(_next_pgm_token(f)) for _ in range(width * height)]
        else:
            bytes_per_value = 1 if max_value < 256 else 2
            data = f.read(width * height * bytes_per_value)
            if len(data) != width * height * bytes_per_value:
                raise ValueError("PGM data is shorter than expected")
            if bytes_per_value == 1:
                values = list(data)
            else:
                values = [
                    int.from_bytes(data[index:index + 2], "big")
                    for index in range(0, len(data), 2)
                ]
    normalized_threshold = free_threshold / 255 * max_value
    return [
        [1 if values[y * width + x] < normalized_threshold else 0 for x in range(width)]
        for y in range(height)
    ]


def _next_pgm_token(stream):
    token = bytearray()
    while True:
        char = stream.read(1)
        if not char:
            if token:
                return bytes(token)
            raise ValueError("unexpected EOF while reading PGM header")
        if char == b"#":
            stream.readline()
            continue
        if char.isspace():
            if token:
                return bytes(token)
            continue
        token.extend(char)


def load_tasks(path):
    if not path.exists():
        raise FileNotFoundError(f"missing tasks file: {path}")
    tasks = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"start_x", "start_y", "goal_x", "goal_y"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"tasks.csv missing columns: {sorted(missing)}")
        for index, row in enumerate(reader):
            tasks.append(
                DatasetTask(
                    start=(int(float(row["start_x"])), int(float(row["start_y"]))),
                    goal=(int(float(row["goal_x"])), int(float(row["goal_y"]))),
                    task_id=row.get("id") or index,
                )
            )
    if not tasks:
        raise ValueError("tasks.csv must contain at least one task")
    return tasks


def load_dynamic_obstacles(path):
    grouped = defaultdict(list)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "t", "x", "y", "radius"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"dynamic_obstacles.csv missing columns: {sorted(missing)}")
        for row in reader:
            grouped[row["id"]].append(
                (
                    float(row["t"]),
                    float(row["x"]),
                    float(row["y"]),
                    float(row["radius"]),
                )
            )
    return [
        DatasetMovingObstacle(obstacle_id, samples)
        for obstacle_id, samples in sorted(grouped.items())
    ]


def select_representative_obstacle(task, obstacles):
    if not obstacles:
        return None
    return min(
        obstacles,
        key=lambda obstacle: _min_distance_to_segment(task.start, task.goal, obstacle),
    )


def _min_distance_to_segment(start, goal, obstacle):
    return min(
        _point_segment_distance((sample[1], sample[2]), start, goal)
        for sample in obstacle.samples
    )


def _point_segment_distance(point, start, goal):
    px, py = point
    sx, sy = start
    gx, gy = goal
    vx, vy = gx - sx, gy - sy
    wx, wy = px - sx, py - sy
    denom = vx * vx + vy * vy
    if denom == 0:
        return ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
    ratio = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    closest = (sx + ratio * vx, sy + ratio * vy)
    return ((px - closest[0]) ** 2 + (py - closest[1]) ** 2) ** 0.5
