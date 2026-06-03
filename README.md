# 2D Occupancy Grid Benchmark

这是一个简化的 2D grid benchmark，用于比较多个 planner 在 narrow-passage 场景下的表现。重点是体现普通 planner 容易失败的情况，以及基于机器人几何尺寸和失败记忆的 `Ours` 方法更安全的通道选择能力。

## 结构

- `benchmark.py`: 实验核心代码
  - 6 个 narrow-passage 场景
  - 机器人矩形 footprint 和 obstacle inflation
  - Planner: A*, Dijkstra, RRT, RRT*, Hybrid-A*, Ours
  - 评估指标: 成功、路径长度、最小 clearance、窄通道成功、碰撞、规划时间

## 运行

```bash
python benchmark.py
```

## 设计要点

- `A*`, `Dijkstra`, `RRT`, `RRT*` 都在 inflated grid 上规划，保证基线公平性。
- `Ours` 额外使用 passage clearance 估计和失败记忆策略，优先避免不可行或狭窄通道。
- 场景包括：直通道、窄门、非对称间隙、杂乱开口、伪可行间隙、多通道选择。
