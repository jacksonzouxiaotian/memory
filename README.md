# Memory-Aware Planning Benchmark

这个仓库包含一个受控二维仿真 benchmark，用来研究机器人记忆模块如何影响路径规划、动态避障和跨场景迁移。

核心问题是：

> 机器人不应该只记住地图坐标，而应该把失败经验、动态交互经验和局部语义结构转化为可迁移的规划偏置。

当前代码已经从最初的 narrow-passage 几何实验，扩展到：

- 传统 planner 在二维几何 narrow passage 中的失效分析。
- 静态失败记忆和 clearance-aware planning。
- 短期动态记忆、轨迹预测和 cost-aware 等待或绕行。
- 非线性动态障碍：转向、突然加速、stop-and-go。
- Kalman 状态估计和 uncertainty-aware risk radius。
- 局部语义锚点记忆，用于跨扰动地图迁移失败经验。
- `(x, y, t)` Space-Time A*，将动态避障、prediction、dynamic memory 和 semantic memory 接入同一个时空规划器。
- Space-Time Kalman 参数校准和 ablation 表。

## 文件结构

主要文件：

- `benchmark.py`：全部实验、planner、memory、动态障碍、统计和可视化代码。
- `results_*.csv`：已运行实验的原始 trials 和 summary。
- `figures/`：论文可用的统计图和路径可视化。
- `BENCHMARK_REPORT.md`：早期 Level-1 benchmark 记录，当前 README 是最新总览。

本地存在的 `PythonRobotics/` 是外部参考仓库，已被 `.gitignore` 排除，不属于本项目提交内容。

## 运行方式

运行全部实验：

```bash
python benchmark.py
```

注意：完整运行会比较慢，尤其是 Space-Time A*、Kalman 校准和 100-seed 统计。论文复现实验建议按函数单独运行，例如：

```python
import benchmark

benchmark.run_sampling_statistics()
benchmark.run_memory_statistics()
benchmark.run_semantic_anchor_migration_statistics()
benchmark.run_space_time_dynamic_statistics()
benchmark.run_space_time_kalman_calibration()
```

重新生成全部图表：

```python
import benchmark

benchmark.generate_visualizations(benchmark.run_realistic_suite())
```

## 主要模块

### 1. 基础 planner

实现了多个二维 planner：

- `AStarPlanner`
- `DijkstraPlanner`
- `RRTPlanner`
- `RRTStarPlanner`
- `HybridAStarPlanner`
- `OursPlanner`

`OursPlanner` 支持 clearance penalty 和 failure memory penalty，用来模拟机器人把历史失败区域写入代价地图后重新规划。

### 2. 静态失败记忆

静态记忆实验模拟一个 deceptive narrow passage：

- 观测地图中该短通道看似可通行。
- 真值地图中该短通道被阻塞。
- 无记忆 planner 会反复选择短但不可执行路径。
- memory-aware planner 会在失败后记住该区域，改走更长但可执行的安全通道。

相关函数：

- `run_repeated_passage_experiment`
- `run_memory_statistics`
- `run_ablation_statistics`
- `run_environment_recovery_experiment`
- `run_unrelated_maps_experiment`

这部分证明：记忆模块的基本价值不是缩短单次路径，而是减少重复失败并提升任务可执行性。

### 3. 动态记忆和预测决策

动态实验加入移动障碍和在线重规划：

- 短期动态记忆 `ShortTermMemory`
- 常速度预测 `OnlineTrajectoryPredictor`
- Kalman 预测 `KalmanTrajectoryPredictor`
- uncertainty-aware risk radius
- cost-aware wait-or-reroute 决策

相关函数：

- `run_dynamic_replanning_statistics`
- `run_trajectory_prediction_statistics`
- `run_prediction_noise_ablation`
- `run_uncertainty_radius_search`
- `run_nonlinear_trajectory_statistics`
- `run_nonlinear_parameter_calibration`
- `run_prediction_decision_policy_evaluation`

非线性动态障碍包括：

- `turn`
- `sudden-acceleration`
- `stop-and-go`

关键结果：

| 方法 | 成功率 | 平均等待 | 平均任务代价 |
|---|---:|---:|---:|
| Constant velocity legacy | 85.56% | 6.28 | 156.50 |
| Constant velocity cost-aware | 100.00% | 3.72 | 134.33 |
| Kalman uncertainty legacy | 75.00% | 6.48 | 177.90 |
| Kalman uncertainty cost-aware | 100.00% | 4.47 | 140.02 |

解释：

> 预测本身并不自动带来规划收益。预测必须经过时间对齐的风险判断，并进入等待或绕行决策层，才能稳定改善动态导航。

### 4. 局部语义锚点记忆

`SemanticAnchorMemory` 不再记忆绝对坐标，而是记忆：

- 局部 occupancy patch
- 通道方向
- 开口宽度
- 锚点中心
- 相对失败区域
- memory type
- confidence

在新地图中，它会扫描相似局部通道结构，并把相对失败区域投影到目标地图。

相关函数：

- `run_semantic_anchor_migration_experiment`
- `run_semantic_anchor_migration_statistics`
- `run_semantic_anchor_specificity_experiment`

迁移实验结果：

| 方法 | 零样本首次可执行率 | 重复进入失败通道 |
|---|---:|---:|
| 无迁移记忆 | 0.00% | 100 / 100 |
| 绝对坐标记忆 | 0.00% | 100 / 100 |
| 语义锚点迁移 | 100.00% | 0 / 100 |

同时，specificity 测试显示不同开口宽度、不同方向和开放地图会被拒绝，没有错误投影。

解释：

> 机器人应该记住“某类局部结构下的失败经验”，而不是记住“某个绝对坐标附近失败”。

### 5. Space-Time A*

`SpaceTimeAStarPlanner` 将状态扩展为：

```python
(x, y, t)
```

动作包括：

- 8 邻域移动
- 原地等待

每个候选状态都会检查：

- 静态地图障碍
- 预测动态障碍占用
- 短期动态 memory 风险
- 语义锚点迁移出的静态失败区域

相关函数：

- `simulate_space_time_prediction_mission`
- `run_space_time_dynamic_statistics`
- `run_space_time_semantic_dynamic_experiment`
- `run_space_time_kalman_calibration`

新增指标包括：

- `CollisionRate`
- `CollisionCount`
- `ExpandedStates`
- `GeneratedStates`
- `MeanPlanningTime`
- `MaxPlanningTime`
- `MinDynamicClearance`

20 seeds x 3 missions 的 Space-Time 动态统计：

| 场景 | 策略 | 成功率 | 碰撞率 | 平均 expanded states | 平均规划时间/replan |
|---|---|---:|---:|---:|---:|
| turn | Space-Time CV | 98.33% | 1.67% | 77,449 | 0.0578s |
| turn | Kalman uncertainty | 71.67% | 28.33% | 200,319 | 0.2325s |
| sudden-acceleration | Space-Time CV | 86.67% | 13.33% | 86,384 | 0.0706s |
| sudden-acceleration | Kalman uncertainty | 75.00% | 25.00% | 194,780 | 0.2025s |
| stop-and-go | Space-Time CV | 98.33% | 1.67% | 84,734 | 0.0679s |
| stop-and-go | Kalman uncertainty | 81.67% | 18.33% | 192,977 | 0.1769s |

这说明默认 Kalman uncertainty 并不自动优于常速度预测。更复杂的预测模型会增加搜索负担，如果风险半径和时空规划器没有共同校准，反而会提高碰撞率。

### 6. Space-Time Kalman 校准和 ablation

为了让 Kalman uncertainty 真正适配 Space-Time A*，代码加入了参数搜索：

- `process_noise`
- `base_risk_radius`
- `uncertainty_scale`
- `max_risk_radius`

最佳参数：

| 参数 | 值 |
|---|---:|
| `process_noise` | 0.01 |
| `base_risk_radius` | 1.8 |
| `uncertainty_scale` | 0.02 |
| `max_risk_radius` | 3.0 |

Ablation 总体结果：

| 方法 | 成功率 | 碰撞率 | 平均代价 |
|---|---:|---:|---:|
| No prediction / no memory | 0.00% | 100.00% | 222.93 |
| CV no dynamic memory | 33.33% | 66.67% | 182.72 |
| CV + dynamic memory | 93.33% | 6.67% | 129.28 |
| Default Kalman + dynamic memory | 53.33% | 46.67% | 156.60 |
| Calibrated Kalman + dynamic memory | 93.33% | 6.67% | 124.65 |

解释：

> Kalman 模型本身并不是结论，经过 Space-Time 风险半径校准后，它才能把更低的预测误差转化为更低碰撞率和更低任务代价。

## 主要结果图

`figures/` 中比较重要的图：

- `sampling_success_rates.png`：RRT/RRT* 在 body-aware narrow passage 下的 100-seed 成功率。
- `memory_ablation.png`：静态失败记忆消融。
- `dynamic_replanning.png`：动态重规划策略比较。
- `trajectory_prediction.png`：轨迹预测对动态任务的影响。
- `prediction_decision_policy.png`：cost-aware wait-or-reroute 决策。
- `semantic_anchor_migration.png`：语义锚点跨地图迁移。
- `space_time_dynamic_avoidance.png`：Space-Time A* 动态避障成功率和代价。
- `space_time_runtime_safety.png`：Space-Time A* 碰撞率、expanded states 和运行时间。
- `space_time_ablation.png`：Space-Time prediction/memory/Kalman 消融表。
- `space_time_kalman_calibration.png`：Kalman 参数搜索结果。

## 论文中可以如何解释

当前仿真支持以下主线：

> Memory improves navigation not by replacing the planner, but by reshaping the planner's cost and risk model using prior execution experience. Static failure memory prevents repeated infeasible paths, short-term dynamic memory supports online avoidance under moving obstacles, and semantic-anchor memory enables zero-shot transfer of failures across perturbed maps. Space-Time A* unifies these memory sources with trajectory prediction in a single `(x, y, t)` planning framework.

中文解释：

> 记忆模块不是替代路径规划器，而是利用历史执行经验重塑规划器的代价和风险模型。静态失败记忆避免重复选择不可执行路径，短期动态记忆帮助机器人处理移动障碍，语义锚点记忆使失败经验能够跨扰动地图迁移。Space-Time A* 进一步把动态预测、短期记忆和语义迁移统一到 `(x, y, t)` 时空规划框架中。

## 当前限制

这些结果仍然是 controlled simulation evidence，不应直接表述为真实机器人部署结果。

主要限制：

- 仿真仍是二维栅格地图。
- 语义锚点是手工特征，不是学习到的表征。
- 动态障碍数量较少，主要是单动态对象。
- Space-Time A* 计算量较大，完整 20-seed 统计约需十几分钟。
- Kalman 的收益依赖风险半径和时空规划预算校准。
- 尚未接入 ROS 2 costmap、SLAM、真实传感器或真实机器人控制闭环。

## 建议的下一步

优先补充：

1. 多动态障碍场景：two crossing、opposite-direction、occluded obstacle。
2. 语义锚点 hard negative：多个相似通道，验证错误迁移率。
3. Space-Time A* 运行效率优化：局部窗口、incremental planning、state pruning。
4. 旋转和尺度扰动迁移：不只测试平移和轻噪声。
5. ROS 2 costmap layer：把 static memory、dynamic memory 和 semantic memory 接入真实导航栈。

## GitHub

仓库地址：

```text
git@github.com:jacksonzouxiaotian/memory.git
```
