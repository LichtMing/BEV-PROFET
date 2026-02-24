# 变更记录 — Active Learning 模式修复

日期: 2025 年

## 概述

为使 INTERACTION 数据集场景（如 `USA_Intersection-1_1_T-1.xml`）在 `active_learning_enabled: true` 模式下正常运行，共修复 5 个 bug。

---

## 修复列表

### 1. Polygon.center 0-d 数组问题
**文件**: `EthicalTrajectoryPlanning/planner/GlobalPath/lanelet_based_planner.py`  
**问题**: 目标状态的 `Polygon.center` 返回 shapely POINT 被包裹在 0 维 numpy 数组中，导致 `find_lanelet_by_position` 调用时 `iteration over a 0-d array` 异常。  
**修复**: 检测 0-d 数组后用 `goal_center.item()` 提取为 `[x, y]` 坐标。

### 2. Lanelet 网络 successor 列表被污染
**文件**: `commonroad_helper_functions/spacial.py`（site-packages）  
**问题**: `all_lanelets_by_merging_predecessors_from_lanelet` 函数中 `j[k].successor.append(pred.lanelet_id)` 直接修改了原始 lanelet 网络的 successor 列表。为第一个障碍物创建 Agent 后，虚拟合并 lanelet ID（如 `124146`）被加入到 lanelet 120 的 successor，导致后续障碍物创建 Agent 时 `ValueError: None cannot be a node`。  
**修复**: 对 `j[k]` 和 `j[0]` 使用 `copy.deepcopy`，在副本上操作而不影响原始网络。

### 3. update_scenario() 多余参数
**文件**: `EthicalTrajectoryPlanning/planner/plannertools/scenario_handler.py`  
**问题**: `update_scenario()` 调用时传递了 `active_learning=self.active_learning` 关键字参数，但 `agent_sim` 的 `update_scenario()` 方法不接受此参数。  
**修复**: 移除多余的 `active_learning` 关键字参数。

### 4. reference_traj 引用被覆盖
**文件**: `EthicalTrajectoryPlanning/planner/Frenet/frenet_planner.py`  
**问题**: `self.reference_traj` 直接引用 scenario 中障碍物轨迹，`update_scenario()` 执行后原始轨迹被修改，导致后续步骤无法获取正确的参考轨迹，引发 `AttributeError: 'NoneType' object has no attribute 'position'`。  
**修复**: 使用 `copy.deepcopy` 复制轨迹；并为 `next_state` 添加 `None` 检查，当轨迹结束时使用当前 ego_state 作为回退。

### 5. 硬编码绝对路径
**文件**: `EthicalTrajectoryPlanning/planner/Frenet/configs/planning_fast.json`  
**问题**: `label_path` 和 `fig_path` 硬编码为 `/home/lqy/Projects/...`，当前机器无此路径，导致 `PermissionError`。  
**修复**: 替换为当前工作空间路径 `/mnt/e/traj_bev/workspacebackup - 副本/EthicalTrajectoryPlanning/label` 和 `.../saved_fig`。

---

## 验证

修复后成功运行：
```bash
cd EthicalTrajectoryPlanning/planner/Frenet
python frenet_planner.py --scenario USA_Intersection-1_1_T-1.xml
```
障碍物 161 的 16 个时间步规划全部完成，输出各步的速度和加速度。
