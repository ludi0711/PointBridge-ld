# Isaac 控制命令源码打白

本文只讲 Isaac Lab / Isaac Sim 源码里的控制命令执行链路：一条 action 从 `env.step(action)` 进入仿真以后，在哪些类里被拆分、预处理、写入 articulation buffer、写入 PhysX target/state，最后怎么触发一次 physics step。

本机源码位置：

```text
Isaac Lab: /home/gxai/IsaacLab/source/isaaclab/isaaclab
Isaac Sim: /home/gxai/IsaacLab/_isaac_sim/exts
```

一句话总览：

```text
ManagerBasedRLEnv.step(action)
  -> ActionManager.process_action(action)             # 每个 env step 一次
  -> for i in range(decimation):
       ActionManager.apply_action()                   # 每个 physics step 一次
       InteractiveScene.write_data_to_sim()
       SimulationContext.step(render=False)
       InteractiveScene.update(dt=physics_dt)
  -> termination / reward / reset / observation
```

## 1. env.step 的源码顺序

入口在 Isaac Lab：

```text
/home/gxai/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:154
```

`ManagerBasedRLEnv.step(action)` 做的事情是：

1. `self.action_manager.process_action(action.to(self.device))`：把外部 action 交给 action manager。源码：`manager_based_rl_env.py:174`
2. 进入 `for _ in range(self.cfg.decimation)`：一个 policy step 内跑多个 physics step。源码：`manager_based_rl_env.py:183`
3. 每个 physics step 先 `self.action_manager.apply_action()`。源码：`manager_based_rl_env.py:186`
4. 再 `self.scene.write_data_to_sim()`，把 scene 里各 asset 的 buffered command 写入 sim。源码：`manager_based_rl_env.py:188`
5. 再 `self.sim.step(render=False)`，真正推进 PhysX。源码：`manager_based_rl_env.py:190`
6. 再 `self.scene.update(dt=self.physics_dt)`，从仿真读回 asset/sensor 状态缓存。源码：`manager_based_rl_env.py:198`
7. 跳出 physics loop 后，算 termination、reward、reset、observation。源码：`manager_based_rl_env.py:205` 起。

时间参数来自 `ManagerBasedEnv`：

```text
physics_dt = cfg.sim.dt
step_dt    = cfg.sim.dt * cfg.decimation
```

源码：`/home/gxai/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py:225` 和 `:233`。

`decimation` 本身是 env cfg 字段，含义是 “每个 policy dt 里有多少个 sim dt”。源码：`/home/gxai/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env_cfg.py:70`。

## 2. ActionManager 怎么处理 action

源码：

```text
/home/gxai/IsaacLab/source/isaaclab/isaaclab/managers/action_manager.py
```

`ActionManager.process_action(action)`：

- 检查 action 维度是否等于所有 action term 的总维度。
- 保存 `_prev_action` 和 `_action`。
- 按每个 term 的 `action_dim` 切片。
- 调每个 term 的 `term.process_actions(term_actions)`。

关键源码：

```text
action_manager.py:371  def process_action(...)
action_manager.py:381  维度检查
action_manager.py:384  保存输入 action
action_manager.py:389  按 term 切片
action_manager.py:391  term.process_actions(...)
```

`ActionManager.apply_action()`：

- 不再重新拆 action。
- 对所有 action term 调 `term.apply_actions()`。
- 它在 `ManagerBasedRLEnv.step()` 的 decimation loop 内，每个 physics step 都会被调用一次。

关键源码：

```text
action_manager.py:394  def apply_action(...)
action_manager.py:401  term.apply_actions()
```

Action term 的基类明确区分两阶段：

- `process_actions`：每个 environment step 一次。
- `apply_actions`：每个 simulation step 一次。

源码说明在 `action_manager.py:27` 附近的 `ActionTerm` docstring。

## 3. Isaac Lab 原生关节 action 怎么做

源码：

```text
/home/gxai/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py
/home/gxai/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py
```

### 3.1 JointAction 通用预处理

`JointAction.process_actions(actions)` 公式是：

```python
processed = raw_action * scale + offset
processed = clamp(processed, clip_low, clip_high)  # 如果配置了 clip
```

关键源码：

```text
joint_actions.py:27   class JointAction
actions_cfg.py:27    class JointActionCfg
manager_term_cfg.py:96 ActionTermCfg.clip
joint_actions.py:168  def process_actions(...)
joint_actions.py:170  保存 raw action
joint_actions.py:172  scale + offset
joint_actions.py:174  clip
```

这些参数在哪里调：

- `asset_name` / `debug_vis` / `clip`：`ActionTermCfg`，源码 `manager_term_cfg.py:78` 到 `:96`。
- `joint_names` / `scale` / `offset` / `preserve_order`：`JointActionCfg`，源码 `actions_cfg.py:27` 到 `:41`。
- 具体任务里，通常在 env cfg 的 `actions = ...` 里实例化某个 `ActionTermCfg`。

### 3.2 JointPositionAction：绝对位置目标

`JointPositionAction.apply_actions()` 只做一件事：

```python
asset.set_joint_position_target(processed_actions, joint_ids=joint_ids)
```

源码：`joint_actions.py:183` 和 `joint_actions.py:198`。

注意：这是设置 “目标”，不是直接改仿真关节状态。它只写到 articulation 的内部 target buffer，后面要等 `scene.write_data_to_sim()` 才真正进 PhysX。

### 3.3 RelativeJointPositionAction：相对当前位置的位置目标

`RelativeJointPositionAction.apply_actions()` 做的是：

```python
current_actions = processed_actions + asset.data.joint_pos[:, joint_ids]
asset.set_joint_position_target(current_actions, joint_ids=joint_ids)
```

源码：`joint_actions.py:201` 到 `joint_actions.py:231`。

关键点：`apply_actions()` 每个 physics step 都会调用一次。如果 `decimation > 1`，原生 `RelativeJointPositionAction` 会在每个 physics substep 里用当时的 `asset.data.joint_pos` 重新算一次 `current + delta`。这和“每个 policy step 只加一次 delta”不是同一种语义。

### 3.4 Task-space / IK action

Isaac Lab 的 task-space action、RMPFlow、Pink IK 等也是 action term。它们会在 `process_actions/apply_actions` 中把末端位姿命令变成 joint target，最终通常仍然走：

```text
asset.set_joint_position_target(...)
-> scene.write_data_to_sim()
-> articulation.write_data_to_sim()
-> PhysX drive target
```

所以它们的下游执行路径和 JointPositionAction 类似，区别在上游多了 IK / task-space controller 计算。

## 4. set_joint_position_target 不等于执行

`Articulation.set_joint_position_target(...)` 的源码说明很明确：

```text
This function does not apply the joint targets to the simulation.
It only fills the buffers with the desired values.
To apply the joint targets, call write_data_to_sim.
```

源码位置：

```text
/home/gxai/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py:1060
```

它实际只写：

```python
self._data.joint_pos_target[env_ids, joint_ids] = target
```

源码：`articulation.py:1082`。

同类 target setter：

- `set_joint_position_target`：`articulation.py:1060`
- `set_joint_velocity_target`：`articulation.py:1084`
- `set_joint_effort_target`：`articulation.py:1108`

这些都是写 buffer，不是立刻推进仿真。

## 5. scene.write_data_to_sim 怎么把 target 推到 articulation

源码：

```text
/home/gxai/IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py:464
```

`InteractiveScene.write_data_to_sim()` 会遍历 scene 里的资产：

```python
for articulation in self._articulations.values():
    articulation.write_data_to_sim()
for deformable_object in ...:
    ...
for rigid_object in ...:
    ...
```

关键源码：`interactive_scene.py:464` 到 `:476`。

也就是说，action term 并不直接调用 PhysX step；它只是把命令写到 asset buffer。进入 PhysX 之前，还要经过 scene 的统一写入阶段。

## 6. Articulation.write_data_to_sim 怎么写 PhysX target

源码：

```text
/home/gxai/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py:187
```

`Articulation.write_data_to_sim()` 的核心顺序：

1. 如果有 external wrench，先写外力/外力矩。
2. `_apply_actuator_model()`：把用户 target buffer 交给 actuator 处理。
3. `root_physx_view.set_dof_actuation_forces(...)`：写 effort command。
4. 如果存在 implicit actuator，再写 position target 和 velocity target：

```python
root_physx_view.set_dof_position_targets(...)
root_physx_view.set_dof_velocity_targets(...)
```

关键源码：

```text
articulation.py:187  def write_data_to_sim(...)
articulation.py:216  self._apply_actuator_model()
articulation.py:218  set_dof_actuation_forces(...)
articulation.py:222  set_dof_position_targets(...)
articulation.py:223  set_dof_velocity_targets(...)
```

到这里，Isaac Lab 原生命令已经写进 PhysX Tensor API 的 articulation view 了。下一步 `sim.step()` 才会让 PhysX 根据这些 target、drive 参数、约束和 solver 真正积分。

## 7. actuator 在哪里处理参数

Actuator 配置源码：

```text
/home/gxai/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_base_cfg.py
/home/gxai/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd_cfg.py
/home/gxai/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py
```

### 7.1 参数字段

`ActuatorBaseCfg` 里可以调：

- `joint_names_expr`：这组 actuator 绑定哪些 joint。
- `effort_limit`：显式 actuator 的模型输出力矩 clip；对 implicit actuator 与 `effort_limit_sim` 等价，但更推荐用 `effort_limit_sim`。
- `velocity_limit`：显式 actuator 的模型参数；对 implicit actuator 旧语义下不直接用。
- `effort_limit_sim`：写进 physics solver 的 effort limit。
- `velocity_limit_sim`：写进 physics solver 的 velocity limit。
- `stiffness`：P gain。implicit actuator 会直接写进 PhysX；explicit actuator 用它自己算 effort。
- `damping`：D gain。implicit actuator 会直接写进 PhysX；explicit actuator 用它自己算 effort。
- `armature` / `friction` 等 solver 参数。

源码位置：`actuator_base_cfg.py:12` 到 `:130`。

### 7.2 ImplicitActuator

`ImplicitActuatorCfg` 的说明：PD control 由仿真处理。源码：`actuator_pd_cfg.py:19`。

`ImplicitActuator.compute(...)` 不真正生成新的 target，它直接返回 control action；只是为了 reward/日志近似算了一下 torque：

```python
computed_effort = stiffness * error_pos + damping * error_vel + effort_ff
applied_effort = clip(computed_effort, effort_limit)
return control_action
```

源码：`actuator_pd.py:35` 到 `:143`。

真正写进 PhysX 的 stiffness/damping/limits 在 articulation 初始化 actuator 时完成：

```text
articulation.py:1688  遍历 cfg.actuators
articulation.py:1733  implicit actuator: write_joint_stiffness_to_sim(...)
articulation.py:1734  implicit actuator: write_joint_damping_to_sim(...)
articulation.py:1743  write_joint_effort_limit_to_sim(...)
articulation.py:1744  write_joint_velocity_limit_to_sim(...)
```

对应 writer 最终调用 PhysX Tensor API：

```text
articulation.py:638  root_physx_view.set_dof_stiffnesses(...)
articulation.py:667  root_physx_view.set_dof_dampings(...)
articulation.py:762  root_physx_view.set_dof_max_velocities(...)
articulation.py:797  root_physx_view.set_dof_max_forces(...)
```

如果 actuator cfg 里某些参数是 `None`，`ActuatorBase` 会从 USD joint prim / 当前 PhysX 默认值继承。源码：`actuator_base.py:173` 到 `:215`。

## 8. sim.step 最后怎么进 PhysX

Isaac Lab 的 `SimulationContext.step(render=False)` 在：

```text
/home/gxai/IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py:525
```

它做完异常检查、暂停处理后调用父类 `super().step(render=render)`，源码：`simulation_context.py:563`。

Isaac Sim core 的 `SimulationContext.step()` 最后会调用 `_physics_context._step(...)`，源码：

```text
/home/gxai/IsaacLab/_isaac_sim/exts/isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py:672
```

再往下，`SimulationManager.step()` 用 PhysX interface 做：

```python
_physx_sim_interface.simulate(physics_dt, simulation_time)
_physx_sim_interface.fetch_results()
```

源码：

```text
/home/gxai/IsaacLab/_isaac_sim/exts/isaacsim.core.simulation_manager/isaacsim/core/simulation_manager/impl/simulation_manager.py:292
/home/gxai/IsaacLab/_isaac_sim/exts/isaacsim.core.simulation_manager/isaacsim/core/simulation_manager/impl/simulation_manager.py:297
```

这就是 Python 源码能看到的最后一层；再往下就是 Isaac Sim / PhysX native backend。

## 9. target 写入和 state 写入是两条路

这是最容易混淆的点。

### 9.1 target 路线：控制器/执行器路线

Isaac Lab 原生 JointPositionAction / RelativeJointPositionAction 走的是 target route：

```text
set_joint_position_target(...)
-> _data.joint_pos_target buffer
-> scene.write_data_to_sim()
-> articulation.write_data_to_sim()
-> actuator.compute(...)
-> root_physx_view.set_dof_position_targets(...)
-> sim.step()
-> PhysX 用 drive stiffness/damping/limits 积分
```

这条路的结果不是瞬间到位，而是由 PhysX articulation drive 和 solver 追 target。

调参位置：

- action scale/offset/clip：action cfg。
- actuator stiffness/damping/limits：`ArticulationCfg.actuators` 里的 actuator cfg，或 USD joint prim 默认值。
- physics dt / decimation / solver：env sim cfg / `SimulationCfg.physx`。

### 9.2 state 路线：直接改仿真状态

`Articulation.write_joint_state_to_sim(position, velocity)` 是另一条路：

```text
write_joint_state_to_sim(...)
-> write_joint_position_to_sim(...)
-> root_physx_view.set_dof_positions(...)
-> write_joint_velocity_to_sim(...)
-> root_physx_view.set_dof_velocities(...)
```

源码：

```text
articulation.py:520  write_joint_state_to_sim
articulation.py:536  write_joint_position_to_sim
articulation.py:537  write_joint_velocity_to_sim
articulation.py:575  root_physx_view.set_dof_positions(...)
articulation.py:605  root_physx_view.set_dof_velocities(...)
```

这不是“让 PD drive 追目标”，而是直接把 articulation 的 generalized coordinates / velocities 写成给定值。它会绕过正常的动态追踪过程。

## 10. 你的运动学 direct-write 方法和 Isaac 原生方法的区别

你当前自定义 action term 是：

```text
configs/xarm7_pick_pose_env_cfg.py:133  KinematicRelativeJointDirectAction
```

它的语义是 joint-space kinematic integration：

```python
delta_q = clip(action * scale, -clip, clip)
q_target = clamp(q_current_at_action_time + delta_q, joint_limits)
```

关键源码：

```text
configs/xarm7_pick_pose_env_cfg.py:207  process_actions
configs/xarm7_pick_pose_env_cfg.py:212  delta_q = actions * scale
configs/xarm7_pick_pose_env_cfg.py:213  delta_q clip
configs/xarm7_pick_pose_env_cfg.py:217  q_current + delta_q
configs/xarm7_pick_pose_env_cfg.py:220  joint limit clamp
configs/xarm7_pick_pose_env_cfg.py:221  cache q_target
configs/xarm7_pick_pose_env_cfg.py:272  apply_actions
configs/xarm7_pick_pose_env_cfg.py:281  joint_pos = q_target
configs/xarm7_pick_pose_env_cfg.py:286  write_joint_state_to_sim(...)
```

### 10.1 和 Isaac 原生 RelativeJointPositionAction 的区别

Isaac 原生相对位置 action：

```text
process_actions: raw -> scale/offset/clip
apply_actions: current_joint_pos + processed_actions -> set_joint_position_target
```

源码：`joint_actions.py:168` 和 `joint_actions.py:227`。

差异：

| 维度 | Isaac 原生 RelativeJointPositionAction | 你的 KinematicRelativeJointDirectAction |
|------|----------------------------------------|------------------------------------------|
| 目标计算次数 | `apply_actions()` 每个 physics step 调一次 | `process_actions()` 每个 env step 只算一次并缓存 |
| decimation 影响 | `decimation > 1` 时，每个 substep 都可能基于当前 joint_pos 再加一次 delta | 同一个 `q_target` 在所有 substep 里重复使用，不会重复加 delta |
| 写入方式 | `set_joint_position_target`，走 actuator/PhysX drive target | `write_joint_state_to_sim`，直接写 joint position/velocity |
| 物理跟踪 | 有，受 stiffness/damping/effort/velocity/solver 影响 | 基本绕过跟踪误差，关节被直接设到目标 |
| 速度 | 由 PhysX 积分和 drive 产生 | 你显式把 joint_vel 写 0 |
| 动力学真实性 | 更像真实伺服/关节驱动 | 更像运动学状态投影/teleport |
| 稳定性 | 可能有跟踪滞后、超调、限力、接触反作用 | 更稳定、更可控，但接触/惯性/执行器限制不真实 |

### 10.2 和 Isaac 原生 JointPositionAction 的区别

Isaac 原生绝对位置 action 每个 substep 写同一个 processed absolute target，因此没有“相对 delta 重复累加”的问题；但它仍然只是 target route：

```text
set_joint_position_target -> PhysX drive 追踪
```

你的方法则是 state route：

```text
write_joint_state_to_sim -> 直接改 q 和 qdot
```

所以即便目标角一样，两者也不是同一个控制器。

### 10.3 和 IK / task-space 方法的区别

IK / task-space action 的上游输入通常是末端位姿或 twist，然后通过 IK/controller 求 joint target。它和你的方法的核心区别不在“是不是位置目标”，而在两层：

1. 上游：IK 从 task-space 解 joint target；你的方法直接在 joint-space 加 delta。
2. 下游：IK action 通常仍走 `set_joint_position_target` target route；你的方法直接 `write_joint_state_to_sim`。

所以你的方法不是 Isaac 的 IK，也不是 Isaac 的 drive controller，而是 joint-space kinematic direct state writer。

## 11. 参数到底在哪里调

按层看：

| 层 | 参数 | 源码定义 | 任务中通常在哪里配 |
|----|------|----------|--------------------|
| 时间步 | `sim.dt` | `SimulationCfg.dt`，`simulation_cfg.py:357` | env cfg 的 `sim.dt` |
| policy/physics 比例 | `decimation` | `ManagerBasedEnvCfg.decimation`，`manager_based_env_cfg.py:70` | env cfg 的 `decimation` |
| 渲染频率 | `sim.render_interval` | `SimulationCfg.render_interval`，`simulation_cfg.py:360` | env cfg 的 `sim.render_interval` |
| action term 绑定资产 | `asset_name` | `ActionTermCfg.asset_name`，`manager_term_cfg.py:78` | `actions.xxx.asset_name` |
| action 绑定关节 | `joint_names` | `JointActionCfg.joint_names`，`actions_cfg.py:31` | `actions.xxx.joint_names` |
| action 缩放 | `scale` | `JointActionCfg.scale`，`actions_cfg.py:33` | `actions.xxx.scale` |
| action offset | `offset` | `JointActionCfg.offset`，`actions_cfg.py:35` | `actions.xxx.offset` |
| action clip | `clip` | `ActionTermCfg.clip`，`manager_term_cfg.py:96` | `actions.xxx.clip` |
| 关节驱动分组 | `joint_names_expr` | `ActuatorBaseCfg.joint_names_expr`，`actuator_base_cfg.py:21` | `ArticulationCfg.actuators` |
| P gain | `stiffness` | `ActuatorBaseCfg.stiffness`，`actuator_base_cfg.py:107` | actuator cfg 或 USD joint prim |
| D gain | `damping` | `ActuatorBaseCfg.damping`，`actuator_base_cfg.py:117` | actuator cfg 或 USD joint prim |
| solver effort limit | `effort_limit_sim` | `ActuatorBaseCfg.effort_limit_sim`，`actuator_base_cfg.py:74` | actuator cfg 或 USD joint prim |
| solver velocity limit | `velocity_limit_sim` | `ActuatorBaseCfg.velocity_limit_sim`，`actuator_base_cfg.py:91` | actuator cfg 或 USD joint prim |
| runtime stiffness write | `write_joint_stiffness_to_sim` | `articulation.py:611` | 代码运行时直接调用 |
| runtime damping write | `write_joint_damping_to_sim` | `articulation.py:640` | 代码运行时直接调用 |
| runtime state write | `write_joint_state_to_sim` | `articulation.py:520` | 自定义 action / reset / debug |

你的自定义方法额外有这些非 Isaac 原生字段：

| 参数 | 位置 | 含义 |
|------|------|------|
| `scale` | `KinematicRelativeJointDirectActionCfg` | raw action 到 joint delta 的缩放 |
| `clip` | `KinematicRelativeJointDirectActionCfg` | 每步最大 joint delta |
| `joint_limits_low/high` | `KinematicRelativeJointDirectActionCfg` | 直接写 state 前的安全 clamp |

源码：`configs/xarm7_pick_pose_env_cfg.py:306` 到 `:316`。

## 12. 调试时看哪一层

如果怀疑 action 没进来：看 `ActionManager._action` / term `raw_actions`。

如果怀疑 action 被缩放错：看 term `processed_actions`。

如果用 Isaac 原生 action，怀疑机器人不跟目标：看 articulation 的 `joint_pos_target`、`_joint_pos_target_sim`、stiffness/damping/effort/velocity limit。

如果用你的 kinematic direct-write，怀疑位置没到：看 `write_joint_state_to_sim` 后的 `asset.data.joint_pos`，以及后续是否被 `scene.update` 或其他 term 覆盖。

如果怀疑 decimation 影响：先分清 action term 是“target route”还是“state route”。原生 RelativeJointPositionAction 会每个 physics substep 重新 `current + delta`；你的方法只缓存一次 `q_target`。
