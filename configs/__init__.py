# Copyright (c) 2024, Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration package for xArm7 USD test."""

import gymnasium as gym

# 注册 Reach 任务
gym.register(
    id="XArm7-Reach-Workbench-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "configs.reach_workbench_env_cfg:XArm7ReachWorkbenchEnvCfg",
    },
    disable_env_checker=True,
)

# 注册 Lift 任务
gym.register(
    id="XArm7-Lift-Workbench-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "configs.lift_workbench_env_cfg:XArm7LiftWorkbenchEnvCfg",
    },
    disable_env_checker=True,
)

# 注册 Lift 任务 - 播放模式
gym.register(
    id="XArm7-Lift-Workbench-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "configs.lift_workbench_env_cfg:XArm7LiftWorkbenchEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

