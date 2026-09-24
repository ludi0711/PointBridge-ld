# Copyright (c) 2024, Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class XArm7ReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000  # Increase from 1000 to 2000
    save_interval = 50
    experiment_name = "xarm7_reach"
    run_name = ""
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 64],  # Increase network size
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7LiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """xArm7 Lift 任务的 PPO 训练配置"""

    # 训练参数
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "xarm7_lift"
    empirical_normalization = False

    # PPO 算法配置
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick 任务的 PPO 训练配置（基于 Mujoco v4.1）"""

    # 训练参数
    num_steps_per_env = 24
    max_iterations = 50000  # 50k iterations
    save_interval = 500  # 每 500 次保存一次
    experiment_name = "xarm7_pick"
    empirical_normalization = False

    # PPO 算法配置
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],  # 更大的网络
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # 稍高的熵系数，鼓励探索
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )



@configclass
class XArm7PickPPORunnerCfg_V10(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v10 — EE 对齐工件任务 PPO 配置

    对齐任务比举起简单，使用较小规模的训练：
    - max_iterations: 20000（对齐比举起容易学，不需要 20 万次）
    - 网络规模与 LiftCube 相同
    - 较高学习率（1e-3），加快收敛
    """

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v10"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V11(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v11 — 增量关节控制对齐任务 PPO 配置"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v11"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V12(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v12 — 增量控制 + 姿态误差观测 PPO 配置"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v12"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V13(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v13 — 从对齐姿态举升工件 PPO 配置"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v13"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V14(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v14 — EE 举升 + 保持初始姿态 PPO 配置"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v14"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V15(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v15 — 基于 v12 PPO 配置"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v15"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V20(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v20 — 基于 v15 PPO 配置"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v20"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V30(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v30 — theia-tiny 视觉特征 (192) + joint_pos (7) = 199 维输入

    网络使用 [512, 256, 128]，比 v12 更大以处理高维视觉特征。
    theia-tiny 权重冻结，policy 只学习从视觉特征到动作的映射。
    """

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v30"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V31(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v31 — v30 + 碰撞惩罚"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v31"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V32(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v32 — 手腕相机，[256, 256, 256] 网络"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v32"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V35(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v35 — 双相机 DeFM depth (384×2) + joint_pos(7) + action(7) = 782 维输入

    网络使用 [512, 512, 256]，与 v33 保持一致以便对比。
    """

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v35"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 512, 256],
        critic_hidden_dims=[512, 512, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V36(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v36 — theia(192×2) + DeFM(384×2) + joint_pos(7) + action(7) = 1166 维输入

    网络使用 [1024, 512, 256] 应对更大输入维度。
    """

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v36"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V37(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v37 — gongjian.usd 工件 + theia(192×2) + DeFM(384×2) = 1166 维输入"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v37"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V33(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v33 — 双相机 theia (192×2) + joint_pos(7) + action(7) = 398 维输入

    网络使用 [512, 512, 256] 应对更大输入维度。
    """

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v33"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 512, 256],
        critic_hidden_dims=[512, 512, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    """xArm7 Lift v21 — EE 举升任务 PPO 配置"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_lift_v21"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


    """xArm7 Pick 任务的 PPO 训练配置 - Lift-Cube 风格（大规模训练版）

    参考 Isaac Lab 标准 Lift-Cube 任务的 PPO 参数：
    - 更小的学习率（1e-4 vs 1e-3）
    - 更小的折扣因子（0.98 vs 0.99）
    - 更小的熵系数（0.006 vs 0.01）
    - 递减网络结构（256→128→64）
    - 大规模训练：20 万次迭代，1000 个环境
    """

    # 训练参数
    num_steps_per_env = 24
    max_iterations = 200000  # 20 万次迭代，充分训练
    save_interval = 100  # 每 100 次保存一次
    experiment_name = "xarm7_pick_liftcube"
    empirical_normalization = False

    # PPO 算法配置
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,  # 更小的学习率，对齐 Lift-Cube
        schedule="adaptive",
        gamma=0.98,  # 更小的折扣因子，更关注短期奖励
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V38(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v38 — gongjian.usd 工件 + 仅全局相机 theia(192) + DeFM(384) = 590 维输入"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v38"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V39(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v39 — 桌子颜色随机化 + 仅全局相机 theia(192) + DeFM(384) = 590 维输入"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v39"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )



@configclass
class XArm7PickPPORunnerCfg_V40(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v40 — grip_on_falan 末端工具 + 桌子颜色随机化 + 全局相机 theia(192) + DeFM(384) = 590 维输入"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v40"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V41(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v41 — grip 碰撞检测 + 桌子颜色随机化 + 全局相机 theia(192) + DeFM(384) = 590 维输入"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v41"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V43(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v43 — D435 对齐相机 + 仅 Theia 视觉编码器 theia(192) = 206 维输入"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v43"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V52(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v52 — 单固定相机 Theia，206 维，两段式奖励"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v52"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V51(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v51 — 双相机纯 Theia，398 维，两段式奖励"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v51"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V50(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v50 — 双相机 Theia+PointBERT，782 维，两段式奖励"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v50"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V53(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v53 — FP16 编码器，782 维，两段式奖励"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v53"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickLiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick LiftCube — FP16 编码器，782 维，两段式奖励"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_vision"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V55(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v55 — v54 reward-fix，建议从 v54 model_1000/1100 微调。"""

    num_steps_per_env = 24
    max_iterations = 12000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v55"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V56(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v56 — 新法兰，超参与 v55 完全一致。"""

    num_steps_per_env = 24
    max_iterations = 12000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v56"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V57(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v57 — 新法兰，超参与 v54 完全一致。"""

    num_steps_per_env = 24
    max_iterations = 12000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v57"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V58(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v58 — 新法兰 + 纯视觉（无点云），398维观测，超参与 v57 完全一致。"""

    num_steps_per_env = 24
    max_iterations = 12000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v58"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PlacePPORunnerCfg_V1(RslRlOnPolicyRunnerCfg):
    """xArm7 Place v1 — 纯状态 RL，35 维输入，[256, 256, 128] 网络。"""

    num_steps_per_env = 32
    max_iterations    = 10000
    save_interval     = 100
    experiment_name   = "xarm7_place_v1"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V47(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v47 — 单固定相机 Theia，206 维输入"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v47"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V46(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v46 — 双相机纯 Theia，398 维输入"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v46"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V45(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v45 — 双相机 Theia+PointBERT，782 维输入"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v45"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPPORunnerCfg_V44(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick v44 — 桌子高度随机化 ±5cm + 工件 z 联动，206 维输入"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "xarm7_pick_liftcube_v44"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PickPosePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """xArm7 Pick 位姿真值训练 — 28维状态观测，动作硬限幅 ±0.6°/step。"""

    num_steps_per_env = 24
    max_iterations    = 12000
    save_interval     = 100
    experiment_name   = "xarm7_pick_pose"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class RslRlPointNetActorCriticCfg(RslRlPpoActorCriticCfg):
    """PointNet actor + privileged MLP critic 的策略配置。

    在内置 RslRlPpoActorCriticCfg 上追加 PointNet 专属字段。这些字段作为
    **kwargs 传给 model/pointnet_actor_critic.py 的 PointNetActorCritic。
    """

    class_name: str = "PointNetActorCritic"

    num_object_points: int = 64
    """物体点数 = M_OBJ。须与环境侧一致。参考实现默认 128。"""

    num_robot_points: int = 6
    """夹爪关键点数 = N_ROBOT。参考实现是 8+1=9（Franka Hand 布局）。"""

    num_wrist_points: int = 0
    """腕部相机点数 = M_WRIST，**0 = 不启用腕部分支**（stage -2..4 的默认）。

    stage 10 由训练脚本置为 M_WRIST(64)：actor 观测 217 → 409 维，网络多一个独立
    权重的腕部 encoder。这个值与观测维度绑死 —— 改了它就必须重训，既有 checkpoint
    加载会直接形状不匹配（这是好事，比静默错位好）。"""

    point_dim: int = 3
    """每点维度 = xyz。对齐参考实现 PointNetEncoderXYZ 的 in_channels=3。
    身份不靠类型通道，而靠物体点/夹爪点分开编码成两个 token。"""

    pointnet_hidden: tuple = (64, 128, 256)
    """逐点共享 MLP 层宽。对齐参考实现 dp3_encoder.py 的 block_channel。"""

    pointnet_out_dim: int = 512
    """PointNet 输出维度。对齐参考实现的 repr_dim=512。
    注意论文里的 hidden_dim=256 是 GPT trunk 宽度，不是 PointNet 输出维度。"""

    normalize_points: bool = True
    """是否对点云做固定 min-max 仿射归一化（对齐参考实现的 past_tracks 预处理）。
    参考实现用数据集 stats；RL 无预先数据集，改用固定工作空间边界。"""

    point_norm_min: tuple = (-1.0, -1.0, -0.10)
    """归一化下界（基座系，米）。物体点与夹爪点共用同一套。"""

    point_norm_max: tuple = (1.0, 1.0, 0.60)
    """归一化上界（基座系，米）。"""


@configclass
class XArm7PickPointCloudPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Point Bridge 点表征 PPO — actor 吃点云(64+6 点, xyz)+关节角，critic 吃 privileged 状态。

    超参完全沿用 V58（规格 §6：只有每物体点数 / hidden dim / H 三项对齐原工作，
    RL 算法与学习率一概不动）。唯一新增的是 obs_groups 与策略类。
    """

    num_steps_per_env = 24
    max_iterations = 12000
    save_interval = 100
    experiment_name = "xarm7_pick_pointcloud"
    empirical_normalization = False

    # actor 只看降级观测，critic 看未降级的 privileged 状态（规格 §5.3）。
    # 这是本方案最重要的调试杠杆：两侧同时降级会让"不收敛"无法归因。
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = RslRlPointNetActorCriticCfg(
        init_noise_std=1.0,
        # actor 归一化关闭：EmpiricalNormalization 逐维统计，而 FPS 后第 k 个点
        # 无固定语义；且它会削掉绝对位置尺度，而绝对位置正是全部任务信息。
        actor_obs_normalization=False,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class XArm7PickPrivilegedBasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """privileged base — actor 与 critic 都吃 GT 状态（35 维），无点云无相机。

    整条链路的最上界，用于验证环境/奖励本身可解：

        privileged base  状态真值        → 不受表征限制
        stage 1          点表征 + GT 位姿 → 受点表征限制
        stage 2          点表征 + 掩码深度 → 表征 + 感知都受限

    这一版学不出来 → 问题在环境（奖励权重 / 动作限幅 / episode 长度 / 碰撞阈值），
    与点云、相机、外参、分割全都无关。学得好而 stage 1 崩 → 才轮到怀疑点表征。

    用内置 ActorCritic（纯 MLP），不用 PointNetActorCritic —— 没有点云要编码。
    超参与 XArm7PickPointCloudPPORunnerCfg 逐项一致，只有观测和策略类不同，
    这样两者的曲线可以直接并排比。
    """

    num_steps_per_env = 24
    max_iterations = 12000
    save_interval = 100
    experiment_name = "xarm7_pick_privileged_base"
    empirical_normalization = False

    # 两组都指向 privileged 状态。actor 与 critic 内容相同但各自独立归一化。
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        # 纯状态观测（位姿 / 关节角 / 速度 / 动作）各维语义固定、量纲不同，
        # 逐维归一化在这里是合适的 —— 与点云的情形不同。
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
