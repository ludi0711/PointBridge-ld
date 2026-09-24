"""
BMW Assembly Staged 环境 - 课程学习版本 PickPlace 风格 V4.3
============================================================

**V4.3 改进**：精简观测空间，移除冗余信息
- 观测维度：37维 → 23维
- 移除绝对位置和姿态，只保留相对信息
- 观测结构更清晰：机器人状态 + 相对位姿信息
- 减少维度灾难，提高学习效率

观测空间结构（23维）：
```
# === 机器人状态 (9维) ===
eef_pos (3)              # 末端位置
eef_quat (4)             # 末端姿态
gripper_state (1)        # 夹爪状态
****obj_lifted (1)       # 是否抬起

# === 相对位姿信息 (14维) ===
obj_rel_eef_pos (3)      # 工件相对夹爪的位置
obj_rel_eef_quat (4)     # 工件相对夹爪的姿态
obj_rel_target_pos (3)   # 工件相对工装的位置
obj_rel_target_quat (4)  # 工件相对工装的姿态
```

**V4.2 改进回顾**：观测空间增广，添加相对位姿信息
- 新增工装位置和姿态（alignment_frame）
- 新增工件相对工装的位置和姿态（obj_rel_target）
- 四元数归一化（w > 0），消除 antipodal ambiguity
- 观测维度：19 → 37 维

**V4.1 改进回顾**：放宽 place 奖励的激活条件
- 移除 place 阶段的 XY 硬截断（place_xy_threshold）
- place 奖励同时考虑 XY 和 Z 距离，使用 3D 欧氏距离
- 让智能体更容易获得 place 奖励的梯度引导
- 解决 V4 中智能体在 hover 阶段持续向上抬的问题

核心设计理念：
1. 使用 max(staged_rewards) 替代累加
2. 所有奖励都是持续性的（基于当前状态）
3. 奖励函数连续可微
4. 松开物体后奖励立即回退，形成隐式惩罚

奖励结构：
- reward = max(r_reach, r_grasp, r_lift, r_hover, r_place) + r_action_penalty + r_success

阶段性奖励范围（严格递增，不重叠，连续）：
- r_reach ∈ [0, 1.0]      接近物体
- r_grasp = 1.5           稳定抓取（持续）
- r_lift ∈ (1.5, 4.0]     抬起物体（持续）
- r_hover ∈ (4.0, 7.0]    移动到目标上方（持续）
- r_place ∈ (7.0, 10.0]   放置到目标位置（持续）

终止奖励：
- success = +1000.0       完成任务（放置成功且松开夹爪）

课程学习功能：
- 支持 approach, grasp, lift, hover, place 五个阶段
- 支持 random 模式（加权随机选择阶段）
- 预热 + 随机游走机制
- 使用 BMWAssemblyStagedV3 底层环境
"""

import sys
import os
import warnings
import logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['GYM_IGNORE_DEPRECATION_WARNINGS'] = '1'
logging.getLogger('robosuite').setLevel(logging.CRITICAL)

from gops.env.env_ocp.robosuite_init import suite

import gym
import numpy as np
from gym import spaces
from gops.env.env_ocp.pyth_base_env import PythBaseEnv


class PythRobosuiteBmwAssemblyStagedCurriculumPickPlaceV4_3(PythBaseEnv):
    """BMW装配环境 - 课程学习 PickPlace 风格奖励 V4.3（精简观测空间）"""
    
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(self, **kwargs):
        # 核心参数
        self.robot_name = kwargs.pop("robosuite_robot", "XArm7")
        self.control_freq = kwargs.pop("control_freq", 20)
        self.horizon = kwargs.pop("horizon", 200)
        self.is_render = kwargs.pop("is_render", False)
        self.verbose = kwargs.pop("verbose", False)
        
        # 使用显式参数 is_eval
        self.is_eval = kwargs.pop("is_eval", False)
        
        # 课程学习参数
        curriculum_stage_arg = kwargs.pop("curriculum_stage", None)
        curriculum_stages_arg = kwargs.pop("curriculum_stages", None)
        curriculum_weights_arg = kwargs.pop("curriculum_weights", None)
        self.randomize_steps = kwargs.pop("randomize_steps", 50)
        self.randomize_action_scale = kwargs.pop("randomize_action_scale", 0.3)
        self.warmup_verbose = kwargs.pop("warmup_verbose", False)
        
        # 根据 is_eval 决定是否使用课程学习
        if self.is_eval:
            self.curriculum_stage = None
            if self.verbose:
                print("[Curriculum PickPlace V4] Evaluator mode: curriculum learning disabled")
        else:
            self.curriculum_stage = curriculum_stage_arg
            if self.verbose and self.curriculum_stage:
                if self.curriculum_stage == 'random':
                    print(f"[Curriculum PickPlace V4] Sampler mode: random curriculum learning enabled")
                else:
                    print(f"[Curriculum PickPlace V4] Sampler mode: curriculum learning enabled (stage={self.curriculum_stage})")

        # 创建底层 robosuite 环境（使用 BMWAssemblyStagedV3）
        self._env = suite.make(
            env_name="BMWAssemblyStagedV3",
            robots=self.robot_name,
            gripper_types="CustomXArm7Gripper",
            use_camera_obs=False,
            use_object_obs=True,
            has_renderer=self.is_render,
            has_offscreen_renderer=False,
            control_freq=self.control_freq,
            horizon=self.horizon,
            reward_shaping=True,
            curriculum_stage=self.curriculum_stage,
            curriculum_stages=curriculum_stages_arg,
            curriculum_weights=curriculum_weights_arg,
            randomize_steps=self.randomize_steps,
            randomize_action_scale=self.randomize_action_scale,
            warmup_verbose=self.warmup_verbose,
        )

        # 观测空间：23维（V4.3：精简观测空间）
        obs_dim = 23
        obs_low = np.full(obs_dim, -np.inf, dtype=np.float32)
        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        work_space = np.stack((obs_low, obs_high))
        
        super(PythRobosuiteBmwAssemblyStagedCurriculumPickPlaceV4_3, self).__init__(work_space=work_space, **kwargs)
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0] * self._env.action_dim, dtype=np.float32),
            high=np.array([1.0] * self._env.action_dim, dtype=np.float32),
            dtype=np.float32
        )
        self.max_episode_steps = self.horizon
        
        # ==================== PickPlace V4 风格奖励权重 ====================
        # 阶段性奖励倍数（范围不重叠，连续）
        self.reach_mult = 1.0        # r_reach ∈ [0, 1.0]
        self.grasp_mult = 1.5        # r_grasp = 1.5
        self.lift_mult = 4.0         # r_lift 最大值
        self.hover_mult = 7.0        # r_hover 最大值【V4：增益更大】
        self.place_mult = 10.0       # r_place 最大值【V4：增益更大】
        
        # tanh scale 因子
        self.reach_scale = 10.0      # reach tanh scale
        self.lift_scale = 15.0       # lift tanh scale
        self.hover_scale = 10.0      # hover tanh scale【V4 新增】
        self.place_scale = 15.0      # place tanh scale【V4 新增】
        
        # 终止奖励
        self.w_success = 1000.0      # 成功奖励
        
        # 动作惩罚（独立于阶段奖励）
        self.w_action_penalty = 0.01
        
        # 稳定性判定
        self.grasp_stable_steps = 1
        self.grasp_counter = 0
        self.stable_grasped = False
        
        # 高度和距离阈值【V4 新增】
        self.table_height = 0.86
        self.target_lift_height = 0.30       # 抬起高度
        self.hover_height_min = 0.20         # hover 最小高度
        self.hover_xy_threshold = 0.10       # hover XY 对齐阈值（10cm）
        self.place_xy_threshold = 0.01       # place XY 对齐阈值（10mm，参考 batch_test）
        self.place_z_threshold = 0.02        # place Z 放置阈值（20mm，参考 batch_test）
        
        # 局部 RNG
        self._rng = None
        
        # 当前动作（用于grasp判定）
        self._current_action = None
        
        # 状态标志
        self.obs = None
        self.lifted = False
        self.hovered = False         # 物体在目标上方【V4 新增】
        self.placed = False          # 物体放置成功【V4 新增】

    def reset(self, **kwargs):
        """重置环境"""
        robosuite_obs = self._env.reset()
        self.obs = self._extract_observation(robosuite_obs)
        
        # 重置状态
        self.lifted = False
        self.hovered = False
        self.placed = False
        self.stable_grasped = False
        self.grasp_counter = 0
        
        return self.obs

    def step(self, action: np.ndarray):
        """执行一步"""
        action = np.clip(action, -1.0, 1.0)
        
        # 保存当前动作（用于grasp判定）
        self._current_action = action
        
        # 执行动作
        robosuite_obs, _, robosuite_done, info = self._env.step(action)
        self.obs = self._extract_observation(robosuite_obs)
        
        # 检查是否在预热或随机游走阶段
        in_warmup = info.get('in_warmup', False)
        in_randomize = info.get('in_randomize', False)
        
        # 只在正常训练阶段计算真实奖励
        if in_warmup or in_randomize:
            # 预热和随机游走阶段：所有奖励分量为0
            reward = 0.0
            # 使用 evaluator1 期望的 key 名称
            info['reward_reach_shape'] = 0.0
            info['reward_grasp_event'] = 0.0
            info['reward_lift_shape'] = 0.0
            info['reward_hold'] = 0.0
            info['reward_action'] = 0.0
            info['reward_time'] = 0.0
            info['reward_drop'] = 0.0
            info['reward_success'] = 0.0
            info['reward_total'] = 0.0
        else:
            # ==================== PickPlace V4 风格奖励计算 ====================
            # 计算阶段性奖励
            r_reach, r_grasp, r_lift, r_hover, r_place = self._staged_rewards()
            
            # 取最大值（核心：max() 而非累加）
            reward_stage = max(r_reach, r_grasp, r_lift, r_hover, r_place)
            
            # 动作惩罚
            r_action = -self.w_action_penalty * np.sum(action ** 2)
            
            # 总奖励
            reward = float(reward_stage + r_action)
            
            # 记录奖励分量到 info
            info['reward_reach_shape'] = float(r_reach)
            info['reward_grasp_event'] = float(r_grasp)
            info['reward_lift_shape'] = float(r_lift)
            info['reward_hover_shape'] = float(r_hover)  # V4 新增
            info['reward_place_shape'] = float(r_place)  # V4 新增
            info['reward_hold'] = 0.0
            info['reward_action'] = float(r_action)
            info['reward_time'] = 0.0
            info['reward_total'] = float(reward)
            
            # 检查成功
            if self._check_success():
                reward += self.w_success
                robosuite_done = True
                info['success'] = True
                info['reward_success'] = float(self.w_success)
            else:
                info['reward_success'] = 0.0
            
            info['reward_drop'] = 0.0
        
        # 暴露关键信息
        info['stable_grasped'] = self.stable_grasped
        info['lifted'] = self.lifted
        info['hovered'] = self.hovered  # V4 新增
        info['placed'] = self.placed    # V4 新增
        info['grasp_counter'] = self.grasp_counter
        info['obj_height'] = self._get_obj_height_safe()
        info['obj_pos'] = self._get_obj_pos_safe()
        info['eef_pos'] = self._get_eef_pos_safe()
        info['target_pos'] = self._get_target_pos_safe()  # V4 新增
        
        return self.obs, reward, robosuite_done, info

    def _extract_observation(self, robosuite_obs):
        """
        提取23维观测（V4.3：精简观测空间）
        
        观测结构：
        - 机器人状态 (9维)：eef_pos, eef_quat, gripper_state, obj_lifted
        - 相对位姿 (14维)：obj_rel_eef (pos+quat), obj_rel_target (pos+quat)
        """
        # ==================== 机器人状态 (9维) ====================
        eef_pos = self._get_eef_pos_safe()  # 3
        
        obs_dict = self._env._get_observations()
        eef_quat = self._normalize_quat(obs_dict.get('grip_site_quat', np.zeros(4)))  # 4
        
        gripper_qpos = obs_dict.get('robot0_gripper_qpos', np.zeros(6))
        gripper_state = np.array([gripper_qpos[2]])  # 1
        
        obj_height = self._get_obj_height_safe()
        obj_lifted = np.array([1.0 if obj_height > self.table_height + 0.03 else 0.0])  # 1
        
        # ==================== 相对位姿信息 (14维) ====================
        # 工件位置和姿态（用于计算相对信息）
        obj_pos = self._get_obj_pos_safe()
        obj_quat = self._normalize_quat(self._get_obj_quat_safe())
        
        # 工装位置和姿态（用于计算相对信息）
        target_pos = self._get_target_pos_safe()
        target_quat = self._normalize_quat(self._get_target_quat_safe())
        
        # 工件相对夹爪的位置和姿态
        obj_rel_eef_pos = obj_pos - eef_pos  # 3
        obj_rel_eef_quat = self._compute_relative_quat(eef_quat, obj_quat)  # 4
        
        # 工件相对工装的位置和姿态
        obj_rel_target_pos = obj_pos - target_pos  # 3
        obj_rel_target_quat = self._compute_relative_quat(target_quat, obj_quat)  # 4
        
        # ==================== 组合观测 (23维) ====================
        obs = np.concatenate([
            # 机器人状态 (9维)
            eef_pos,                # 3
            eef_quat,               # 4
            gripper_state,          # 1
            obj_lifted,             # 1
            
            # 相对位姿信息 (14维)
            obj_rel_eef_pos,        # 3
            obj_rel_eef_quat,       # 4
            obj_rel_target_pos,     # 3
            obj_rel_target_quat,    # 4
        ])
        
        return obs.astype(np.float32)

    # 安全的数据获取方法
    def _get_eef_pos_safe(self):
        """安全地获取末端执行器位置（使用 grip_site_pos）"""
        try:
            obs_dict = self._env._get_observations()
            return np.array(obs_dict.get('grip_site_pos', np.zeros(3)))
        except:
            return np.zeros(3)
    
    def _get_obj_pos_safe(self):
        """安全地获取物体抓取点位置"""
        try:
            return np.array(self._env.sim.data.site_xpos[self._env.obj_grasp_site_id])
        except:
            return np.zeros(3)
    
    def _get_obj_quat_safe(self):
        """安全地获取物体抓取点姿态（使用 grasp site）"""
        try:
            from robosuite.utils.transform_utils import mat2quat
            # 使用 grasp site 的旋转矩阵转换为四元数
            obj_mat = self._env.sim.data.site_xmat[self._env.obj_grasp_site_id].reshape(3, 3)
            return mat2quat(obj_mat)
        except:
            return np.array([0, 0, 0, 1])
    
    def _get_obj_height_safe(self):
        """安全地获取物体高度（使用 grasp site）"""
        try:
            # 使用 grasp site 的 Z 坐标，而不是 body_xpos
            return self._env.sim.data.site_xpos[self._env.obj_grasp_site_id][2]
        except:
            return self.table_height
    
    def _get_target_pos_safe(self):
        """安全地获取目标位置（alignment_frame_pos）【V4 新增】"""
        try:
            obs_dict = self._env._get_observations()
            target_pos = obs_dict.get('alignment_frame_pos', None)
            if target_pos is not None:
                return np.array(target_pos)
            else:
                # 如果没有目标位置，返回默认值
                return np.array([0.0, 0.0, self.table_height])
        except:
            return np.array([0.0, 0.0, self.table_height])
    
    def _get_target_quat_safe(self):
        """安全地获取目标姿态（alignment_frame_quat）【V4.2 新增】"""
        try:
            obs_dict = self._env._get_observations()
            target_quat = obs_dict.get('alignment_frame_quat', None)
            if target_quat is not None:
                return np.array(target_quat)
            else:
                # 如果没有目标姿态，返回单位四元数
                return np.array([0.0, 0.0, 0.0, 1.0])
        except:
            return np.array([0.0, 0.0, 0.0, 1.0])
    
    def _normalize_quat(self, quat):
        """归一化四元数，确保 w > 0，消除 antipodal ambiguity【V4.2 新增】

        注意：robosuite 使用 [w, x, y, z] 格式，w 在索引 0
        """
        quat = np.array(quat)
        if quat[0] < 0:  # w 分量（第1个元素，索引0）
            return -quat
        return quat
    
    def _compute_relative_quat(self, quat_base, quat_target):
        """计算相对四元数：quat_base^{-1} * quat_target【V4.2 新增】"""
        try:
            from robosuite.utils.transform_utils import quat_multiply, quat_inverse
            rel_quat = quat_multiply(quat_inverse(quat_base), quat_target)
            return self._normalize_quat(rel_quat)
        except:
            return np.array([0.0, 0.0, 0.0, 1.0])

    # ==================== PickPlace V4 风格阶段性奖励 ====================
    
    def _staged_rewards(self):
        """
        计算阶段性奖励（V4 完整版）
        
        Returns:
            5-tuple: (r_reach, r_grasp, r_lift, r_hover, r_place)
            
        V4 新增：
        - hover: 移动到目标上方（XY对齐）
        - place: 放置到目标位置（下降）
        """
        # 阶段1: 接近奖励 [0, 1.0]
        r_reach = self._compute_reach_reward()
        
        # 阶段2: 抓取奖励 {0, 1.5}
        r_grasp = self._compute_grasp_reward()
        
        # 阶段3: 抬起奖励 (1.5, 4.0]
        r_lift = self._compute_lift_reward()
        
        # 阶段4: hover 奖励（动态范围）【V4 改进：传入r_lift】
        r_hover = self._compute_hover_reward(r_lift)
        
        # 阶段5: place 奖励（动态范围）【V4 改进：传入r_hover】
        r_place = self._compute_place_reward(r_hover)
        
        return r_reach, r_grasp, r_lift, r_hover, r_place
    
    def _compute_reach_reward(self):
        """
        接近奖励：[0, 1.0]
        
        基于末端执行器到物体的距离
        """
        eef_pos = self._get_eef_pos_safe()
        obj_pos = self._get_obj_pos_safe()
        distance = np.linalg.norm(eef_pos - obj_pos)
        
        # 使用 tanh 函数，距离越近奖励越高
        r_reach = (1.0 - np.tanh(self.reach_scale * distance)) * self.reach_mult
        
        return r_reach
    
    def _compute_grasp_reward(self):
        """
        抓取奖励：{0, 1.5}
        
        持续性奖励：只要保持稳定抓取就持续给
        """
        # 更新抓取状态（传入当前动作）
        self._update_grasp_state(self._current_action)
        
        # 只要稳定抓取就给奖励
        if self.stable_grasped:
            return self.grasp_mult  # 1.5
        else:
            return 0.0
    
    def _compute_lift_reward(self):
        """
        抬起奖励：[1.5, 4.0]（参考robosuite风格，保留原量级）
        
        使用连续可微的 tanh 函数，从 height > 0 就开始给予梯度
        
        r_lift = grasp_mult + (1 - tanh(scale * z_dist)) * (lift_mult - grasp_mult)
        
        其中：
        - grasp_mult = 1.5（基础值）
        - lift_mult = 4.0（最大值）
        - scale = 15.0（robosuite的值）
        - z_dist = 目标高度 - 当前高度
        
        **持续性**：只要保持抓取并抬起就持续给奖励
        """
        if not self.stable_grasped:
            return 0.0
        
        obj_height = self._get_obj_height_safe()
        
        # 计算到目标高度的距离
        z_target = self.table_height + self.target_lift_height
        z_dist = max(0.0, z_target - obj_height)
        
        # 参考robosuite的公式
        r_lift = self.grasp_mult + (1.0 - np.tanh(self.lift_scale * z_dist)) * (self.lift_mult - self.grasp_mult)
        
        # 更新 lifted 状态（用于成功判定）
        if z_dist < 0.05:  # 接近目标高度
            self.lifted = True
        
        return r_lift
    
    def _compute_hover_reward(self, r_lift):
        """
        Hover 奖励：动态范围【V4：robosuite风格 + 原量级】
        
        将物体移动到目标位置上方（XY对齐）
        
        参考robosuite的设计：
        - 在目标上方：base = lift_mult (4.0)
        - 不在目标上方：base = r_lift（已计算的lift奖励）
        - 增益部分固定：hover_mult - lift_mult = 3.0
        
        Args:
            r_lift: 已计算的lift奖励值
        
        条件：
        1. 必须保持稳定抓取
        2. 物体已抬起
        3. 基于 XY 距离给予奖励
        
        持续性：只要保持抓取、已抬起、且在向目标移动就持续给奖励
        """
        if not self.stable_grasped or not self.lifted:
            return 0.0
        
        obj_pos = self._get_obj_pos_safe()
        obj_height = self._get_obj_height_safe()
        target_pos = self._get_target_pos_safe()
        
        # 计算 XY 平面距离
        xy_dist = np.linalg.norm(obj_pos[:2] - target_pos[:2])
        
        # 判断是否在目标上方（参考robosuite：只看XY位置）
        height_ok = obj_height >= self.table_height + self.hover_height_min
        xy_aligned = xy_dist < self.hover_xy_threshold
        object_above_target = height_ok and xy_aligned
        
        # 计算hover增益（固定：hover_mult - lift_mult = 3.0）
        hover_bonus = (self.hover_mult - self.lift_mult) * (1.0 - np.tanh(self.hover_scale * xy_dist))
        
        # 关键：根据位置选择基础值（参考robosuite）
        if object_above_target:
            # 在目标上方：从lift_mult开始
            r_hover = self.lift_mult + hover_bonus
        else:
            # 不在目标上方：从当前r_lift开始（平滑过渡）
            r_hover = r_lift + hover_bonus
        
        # 更新 hovered 状态
        if xy_dist < self.hover_xy_threshold:
            self.hovered = True
        
        return r_hover
    
    def _compute_place_reward(self, r_hover):
        """
        Place 奖励：动态范围【V4.1：移除 XY 硬截断】
        
        将物体放置到目标位置
        
        V4.1 改进：
        - 移除 XY 对齐的硬截断条件
        - 使用 3D 欧氏距离（XYZ）计算 place 奖励
        - 让智能体更容易获得 place 奖励的梯度引导
        
        参考robosuite的设计：
        - 在目标位置附近：base = hover_mult (7.0)
        - 不在目标位置：base = r_hover（已计算的hover奖励）
        - 增益部分固定：place_mult - hover_mult = 3.0
        
        Args:
            r_hover: 已计算的hover奖励值
        
        条件：
        1. 必须先完成hover（lifted状态）
        2. 基于 3D 距离给予奖励（XYZ 同时考虑）
        
        持续性：只要在向目标位置移动就持续给奖励
        """
        if not self.hovered:
            return 0.0  # 必须先完成 hover
        
        obj_pos = self._get_obj_pos_safe()
        target_pos = self._get_target_pos_safe()
        
        # V4.1 改进：使用 3D 欧氏距离，同时考虑 XY 和 Z
        dist_3d = np.linalg.norm(obj_pos - target_pos)
        
        # 判断是否在目标位置附近（使用 3D 距离）
        object_near_target = dist_3d < 0.05  # 5cm
        
        # 计算place增益（固定：place_mult - hover_mult = 3.0）
        place_bonus = (self.place_mult - self.hover_mult) * (1.0 - np.tanh(self.place_scale * dist_3d))
        
        # 关键：根据位置选择基础值（参考robosuite）
        if object_near_target:
            # 在目标附近：从hover_mult开始
            r_place = self.hover_mult + place_bonus
        else:
            # 不在目标附近：从当前r_hover开始（平滑过渡）
            r_place = r_hover + place_bonus
        
        # 更新 placed 状态（使用更严格的阈值用于成功判定）
        xy_dist = np.linalg.norm(obj_pos[:2] - target_pos[:2])
        z_dist = abs(obj_pos[2] - target_pos[2])
        if xy_dist < self.place_xy_threshold and z_dist < self.place_z_threshold:
            self.placed = True
        
        return r_place
    
    def _update_grasp_state(self, action):
        """更新抓取状态（V3: 添加主动抓取意图检查）"""
        # 1. 检查接触
        contact = self._check_grasp_robosuite()
        
        # 2. 检查距离
        eef_pos = self._get_eef_pos_safe()
        obj_pos = self._get_obj_pos_safe()
        distance = np.linalg.norm(eef_pos - obj_pos)
        
        # 3. 检查夹爪是否闭合
        obs_dict = self._env._get_observations()
        gripper_qpos = obs_dict.get('robot0_gripper_qpos', np.zeros(6))
        gripper_closed = gripper_qpos[2] < 0.7 and gripper_qpos[2] > 0.2
        
        # 4. 检查夹爪闭合动作（主动抓取意图）
        grasp_action = action[6] > 0 
        
        # V3 改进：添加主动抓取意图判定
        # - 距离阈值放宽到 5cm
        # - 夹爪闭合阈值放宽
        # - 只需要 1 步稳定（grasp_stable_steps=1）
        # - 必须有主动闭合动作（action[6] > 0）
        MAX_GRASP_DISTANCE = 0.05  # 5cm（放宽）
        valid_grasp = contact and (distance < MAX_GRASP_DISTANCE) and gripper_closed and grasp_action
        
        if valid_grasp:
            self.grasp_counter += 1
        else:
            self.grasp_counter = 0
        
        self.stable_grasped = (self.grasp_counter >= self.grasp_stable_steps)
    
    def _check_success(self):
        """
        检查成功【V4.1 改进版本】
        
        成功条件（必须全部满足）：
        1. 物体 XY 对齐：xy_dist < place_xy_threshold (10mm)
        2. 物体 Z 对齐：z_dist < place_z_threshold (20mm)
        
        V4.1 改进：移除"夹爪已松开"的要求
        - 只要物体放置到位就算成功
        - 不需要松开夹爪（复位动作单独训练）
        """
        obj_pos = self._get_obj_pos_safe()
        target_pos = self._get_target_pos_safe()
        
        # 检查 XY 对齐
        xy_dist = np.linalg.norm(obj_pos[:2] - target_pos[:2])
        xy_aligned = xy_dist < self.place_xy_threshold  # 10mm
        
        # 检查 Z 对齐
        z_dist = abs(obj_pos[2] - target_pos[2])
        z_aligned = z_dist < self.place_z_threshold  # 20mm
        
        return xy_aligned and z_aligned

    def _check_grasp_robosuite(self):
        """使用 robosuite 底层的 _check_grasp 方法"""
        try:
            return self._env._check_grasp(
                gripper=self._env.robots[0].gripper,
                object_geoms=self._env.bmwobject
            )
        except:
            return False

    def render(self, mode="human"):
        """渲染"""
        if mode == "human":
            self._env.render()
        return None

    def close(self):
        """关闭环境"""
        if hasattr(self, '_env'):
            self._env.close()

    def seed(self, seed=None):
        """设置随机种子"""
        if seed is not None:
            self._rng = np.random.RandomState(seed)
            try:
                if hasattr(self._env, 'seed'):
                    self._env.seed(seed)
                else:
                    self._env.reset()
            except:
                pass
        
        return super().seed(seed)
    
    def get_curriculum_info(self):
        """获取课程学习信息"""
        if hasattr(self._env, 'get_curriculum_info'):
            return self._env.get_curriculum_info()
        return None


def env_creator(**kwargs):
    """环境创建函数 - GOPS注册用"""
    return PythRobosuiteBmwAssemblyStagedCurriculumPickPlaceV4_3(**kwargs)