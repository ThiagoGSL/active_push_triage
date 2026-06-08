import os
import numpy as np
import logging
from gymnasium import spaces
from collections import OrderedDict
import mujoco
from ur3_push_mujoco.gym_ur3_push.envs.mujoco.ur3_push_base_env import MuJoCoUR3PushBaseEnv
import ur3_push_mujoco

class MuJoCoUR3PushSimpleEnv(MuJoCoUR3PushBaseEnv):

    def __init__(self, 
                render_mode="human",
                object_reset_options=None,
                fixed_object_height=None,
                sample_mass_slidfric_from_uniform_dist=False,
                scale_exponential=1/7,
                threshold_pos=0.05,
                log_level=logging.WARNING,
                sparse_reward=True,
                action_scaling_factor=0.6,
                n_substeps=40,
                use_sim_config=True,
                safety_dq_scale=1.0,
                ee_to_obj_reward_scale=0.2,
                torque_penalty_scale=0.001,
                manipulability_reward_scale=0.0,
                manipulability_metric="yoshikawa"):
        
        # initial joint pos — UR3 ready pose (garra acima da mesa de trabalho)
        # Ângulos em radianos: configuração de elbow-up padrão para pushing
        initial_qpos_dict = {
                "shoulder_pan_joint":   0.0,      #   0°  (aponta o braço para frente, em direção a mesa)
                "shoulder_lift_joint": -1.5708,   # -90°  (elbow up)
                "elbow_joint":          1.5708,   #  90°  (antebraço paralelo à mesa)
                "wrist_1_joint":       -1.5708,   # -90°
                "wrist_2_joint":       -1.5708,   # -90°
                "wrist_3_joint":        0.0,      #   0°  (garra orientada para baixo)
                "finger_joint":               0.700,    # Master joint (fechada sem auto-penetração: qmax=0.785 causava self-contact de 9mm)
                "right_outer_knuckle_joint": -0.700,
                "left_inner_knuckle_joint":  -0.700,
                "right_inner_knuckle_joint": -0.700,
                "left_inner_finger_joint":    0.700,
                "right_inner_finger_joint":   0.700
            }
        
        # params
        self.encode_ee_pos = False # model reload
        self.use_fingertip_sensor = False # model reload
        self.action_scaling_factor = action_scaling_factor
        self.sparse_reward = sparse_reward
        # Reward shaping: scale for EE→object distance penalty (0 = disabled).
        # Guides the approach phase so the agent learns to touch the object before pushing.
        self.ee_to_obj_reward_scale = ee_to_obj_reward_scale
        # Torque penalty: penalises excessive joint effort (0 = disabled).
        # Only applied during env steps (batch_size == 1), not HER replay.
        self.torque_penalty_scale = torque_penalty_scale
        # Manipulability bonus: rewards configurations far from kinematic singularities.
        # metric: 'yoshikawa' = sqrt(det(J Jᵀ)), 'min_sv' = smallest singular value of J.
        self.manipulability_reward_scale = manipulability_reward_scale
        assert manipulability_metric in ("yoshikawa", "min_sv"), \
            f"manipulability_metric must be 'yoshikawa' or 'min_sv', got '{manipulability_metric}'"
        self.manipulability_metric = manipulability_metric

        
        MuJoCoUR3PushBaseEnv.__init__(self,
                                initial_qpos_dict=initial_qpos_dict,
                                load_fingertip_model=self.use_fingertip_sensor,
                                render_mode=render_mode,
                                # Spawn range is a RELATIVE OFFSET from the initial EE position.
                                # initial_ee_xypos ≈ [0.35, 0.0] in world frame.
                                # Controller workspace: x ∈ [0.10, 0.65], y ∈ [-0.25, 0.25]
                                # So max safe x_offset = 0.65 - 0.35 = 0.30m
                                # Both object AND target use the same safe ranges to guarantee
                                # they always spawn within the robot's reachable workspace.
                                object_params={"range_x_pos":        np.array([0.07, 0.27]),
                                               "range_y_pos":        np.array([-0.12, 0.12]),
                                               "range_target_x_pos": np.array([0.07, 0.27]),
                                               "range_target_y_pos": np.array([-0.12, 0.12])},
                                object_reset_options=object_reset_options,
                                fixed_object_height=fixed_object_height,
                                sample_mass_slidfric_from_uniform_dist=sample_mass_slidfric_from_uniform_dist,
                                scale_exponential=scale_exponential,
                                threshold_pos=threshold_pos,
                                threshold_zangle=5, # not used
                                consider_object_orientation=False,
                                log_level=log_level,
                                n_substeps=n_substeps,
                                camera_options=dict(),
                                use_sim_config=use_sim_config,
                                safety_dq_scale=safety_dq_scale)
        
        # action and observation space 
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype="float32")
        self.observation_space = spaces.Dict({
            "desired_goal" : spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
            "achieved_goal" : spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
            # observation: [ee_pos(2), obj_pos(2), ee→obj(2), obj→target(2)] = 8 values
            # Relative distances allow PPO to learn the two-phase strategy:
            #   phase 1: minimise ee→obj (approach), phase 2: minimise obj→target (push)
            "observation" : spaces.Box(-np.inf, np.inf, shape=(8,), dtype="float64")
        })

    def _reset_callback(self, options={}):
        pass

    def _step_callback(self, action):
        # Compute the desired EE position from the current position + scaled action (called once per gym step)
        self.controller.update_desired_pose(self.model, self.data, action * self.action_scaling_factor)
        # Run the IK controller at EVERY simulation timestep (substep loop).
        # Without this, the controller computes joint velocities once and applies them for
        # n_substeps * dt seconds without feedback, causing overshoot and oscillation (tremor).
        for _ in range(self.n_substeps):
            self.controller.update(self.model, self.data, action * self.action_scaling_factor)
            mujoco.mj_step(self.model, self.data, nstep=1)
        self.render()
        
    def compute_reward(self, achieved_goal, desired_goal, info):
        # Calculation of the reward must be vectorized (HerReplayBuffer, stable-baselines3)
        batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1
        if batch_size == 1:
            achieved_goal = achieved_goal.reshape(batch_size,-1)
            desired_goal = desired_goal.reshape(batch_size,-1)

        if self.sparse_reward:
            reward = -np.bitwise_not(self._is_success(achieved_goal, desired_goal)).astype(np.float32)
        else:
            # Primary reward: -dist(object, target) — push phase
            reward = -self._calc_dist_pos(achieved_goal, desired_goal)
            if self.consider_object_orientation:
                reward -= self._calc_dist_zangle(achieved_goal, desired_goal)

            # Reward shaping: -ee_to_obj_reward_scale * dist(EE, object) — approach phase
            # Only applied during environment steps (batch_size==1).
            # Skipped for batch/replay calls to preserve compatibility with HER.
            if self.ee_to_obj_reward_scale > 0 and batch_size == 1:
                ee_pos = self.data.site_xpos[self.ee_site_id][:2]
                dist_ee_to_obj = np.linalg.norm(ee_pos - achieved_goal[0, :2])
                reward = reward - self.ee_to_obj_reward_scale * dist_ee_to_obj

            # Torque penalty: penalises joint effort — encourages smooth, efficient motions.
            # Only applied during env steps (batch_size==1); skipped for HER replay.
            if self.torque_penalty_scale > 0 and batch_size == 1:
                # qfrc_actuator: force produced by each actuator in joint space (good torque proxy)
                ur3_torques = self.data.qfrc_actuator[:6]  # first 6 DOFs belong to UR3 arm
                torque_norm = np.linalg.norm(ur3_torques) / 6.0
                reward = reward - self.torque_penalty_scale * torque_norm

            # Manipulability bonus: rewards configurations far from kinematic singularities.
            # Only applied during env steps (batch_size==1); skipped for HER replay.
            if self.manipulability_reward_scale > 0 and batch_size == 1:
                # Build the 3×nv translational Jacobian for the EE site
                J_full = np.zeros((6, self.model.nv))
                mujoco.mj_jacSite(
                    self.model, self.data,
                    J_full[:3], J_full[3:],  # translational rows, rotational rows
                    self.ee_site_id
                )
                J = J_full[:3, :6]  # 3×6 — translational part, UR3 joints only

                if self.manipulability_metric == "yoshikawa":
                    # Yoshikawa index: w = sqrt(det(J Jᵀ))
                    # Measures the volume of the velocity ellipsoid; 0 at singularities.
                    JJT = J @ J.T  # 3×3
                    w = np.sqrt(max(0.0, np.linalg.det(JJT)))
                else:  # "min_sv"
                    # Smallest singular value of J.
                    # Direct measure of proximity to singularity; 0 at singularities.
                    sv = np.linalg.svd(J, compute_uv=False)
                    w = float(sv[-1])

                reward = reward + self.manipulability_reward_scale * w

        assert len(reward.shape) == 1
        assert reward.shape[0] == batch_size
        return reward

    def _get_obs(self):
        # obj pos
        obj_pos = self._get_object_pose(site_name="object_site", add_noise=True)
        # ee pos
        ee_pos = self.data.site_xpos[self.ee_site_id][0:2] + self.rng_noise.normal(loc=0, scale=0.0001, size=2)
        # target pos
        desired_goal = self._get_target_pose()

        # Relative distances give PPO a complete picture of the task:
        #   ee_to_obj: guide the EE towards the object (approach phase)
        #   obj_to_target: guide the object towards the goal (push phase)
        ee_to_obj = obj_pos - ee_pos          # [2]
        obj_to_target = desired_goal - obj_pos  # [2]

        observation = np.concatenate([ee_pos, obj_pos, ee_to_obj, obj_to_target])  # [8]
        achieved_goal = obj_pos.copy()

        return OrderedDict([
                ("observation", observation),
                ("achieved_goal", achieved_goal),
                ("desired_goal", desired_goal),
               ])
