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
                safety_dq_scale=1.0):
        
        # initial joint pos — UR3 ready pose (garra acima da mesa de trabalho)
        # Ângulos em radianos: configuração de elbow-up padrão para pushing
        initial_qpos_dict = {
                "shoulder_pan_joint":  -1.5708,   # -90°  (aponta o braço para frente)
                "shoulder_lift_joint": -1.5708,   # -90°  (elbow up)
                "elbow_joint":          1.5708,   #  90°  (antebraço paralelo à mesa)
                "wrist_1_joint":       -1.5708,   # -90°
                "wrist_2_joint":       -1.5708,   # -90°
                "wrist_3_joint":        0.0       #   0°  (garra orientada para baixo)
            }
        
        # params
        self.encode_ee_pos = False # model reload
        self.use_fingertip_sensor = False # model reload
        self.action_scaling_factor = action_scaling_factor
        self.sparse_reward = sparse_reward

        
        MuJoCoUR3PushBaseEnv.__init__(self,
                                initial_qpos_dict=initial_qpos_dict,
                                load_fingertip_model=self.use_fingertip_sensor,
                                render_mode=render_mode,
                                object_params={"range_x_pos":np.array([-0.15,0.15]),
                                                "range_y_pos":np.array([-0.15,0.15])},
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
            "observation" : spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64")
        })

    def _reset_callback(self, options={}):
        pass

    def _step_callback(self, action):
        self.controller.update_desired_pose(self.model, self.data, action*self.action_scaling_factor)
        self.controller.update(self.model, self.data, action*self.action_scaling_factor)
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
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
            reward = -self._calc_dist_pos(achieved_goal,desired_goal)
            if self.consider_object_orientation:
                reward -= self._calc_dist_zangle(achieved_goal, desired_goal)

        assert len(reward.shape) == 1
        assert reward.shape[0] == batch_size
        return reward

    def _get_obs(self):
        # obj pos
        obj_pos = self._get_object_pose(site_name="object_site", add_noise=True)
        # ee pos
        ee_pos = self.data.site_xpos[self.ee_site_id][0:2] + self.rng_noise.normal(loc=0, scale=0.0001, size=2)

        observation = ee_pos.copy()
        achieved_goal = obj_pos.copy()
        desired_goal = self._get_target_pose()

        return OrderedDict([
                ("observation", observation),
                ("achieved_goal", achieved_goal),
                ("desired_goal", desired_goal),
               ])
