import os
import numpy as np
import logging
from copy import deepcopy
from gymnasium import spaces
from collections import OrderedDict
import mujoco, torch
from gymnasium_robotics.utils import rotations
from ur3_push_mujoco.gym_ur3_push.envs.mujoco.ur3_push_base_env import MuJoCoUR3PushBaseEnv
from ur3_push_mujoco.autoencoder.VAE import VAE

DEFAULT_CAMERA_CONFIG_REAL = {
    "distance": 1.0,
    "azimuth": -100,
    "elevation": -30,
    "lookat": np.array([-0.1, 0.7, 0.6]),
}

DEFAULT_CAMERA_CONFIG_SIM = {
    "distance": 1.0,
    "azimuth": 150.0,
    "elevation": -25.0,
    "lookat": np.array([0.7, -0.1, 0.6]),
}

class MuJoCoUR3PushEnv(MuJoCoUR3PushBaseEnv):

    def __init__(self, 
                render_mode="human",
                object_reset_options=None,
                object_params=dict(),
                fixed_object_height=None,
                fixed_object_height_VAE=None, # evaluation
                sample_mass_slidfric_from_uniform_dist=False,
                scale_exponential = 1/7,
                threshold_pos=0.05,
                threshold_zangle=10,
                threshold_latent_space=0.15,
                log_level=logging.WARNING,
                sparse_reward=True,
                ground_truth_dense_reward=False,
                consider_object_orientation=False,
                action_scaling_factor=0.6,
                n_substeps=120,
                latent_dim=6,
                encode_ee_pos=False,
                camera_options={"camera_name": "rgb_cam"},
                use_obs_history = False,
                num_stack_obs = None,
                use_fingertip_sensor = False,
                use_sim_config = True,
                safety_dq_scale=0.37
                ):
        
        # initial joint pos
        if use_sim_config:
            # UR3 ready pose (sim) — garra acima da mesa, elbow-up
            # Posição end-effector aprox.: [0.40, 0, 0.55]
            initial_qpos_dict = {
                    "shoulder_pan_joint":   0.0,      #   0°
                    "shoulder_lift_joint": -1.5708,   # -90°  (elbow up)
                    "elbow_joint":          1.5708,   #  90°
                    "wrist_1_joint":       -1.5708,   # -90°
                    "wrist_2_joint":       -1.5708,   # -90°
                    "wrist_3_joint":        0.0,      #   0°
                    "finger_joint":               0.785398,
                    "right_outer_knuckle_joint": -0.785398,
                    "left_inner_knuckle_joint":  -0.785398,
                    "right_inner_knuckle_joint": -0.785398,
                    "left_inner_finger_joint":    0.785398,
                    "right_inner_finger_joint":   0.785398
                }
        else:
            # UR3 ready pose (real) — alinhada com o referencial da câmara real
            # Posição end-effector aprox.: [0, 0.41, 0.59]
            initial_qpos_dict = {
                    "shoulder_pan_joint":   0.0,
                    "shoulder_lift_joint": -1.5708,
                    "elbow_joint":          1.5708,
                    "wrist_1_joint":       -1.5708,
                    "wrist_2_joint":       -1.5708,
                    "wrist_3_joint":        1.5708,   # +90° para alinhar com câmara real
                    "finger_joint":               0.785398,
                    "right_outer_knuckle_joint": -0.785398,
                    "left_inner_knuckle_joint":  -0.785398,
                    "right_inner_knuckle_joint": -0.785398,
                    "left_inner_finger_joint":    0.785398,
                    "right_inner_finger_joint":   0.785398
                }
        
        # params
        self.use_fingertip_sensor = use_fingertip_sensor
        self.action_scaling_factor = action_scaling_factor
        self.use_obs_history = use_obs_history
        self.num_stack_obs = num_stack_obs
        if self.use_obs_history and self.num_stack_obs is not None:
            raise ValueError("use_obs_history = True and num_stack_obs != None. Please choose one observation history type.")
        self.reset_obs = False
        self.max_episode_length = 50 # cannot access Gymnasium TimeLimitWrapper.max_episode_steps 
        self.encode_ee_pos = encode_ee_pos
        self.threshold_latent_space = threshold_latent_space
        self.num_taxel = 1

        if use_sim_config:
            default_camera_config = DEFAULT_CAMERA_CONFIG_SIM
            if "range_x_pos" not in object_params.keys():
                object_params.update({"range_x_pos": np.array([0.2,0.45])})
            if "range_y_pos" not in object_params.keys():
                object_params.update({"range_y_pos": np.array([-0.15,0.15])})
            
        else:
            default_camera_config = DEFAULT_CAMERA_CONFIG_REAL
            if "range_x_pos" not in object_params.keys():
                object_params.update({"range_x_pos": np.array([0.2,0.45])})
            if "range_y_pos" not in object_params.keys():
                object_params.update({"range_y_pos": np.array([-0.15,0.15])})
                
        MuJoCoUR3PushBaseEnv.__init__(self,
                                        initial_qpos_dict=initial_qpos_dict,
                                        default_camera_config=default_camera_config,
                                        load_fingertip_model=self.use_fingertip_sensor,
                                        render_mode=render_mode,
                                        object_params=object_params,
                                        object_reset_options=object_reset_options,
                                        fixed_object_height=fixed_object_height,
                                        sample_mass_slidfric_from_uniform_dist=sample_mass_slidfric_from_uniform_dist,
                                        scale_exponential=scale_exponential,
                                        threshold_pos=threshold_pos,
                                        threshold_zangle=threshold_zangle,
                                        consider_object_orientation=consider_object_orientation,
                                        log_level=log_level,
                                        n_substeps=n_substeps,
                                        camera_options=camera_options,
                                        use_sim_config=use_sim_config,
                                        safety_dq_scale=safety_dq_scale
                                        )
        # rewards
        self.sparse_reward = sparse_reward
        self.ground_truth_dense_reward = ground_truth_dense_reward
        if self.sparse_reward and self.ground_truth_dense_reward:
            raise ValueError("sparse_reward = True and ground_truth_dense_reward = True. Please choose one reward type (sparse or dense reward).")

        # tactile sensor
        if self.use_fingertip_sensor:
            tactile_sensor = self.model.sensor("tactile_sensor")
            self.tactile_sensor_id = tactile_sensor.id
            self.tactile_sensor_adr = tactile_sensor.adr[0]
            self.tactile_sensor_dim = tactile_sensor.dim[0]
            self.tactile_sensor_max_value = 10

        # load trained VAE
        self.data_path = os.getenv("UR3_PUSH_DATAPATH")
        self.latent_dim = latent_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        assert self.image_height == self.image_width
        # latent_dim=None used to sample train and test images for VAE training
        if self.latent_dim is not None:
            if self.fixed_obj_height != fixed_object_height_VAE:
                logging.warning(f"Fixed object environment != fixed object height VAE")
            with torch.inference_mode():
                self.vae = VAE(latent_dim=latent_dim)
                self.vae.load_state_dict(torch.load(os.path.join(self.data_path, "net", f"vae_latentdim_{self.latent_dim}_imgHeight_{self.image_height}_imgWidth_{self.image_width}_fixedObjectHeight_{fixed_object_height_VAE}_simConfig_{int(self.use_sim_config)}")))
                self.vae.to(self.device)
        if self.latent_dim is None and (self.encode_ee_pos or (not self.sparse_reward and not self.ground_truth_dense_reward)):
            raise ValueError("latent_dim is None, but encode_ee_pos==True or dense_latent_space reward is selected")
        
        # observation space
        # observation: - data fingertip sensor if sensor is used 
        #              - latent ee state or measured ee pos w.r.t. base frame: [x pos [m], y pos [m], angle about z-axis [rad] if fingertip sensor is used]
        # achieved_goal: if latent_dim not None latent object state, else measured object pose w.r.t. base frame: [x pos [m], y pos [m]] or [x pos [m], y pos [m], cos(angle about z), sin(angle about z)]
        # desired_goal: latent target state
        shape_sensor_output = self.num_taxel if self.use_fingertip_sensor else 0
        if self.encode_ee_pos:
            shape_ee_observation = self.latent_dim
        else:
            if self.latent_dim is None:
                # Ground-truth mode: observation enriched with obj_pos + relative distances
                # ee_pos[2] + (cos/sin[2] if fingertip) + obj_pos[2] + ee→obj[2] + obj→target[2]
                shape_ee_observation = (4 if self.use_fingertip_sensor else 2) + 6
            else:
                # VAE latent mode: standard ee pose only (unchanged)
                shape_ee_observation = 3 if self.use_fingertip_sensor else 2
        self.single_obs_length = shape_sensor_output + shape_ee_observation
        self.single_ag_length = self.latent_dim if self.latent_dim is not None else 2 + 2*int(self.consider_object_orientation)
        
        if self.use_obs_history:
            observation_shape = (self.max_episode_length, self.single_obs_length)
            achieved_goal_shape = (self.max_episode_length, self.single_ag_length)
        elif self.num_stack_obs is not None:
            observation_shape = (self.num_stack_obs * self.single_obs_length, )
            achieved_goal_shape = (self.num_stack_obs * self.single_ag_length,)
        else:
            observation_shape = (self.single_obs_length,)
            achieved_goal_shape = (self.single_ag_length,)

        self.observation_space = spaces.Dict({
            "observation" : spaces.Box(low=-np.inf, high=np.inf, shape=observation_shape, dtype=np.float32),
            "achieved_goal" : spaces.Box(low=-np.inf, high=np.inf, shape=achieved_goal_shape, dtype=np.float32), 
            "desired_goal" : spaces.Box(low=-np.inf, high=np.inf, shape=(self.single_ag_length,), dtype=np.float32)
        })

        # action space
        low = np.array([-1.0, -1.0])
        high = np.array([1.0, 1.0])
        if self.use_fingertip_sensor:
            low = np.append(low, -np.pi)
            high = np.append(high, np.pi)
        if self.n_substeps == -1:
            if self.use_sim_config:
                low = np.append(low, 10)
                high = np.append(high, 150)
            else:
                low = np.append(low, 10)
                high = np.append(high, 600)
        self.action_space = spaces.Box(low = low.astype(np.float32), high = high.astype(np.float32), dtype=np.float32)

        # current observation 
        self.current_obs = self._reset_current_obs()
        self.reset_obs = False

        # corrective movements per episode
        self.cnt_ep_corrections = 0
        self.cnt_dist_ep_corrections = 0
        self.success_last_step = False
        self.last_dist_pos = np.inf
        self.dist_increased = False

    def _reset_callback(self, options={}):
        # reset counter corrective movements per episode
        self.cnt_ep_corrections = 0
        self.cnt_dist_ep_corrections = 0
        self.success_last_step = False
        self.last_dist_pos = np.inf
        self.dist_increased = False

        # reset episode history
        if self.use_obs_history or self.num_stack_obs is not None:
            self._reset_current_obs()
            if self.num_stack_obs is not None:
                self.reset_obs = True
                self._get_obs()
                self.reset_obs = False

    def _step_callback(self, action, scale_actions=True):
        # action: 2-dim, 3-dim or 4-dim
        # action[0], action[1]: x,y-position offset (w.r.t. world frame)
        # action[2]: orientation offset about z-axis of world frame
        # action[-1]: number of sim steps

        # scale actions
        if scale_actions:
            action[:2] = action[:2] * self.action_scaling_factor

        if self.n_substeps == -1:
            n_steps = int(action[-1])
            ctrl = action[:-1]
        else:
            n_steps = self.n_substeps
            ctrl = action

        self.controller.update_desired_pose(self.model, self.data, ctrl)
        if self.use_sim_config:
            self.controller.update(self.model, self.data, ctrl)

        for _ in range(n_steps):
            if not self.use_sim_config:
                self.controller.update(self.model, self.data, ctrl)

            mujoco.mj_step(self.model, self.data)
            # update counter corrective movements
            object_pose = self._get_object_pose(site_name="object_site", add_noise=False)
            target_pose = self._get_target_pose()

            dist_pos = self._calc_dist_pos(object_pose,target_pose)
            # necessary corrections
            success_step = (dist_pos < self.threshold_pos)[0]
            if self.success_last_step and not success_step and np.abs(dist_pos - self.threshold_pos)>2e-5:
                self.cnt_ep_corrections += 1 
            self.success_last_step = success_step
            # distance corrections
            if self.last_dist_pos < dist_pos and np.abs(self.last_dist_pos - dist_pos)>2e-5:
                self.dist_increased = True
            elif self.dist_increased and dist_pos < self.last_dist_pos and np.abs(self.last_dist_pos - dist_pos)>2e-5:
                self.dist_increased = False
                self.cnt_dist_ep_corrections += 1
            self.last_dist_pos = dist_pos
            
        self.render()
        
    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.sparse_reward or self.ground_truth_dense_reward:
            # ground truth rewards
            achieved_goal, desired_goal, batch_size = self._preprocess_info_dict(info) # returns ground truth object/target pose
            if self.consider_object_orientation:
                desired_shape = 3
            else:
                desired_shape = 2
            assert achieved_goal.shape[1] == desired_shape
            assert desired_goal.shape[1] == desired_shape

            if self.sparse_reward:
                # ground truth sparse reward
                # 0 if distance between current object pose and target pose < threshold (orientation is considered if self.consider_object_orientation=True)
                # otherwise: -1
                reward = -np.bitwise_not(self._is_success(achieved_goal, desired_goal)).astype(np.float32)
            else:
                # ground truth dense reward
                reward = -self._calc_dist_pos(achieved_goal,desired_goal)
                if self.consider_object_orientation:
                    reward -= self._calc_dist_zangle(achieved_goal, desired_goal)

        else:
            # dense reward in latent space
            batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1
            if batch_size == 1:
                achieved_goal = achieved_goal.reshape(batch_size,-1)
                desired_goal = desired_goal.reshape(batch_size,-1)

            desired_shape = self.latent_dim if self.num_stack_obs is None else self.latent_dim * self.num_stack_obs
            assert achieved_goal.shape[1] == desired_shape
            assert desired_goal.shape[1] == self.latent_dim
            reward = -np.linalg.norm(achieved_goal[:,-self.single_ag_length:]-desired_goal, ord=2, axis=1)

        assert len(reward.shape) == 1
        assert reward.shape[0] == batch_size
        return reward

    def _calc_dist_latent_space(self, achieved_goal, desired_goal):
        batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1
        # Calculation must be vectorized (HerReplayBuffer, stable-baselines3)
        if batch_size == 1:
            achieved_goal = achieved_goal.reshape(batch_size,-1)
            desired_goal = desired_goal.reshape(batch_size,-1)

        desired_shape = self.latent_dim if self.num_stack_obs is None else self.latent_dim * self.num_stack_obs
        assert achieved_goal.shape[1] == desired_shape
        assert desired_goal.shape[1] == self.latent_dim

        return np.linalg.norm(achieved_goal[:,-self.single_ag_length:]-desired_goal, ord=2, axis=1)

    def _is_success(self, achieved_goal, desired_goal):
        if self.sparse_reward or self.ground_truth_dense_reward:
            return super()._is_success(achieved_goal, desired_goal)
        else:
            # dense reward in latent space
            return self._calc_dist_latent_space(achieved_goal, desired_goal) < self.threshold_latent_space

    def _get_obs(self):
        # data tactile sensor
        if self.use_fingertip_sensor:
            # get sensor data from virtual taxel
            sensor_data = self.data.sensordata[self.tactile_sensor_adr]
            # scale and add noise
            sensor_data = ((sensor_data + self.rng_noise.normal(loc=0, scale=1e-4, size=self.num_taxel))/self.tactile_sensor_max_value).astype(np.float32)
        
        if self.latent_dim is not None:
            # binary object and target image
            object_binary_img,_ = self.get_binaryGeomImg(visible_geom="object")
            binary_images_np = np.concatenate(( object_binary_img.reshape((1, 1, self.image_height, self.image_width)), 
                                                self.target_binary_image.reshape((1, 1, self.image_height, self.image_width))), 
                                                axis=0).astype(np.float32)
        else:
            object_pose = self._get_object_pose(site_name="object_site", add_noise=True)
            target_pose = self._get_target_pose()
            if self.consider_object_orientation:
                achieved_goal = np.array([object_pose[0], object_pose[1], np.cos(object_pose[2]), np.sin(object_pose[2])]).astype(np.float32)
                desired_goal = np.array([target_pose[0], target_pose[1], np.cos(target_pose[2]), np.sin(target_pose[2])]).astype(np.float32)
            else:
                achieved_goal = np.array([object_pose[0], object_pose[1]]).astype(np.float32)
                desired_goal = np.array([target_pose[0], target_pose[1]]).astype(np.float32)

        if self.encode_ee_pos and self.latent_dim is not None:
            # binary ee image
            ee_binary_img,_ = self.get_binaryGeomImg(visible_geom="ee")        
            binary_images_np = np.concatenate((binary_images_np, 
                                            ee_binary_img.reshape((1, 1, self.image_height, self.image_width))), 
                                            axis=0).astype(np.float32)
        else:
            # ee pose
            ee_pose = self.data.site_xpos[self.ee_site_id][0:2] + self.rng_noise.normal(loc=0, scale=0.0001, size=2)
            if self.use_fingertip_sensor:
                ee_rotmat = self.data.site_xmat[self.ee_site_id,:].reshape(3, 3)
                ee_x_axis = ee_rotmat[:,0]
                ee_zangle = np.arctan2(ee_x_axis[1], ee_x_axis[0]) + (np.pi/180)*self.rng_noise.standard_normal() # euler: xyz
                ee_zangle = rotations.normalize_angles(ee_zangle) # -pi <= ee_zangle <= pi
                
                if self.latent_dim is None:
                    ee_pose = np.append(np.append(ee_pose, np.cos(ee_zangle)), np.sin(ee_zangle))
                else:
                    ee_pose = np.append(ee_pose, ee_zangle)
            ee_pose = ee_pose.astype(np.float32)

        if self.latent_dim is not None:
            # latent states      
            binary_images = torch.from_numpy(binary_images_np).to(self.device)
            with torch.inference_mode():
                latent_vecs = self.vae.encoder(binary_images)

            achieved_goal = latent_vecs[0].cpu().numpy()
            desired_goal = latent_vecs[1].cpu().numpy()
            if self.encode_ee_pos:
                ee_pose = latent_vecs[2].cpu().numpy()
            
        # Enrich observation with ground-truth relative distances (non-latent, non-encoded mode)
        #   ee→obj: approach phase   |   obj→target: push phase
        if self.latent_dim is None and not self.encode_ee_pos:
            obj_pos_2d = achieved_goal[:2].astype(np.float32)
            target_pos_2d = desired_goal[:2].astype(np.float32)
            ee_to_obj = obj_pos_2d - ee_pose[:2]
            obj_to_target = target_pos_2d - obj_pos_2d
            ee_pose = np.concatenate([ee_pose, obj_pos_2d, ee_to_obj, obj_to_target])

        # concat observation
        if self.use_fingertip_sensor:
            observation = np.append(sensor_data, ee_pose)
        else:
            observation = ee_pose

        if self.use_obs_history:
            self.current_obs["observation"][self.elapsed_steps,:] = observation
            self.current_obs["achieved_goal"][self.elapsed_steps,:] = achieved_goal
            self.current_obs["desired_goal"] = desired_goal
            return deepcopy(self.current_obs)
        elif self.num_stack_obs is not None:
            if not self.reset_obs:
                self.current_obs["observation"] = np.roll(self.current_obs["observation"], -self.single_obs_length)
                self.current_obs["achieved_goal"] = np.roll(self.current_obs["achieved_goal"], -self.single_ag_length)

                self.current_obs["observation"][-self.single_obs_length:] = observation
                self.current_obs["achieved_goal"][-self.single_ag_length:] = achieved_goal
            else:
                self.current_obs["observation"] = np.tile(observation, self.num_stack_obs)
                self.current_obs["achieved_goal"] = np.tile(achieved_goal, self.num_stack_obs)
            self.current_obs["desired_goal"] = desired_goal
            return deepcopy(self.current_obs)
        else:
            return OrderedDict([
                    ("observation", observation),
                    ("achieved_goal", achieved_goal),
                    ("desired_goal", desired_goal),
                ])

    def _reset_current_obs(self):
        self.current_obs = {"observation": np.zeros(self.observation_space["observation"].shape, dtype=np.float32),
                            "achieved_goal": np.zeros(self.observation_space["achieved_goal"].shape, dtype=np.float32),
                            "desired_goal": np.zeros(self.observation_space["desired_goal"].shape, dtype=np.float32)}

    def _get_info(self, achieved_goal=None, desired_goal=None):
        if self.sparse_reward or self.ground_truth_dense_reward:
            object_pose = self._get_object_pose(site_name="object_site", add_noise=False)
            target_pose = self._get_target_pose()
            if self.consider_object_orientation:
                assert object_pose.shape == (3,)
                assert target_pose.shape == (3,)
            else:
                assert object_pose.shape == (2,)
                assert target_pose.shape == (2,)
            
            info = {"achieved_goal": object_pose, "desired_goal": target_pose}
            info.update(MuJoCoUR3PushBaseEnv._get_info(self,object_pose, target_pose))
        else:
            if not self.use_obs_history:
                info = {"is_success": self._is_success(achieved_goal, desired_goal)[0]}
            else:
                info = {"is_success": self._is_success(achieved_goal[self.elapsed_steps,:], desired_goal)[0],
                        "elapsed_steps": self.elapsed_steps}
                
        info.update({"num_corrective_movements": self.cnt_ep_corrections, "num_distance_corrections": self.cnt_dist_ep_corrections})

        return info