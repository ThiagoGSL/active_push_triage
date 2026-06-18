import numpy as np
import logging
from gymnasium_robotics.utils import rotations
from ur3_push_mujoco.utils import rotations_utils
from ur3_push_mujoco.gym_ur3_push.envs.base_goal_env import BaseGoalEnv

class BasePushEnv(BaseGoalEnv):
    def __init__(self, object_params=dict(),
                       object_reset_options=None,
                       fixed_object_height=None,
                       sample_mass_slidfric_from_uniform_dist=False,
                       scale_exponential = 1/7,
                       threshold_pos=0.05,
                       threshold_zangle=5,
                       consider_object_orientation=False,
                       log_level=logging.WARNING):
        # object_params - keys:
        #    - "types": list of possible object types
        #    - "range_radius": np.array([min radius, max radius]) in [m], only used if object is a cylinder
        #    - "range_height": np.array([min height, max height]) in [m], used for all object types
        #    - "range_mass": np.array([min mass, max mass]) in [kg]
        #    - "range_wl": np.array([min wl, max wl]) in [m]; range width and length; only used if object is a box
        #    - "range_sliding_fric": np.array([min sliding friction, max sliding friction])
        #    - "range_x_pos": np.array([min x pos, max x pos]) in [m]
        #    - "range_y_pos": np.array([min y pos, max y pos]) in [m]
        # object_reset_options - keys: 
        #    - "obj_type": None, "" (parameter is not changed), "box" or "cylinder"
        #    - "obj_mass": None, -1 (parameter is not changed) or float > 0
        #    - "obj_sliding_friction": None, -1 (parameter is not changed) or float > 0
        #    - "obj_size_0": None, -1 (parameter is not changed) or float > 0 (box: obj_width/2, cylinder: obj_radius)
        #    - "obj_size_1": None, -1 (parameter is not changed), -2 (box: width = length) or float > 0 (box: obj_length/2, cylinder: obj_height/2)
        #    - "obj_size_2": None, -1 (parameter is not changed) or float > 0 (box: obj_height/2, cylinder: ignored)
        #    - "obj_xy_pos": None or numpy array, order: (x,y), the z-pos is determined by the height of the object
        #    - "obj_quat": None or numpy array, order: (x,y,z,w)
        #    - "target_xy_pos": None or numpy array, order: (x,y), the z-pos is determined by the height of the object
        #    - "target_quat": None or numpy array, order: (x,y,z,w)
        #    Each parameter is sampled if either the corresponding key is not in the dictionary or the value is None
        # fixed_object_height: if None: object height is not fixed; else: object_height/2 [m]; 
        #                      this parameter is used (if not None) even if another value is specified in object_reset_options
        # sample_mass_slidfric_from_uniform_dist: sample mass and sliding friction coefficient from continuous uniform distribution?
        #                                     if False and both (mass and friction) are sampled: sample from modified exponential distribution
        # scale_exponential: sclae of the exponential distribution used to sample mass and sliding fricition coefficient if sample_mass_slidfric_from_uniform_dist = 0
        # threshold_pos: position threshold [m]
        # threshold_zangle: rotation threshold [deg]
        # consider_object_orientation: Consider object orientation in learning task?
        
        BaseGoalEnv.__init__(self)

        # set up logger
        logging.basicConfig(level = log_level, format = '%(asctime)s:%(levelname)s: %(message)s')

        self.consider_object_orientation = consider_object_orientation
        
        # thresholds 
        self.threshold_pos = threshold_pos
        self.threshold_zangle = threshold_zangle * (np.pi/180) # [rad]
        self.threshold_obj_start_pose = 0.1
        
        self.object_reset_options = object_reset_options
        self.initial_ee_xypos = None

        # object parameters
        self.obj_types = object_params.get("types", ["box", "cylinder"])
        self.range_obj_radius = object_params.get("range_radius", np.array([0.08/2, 0.11/2])) # [m]
        self.range_obj_height = object_params.get("range_height", np.array([0.026, 0.08/2])) # [m]
        self.range_obj_wl = object_params.get("range_wl", np.array([0.05/2, 0.11/2])) # [m]
        self.range_obj_mass = object_params.get("range_mass", np.array([0.001, 1.0])) # [kg]
        self.range_obj_sliding_fric = object_params.get("range_sliding_fric", np.array([0.2, 1.0])) 
        self.range_obj_torsional_fric = object_params.get("range_torsional_fric", np.array([0.001, 0.01])) 
        self.range_obj_x_pos = object_params.get("range_x_pos", np.array([-0.1,0.1])) #[m]
        self.range_obj_y_pos = object_params.get("range_y_pos", np.array([-0.1,0.1])) #[m]
        # Target uses its own ranges (fallback: same as object) to guarantee it spawns
        # within the robot's reachable workspace.
        self.range_target_x_pos = object_params.get("range_target_x_pos", self.range_obj_x_pos.copy()) #[m]
        self.range_target_y_pos = object_params.get("range_target_y_pos", self.range_obj_y_pos.copy()) #[m]
        self.fixed_obj_height = fixed_object_height

        # sample mass and sliding friction coefficient from continuous uniform distribution?
        # if False: sample from exponential distribution 
        self.sample_mass_slidfric_from_uniform_dist = sample_mass_slidfric_from_uniform_dist
        self.scale_exponential = scale_exponential

        self.elapsed_steps = 0

    def reset(self, seed=None, options={}):
        # options: see object_reset_options dictionary in BasePushEnv
        super().reset(seed=seed)

        if len(options)==0 and self.object_reset_options is not None:
            options = self.object_reset_options.copy()

        self.elapsed_steps = 0

    def step(self, action):
        if action.dtype != np.float32:
            action = action.astype(np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError("action dim != action_space dim")
        if not self.action_space.contains(action):
            logging.warning(f"Action {action} not in action space. Will clip invalid values to interval edges.")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self._step_callback(action)
        
        # get new observation 
        observation = self._get_obs()
        info = self._get_info(observation["achieved_goal"], observation["desired_goal"])

        if "elapsed_steps" in info.keys():
            achieved_goal = observation["achieved_goal"][info["elapsed_steps"],:]
        else:
            achieved_goal = observation["achieved_goal"]
        reward = self.compute_reward(achieved_goal, observation["desired_goal"], info)
        if reward.shape[0] > 1:
            logging.warning(f"Unexpected shape of reward returned by 'env.compute_reward()'. Current shape is: {reward.shape}, expected shape: (1,)")
        terminated = self.compute_terminated(achieved_goal, observation["desired_goal"], info)
        truncated = self.compute_truncated(achieved_goal, observation["desired_goal"], info)

        self.elapsed_steps += 1

        return observation, reward[0], terminated, truncated, info    

    def _step_callback(self, action):
        pass

    def _reset_object_target_config(self, options={}):
        if len(options) == 0 and self.object_reset_options is not None:
            options = self.object_reset_options.copy()

        # sample object type, mass, sliding friction, height, radius, length, width
        self._sample_obj_params(options=options)        

        # sample object start pose
        self.obj_start_pos = self.initial_ee_xypos.copy()
        while np.linalg.norm(self.obj_start_pos[:2] - self.initial_ee_xypos) < self.threshold_obj_start_pose:
            self.obj_start_pos, self.obj_start_quat = self._sample_object_pose(self.obj_height)
        if "obj_xy_pos" in options and options["obj_xy_pos"] is not None:
            self.obj_start_pos[0:2] = options["obj_xy_pos"]
        if "obj_quat" in options and options["obj_quat"] is not None:
            self.obj_start_quat = options["obj_quat"]

        # sample new target pose (uses range_target_*_pos to stay within robot workspace)
        self.target_start_pos, self.target_start_quat = self._sample_target_pose(self.obj_height)
        # Ensure target is at least 10cm away from the object
        while np.linalg.norm(self.target_start_pos[0:2] - self.obj_start_pos[0:2]) < 0.10:
            self.target_start_pos, self.target_start_quat = self._sample_target_pose(self.obj_height)

        if "target_xy_pos" in options and options["target_xy_pos"] is not None:
            self.target_start_pos[0:2] = options["target_xy_pos"]
        if "target_quat" in options and options["target_quat"] is not None:
            self.target_start_quat = options["target_quat"]
        target_xypos = self.target_start_pos[0:2]
        target_zangle = rotations.quat2euler(rotations_utils.tftansformationsQuat_to_mujocoQuat(self.target_start_quat))[2]
        self.target_pose = np.append(target_xypos, target_zangle)

    def _sample_obj_params(self, options):
        # set new object/target type
        if "obj_type" not in options or (options["obj_type"] != "box" and options["obj_type"] != "cylinder" and options["obj_type"] != ""):
            # sample new object/target type
            idx_type = self.np_random.integers(low=0, high=len(self.obj_types))
            self.obj_type = self.obj_types[idx_type]
        elif options["obj_type"] != "":
            self.obj_type = options["obj_type"]

        # sliding friction coefficient and object mass
        if not self.sample_mass_slidfric_from_uniform_dist and ("obj_mass" not in options or options["obj_mass"] is None) and ("obj_sliding_friction" not in options or options["obj_sliding_friction"] is None):
            # sample from modified exponential distribution
            max_force = self.range_obj_sliding_fric[1]*self.range_obj_mass[1]
            min_force = self.range_obj_sliding_fric[0]*self.range_obj_mass[0]
            self.obj_sliding_fric = (self.range_obj_sliding_fric[1] - self.range_obj_sliding_fric[0])/2
            assert self.obj_sliding_fric > 0 
            new_range_mass = np.array([min_force/self.obj_sliding_fric, max_force/self.obj_sliding_fric])

            rv_e = np.clip(self.np_random.exponential(scale=self.scale_exponential,size=1), a_min=0, a_max=1)
            rv_b = self.np_random.binomial(1, 0.5, size=1)
            rv = (1 - rv_b) * (1-rv_e) + rv_b * rv_e
            self.obj_mass = (new_range_mass[1] - new_range_mass[0])* rv[0] + new_range_mass[0]
        else:
            # set new sliding friction parameter
            if "obj_sliding_friction" not in options or options["obj_sliding_friction"] is None:
                # sample new sliding friction parameter from uniform distribution
                self.obj_sliding_fric = (self.range_obj_sliding_fric[1] - self.range_obj_sliding_fric[0]) * self.np_random.random() + self.range_obj_sliding_fric[0]
            elif options["obj_sliding_friction"] != -1:
                self.obj_sliding_fric = options["obj_sliding_friction"]

            # set new object/target mass
            if "obj_mass" not in options or options["obj_mass"] is None:
                # sample new object/target mass from uniform distribution
                self.obj_mass = (self.range_obj_mass[1] - self.range_obj_mass[0])* self.np_random.random() + self.range_obj_mass[0]
            elif options["obj_mass"] != -1:
                self.obj_mass = options["obj_mass"]

        # set new sliding friction parameter
        if "obj_torsional_friction" not in options or options["obj_torsional_friction"] is None:
            # sample new torsional friction parameter from uniform distribution
            self.obj_torsional_fric = (self.range_obj_torsional_fric[1] - self.range_obj_torsional_fric[0]) * self.np_random.random() + self.range_obj_torsional_fric[0]
        elif options["obj_torsional_friction"] != -1:
            self.obj_torsional_fric = options["obj_torsional_friction"]

        # sample new object/target size depending on object type
        if self.obj_type == "box":
            self.obj_geom_type_value = 6
            # set object width/2
            if "obj_size_0" not in options or options["obj_size_0"] is None:
                self.obj_size_0 = (self.range_obj_wl[1] - self.range_obj_wl[0]) * self.np_random.random() + self.range_obj_wl[0] # width/2
            elif options["obj_size_0"] != -1:
                self.obj_size_0 = options["obj_size_0"]
            # set object length/2
            if "obj_size_1" not in options or options["obj_size_1"] is None:
                self.obj_size_1 = (self.range_obj_wl[1] - self.range_obj_wl[0]) * self.np_random.random() + self.range_obj_wl[0] # length/2
            elif options["obj_size_1"] == -3:
                self.obj_size_1 = (self.range_obj_wl[1] - self.range_obj_wl[0]) * self.np_random.random() + self.range_obj_wl[0] # length/2
                while abs(self.obj_size_0 - self.obj_size_1) < 1e-2:
                    self.obj_size_1 = (self.range_obj_wl[1] - self.range_obj_wl[0]) * self.np_random.random() + self.range_obj_wl[0] # length/2
            elif options["obj_size_1"] == -2:
                self.obj_size_1 = self.obj_size_0
            elif options["obj_size_1"] != -1:
                self.obj_size_1 = options["obj_size_1"]
            # set object height/2 
            if self.fixed_obj_height is not None:
                self.obj_size_2 = self.fixed_obj_height
            elif "obj_size_2" not in options or options["obj_size_2"] is None:  
                max_height = np.amin(np.array([self.range_obj_height[1], self.obj_size_0, self.obj_size_1]))
                # assert self.range_obj_height[0] <= max_height  
                self.obj_size_2 = (max_height - self.range_obj_height[0]) * self.np_random.random() + self.range_obj_height[0] # height/2
            elif options["obj_size_2"] != -1:
                self.obj_size_2 = options["obj_size_2"]

            self.obj_height = self.obj_size_2
            # assert self.obj_height <= self.obj_size_0 or self.obj_height <= self.obj_size_1
        else:
            # self.obj_type == "cylinder"
            self.obj_geom_type_value = 5
            # set object radius
            if "obj_size_0" not in options or options["obj_size_0"] is None:
                self.obj_size_0 = (self.range_obj_radius[1] - self.range_obj_radius[0]) * self.np_random.random() + self.range_obj_radius[0] # radius 
            elif options["obj_size_0"] != -1:
                self.obj_size_0 = options["obj_size_0"]
            # set object height/2
            if self.fixed_obj_height is not None:
                self.obj_size_1 = self.fixed_obj_height
            elif "obj_size_1" not in options or options["obj_size_1"] is None:
                max_height = np.min(np.array([self.range_obj_height[1], self.obj_size_0]))
                # assert self.range_obj_height[0] <= max_height
                self.obj_size_1 = (max_height - self.range_obj_height[0]) * self.np_random.random() + self.range_obj_height[0] # height/2
            elif options["obj_size_1"] != -1:
                self.obj_size_1 = options["obj_size_1"]

            self.obj_height = self.obj_size_1
            assert self.obj_height <= self.obj_size_0

        # assert self.range_obj_height[0] <= self.obj_height and self.obj_height <= self.range_obj_height[1]

    def _sample_object_pose(self, obj_height):
        # sample rotation about z-axis
        z_angle = (np.pi - (-np.pi)) * self.np_random.random() - np.pi
        # sample x-pos
        x_pos = self.np_random.uniform(np.amin(self.range_obj_x_pos), np.amax(self.range_obj_x_pos))
        # sample y-pos 
        y_pos = self.np_random.uniform(np.amin(self.range_obj_y_pos), np.amax(self.range_obj_y_pos))

        pos = np.array([x_pos, y_pos, obj_height + self.height_table + 0.001])
        quat = rotations_utils.mujocoQuat_to_tftransformationsQuat(rotations.euler2quat(np.array([0,0,z_angle]))) # tf.transformations order

        return pos, quat

    def _sample_target_pose(self, obj_height):
        """Sample a random target pose using range_target_*_pos (may differ from object ranges).

        Keeping separate ranges for object and target ensures both spawn inside the
        controller's reachable workspace even when the caller uses asymmetric ranges.
        """
        # sample rotation about z-axis
        z_angle = (np.pi - (-np.pi)) * self.np_random.random() - np.pi
        # sample x-pos using target-specific range
        x_pos = self.np_random.uniform(
            np.amin(self.range_target_x_pos), np.amax(self.range_target_x_pos))
        # sample y-pos using target-specific range
        y_pos = self.np_random.uniform(
            np.amin(self.range_target_y_pos), np.amax(self.range_target_y_pos))

        pos = np.array([x_pos, y_pos, obj_height + self.height_table + 0.001])
        quat = rotations_utils.mujocoQuat_to_tftransformationsQuat(rotations.euler2quat(np.array([0, 0, z_angle])))

        return pos, quat

    def _get_target_pose(self):
        return self.target_pose.copy() if self.consider_object_orientation else self.target_pose[0:2].copy()

    def _is_success(self, achieved_goal, desired_goal, consider_symmetry=True):
        is_success_pos = self._calc_dist_pos(achieved_goal,desired_goal) < self.threshold_pos
        if self.consider_object_orientation:
            is_success_zangle = self._calc_dist_zangle(achieved_goal, desired_goal, consider_symmetry) < self.threshold_zangle
            return np.bitwise_and(is_success_pos,is_success_zangle)
        else:
            return is_success_pos
        
    def _calc_dist_pos(self, achieved_goal, desired_goal):
        # calculate Euclidean distance
        if len(achieved_goal.shape) == 1:
            desired_goal = desired_goal.reshape(1,-1)
            achieved_goal = achieved_goal.reshape(1,-1)

        return np.linalg.norm(desired_goal[:,0:2]-achieved_goal[:,0:2], ord=2, axis=1)
    
    def _calc_dist_zangle(self, achieved_goal, desired_goal, consider_symmetry=True):
        # rotation about x or y axis not possible -> consider rotation about z axis
        if len(achieved_goal.shape) > 1:
            target_z_angle = desired_goal[:,2]
            object_z_angle = achieved_goal[:,2]
        else:
            target_z_angle = np.array([desired_goal[2]])
            object_z_angle = np.array([achieved_goal[2]])
        
        assert (object_z_angle <= np.pi).all() and (object_z_angle >= -np.pi).all()
        assert (target_z_angle <= np.pi).all() and (target_z_angle >= -np.pi).all()

        if consider_symmetry and not self.obj_type == "box":
            # self.obj_type == "cylinder" -> no rotation required
            return np.zeros(object_z_angle.shape)

        euler_obj = np.zeros((object_z_angle.shape[0], 3))
        euler_obj[:,2] = object_z_angle

        euler_tar = np.zeros((target_z_angle.shape[0], 3))
        euler_tar[:,2] = target_z_angle

        dist = np.abs(rotations.subtract_euler(euler_obj, euler_tar)[:,2])
        assert (dist <= np.pi).all()

        if consider_symmetry:
            # self.obj_type == "box"
            if abs(self.obj_size_0 - self.obj_size_1) < 1e-2:
                # similar width and length
                max_dist = np.pi/2
            else:
                max_dist = np.pi
            dist_tmp = np.mod(dist, max_dist)
            dist = np.minimum(dist_tmp, max_dist - dist_tmp)

        return dist

    def _get_info(self, achieved_goal, desired_goal):
        info = {"dist_pos": self._calc_dist_pos(achieved_goal,desired_goal),
                "is_success": self._is_success(achieved_goal, desired_goal)[0]} # info["is_success"] = True/False required by stable-baselines3 logger to log mean success rate during training
        
        if self.consider_object_orientation:
            info.update({"dist_zangle": self._calc_dist_zangle(achieved_goal,desired_goal)})
        
        return info
    
    def _preprocess_info_dict(self, info={}):
        # use info dict to calculate reward?
        achieved_goal, desired_goal, batch_size = None, None, None
        if isinstance(info,np.ndarray) and "achieved_goal" in info[0].keys():
            batch_size = info.shape[0]
            if "achieved_goal" in info[0].keys():
                achieved_goal = np.zeros((batch_size,info[0]["achieved_goal"].shape[0]))
                desired_goal = np.zeros((batch_size,info[0]["desired_goal"].shape[0]))

                for i in range(0,batch_size):
                    achieved_goal[i,:] = info[i]["achieved_goal"]
                    desired_goal[i,:] = info[i]["desired_goal"]
        elif isinstance(info,dict) and "achieved_goal" in info.keys():
            batch_size = 1
            achieved_goal = info["achieved_goal"].reshape((1,-1))
            desired_goal = info["desired_goal"].reshape((1,-1))
        
        return achieved_goal, desired_goal, batch_size