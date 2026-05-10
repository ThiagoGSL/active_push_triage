import logging
import os
import numpy as np
import mujoco
import ur3_push_mujoco
import cv2 as cv
from ur3_push_mujoco.gym_ur3_push.envs.base_push_env import BasePushEnv
from ur3_push_mujoco.utils.rendering_utils import CustomMujocoRenderer
from ur3_push_mujoco.utils import image_utils, mujoco_utils, rotations_utils
from gymnasium_robotics.utils import rotations


DEFAULT_CAMERA_CONFIG_SIM = {
    "distance": 1.0,
    "azimuth": 150.0,
    "elevation": -25.0,
    "lookat": np.array([0.7, -0.1, 0.6]),
}


class MuJoCoUR3PushBaseEnv(BasePushEnv):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, initial_qpos_dict,
                       load_fingertip_model=False,
                       default_camera_config=DEFAULT_CAMERA_CONFIG_SIM,
                       render_mode="human",
                       object_params=dict(),
                       object_reset_options=None,
                       fixed_object_height=None,
                       sample_mass_slidfric_from_uniform_dist=False,
                       scale_exponential=1/7,
                       threshold_pos=0.05,
                       threshold_zangle=5,
                       consider_object_orientation=False,
                       log_level=logging.WARNING,
                       n_substeps=40,
                       camera_options=dict(),
                       use_sim_config=True,
                       safety_dq_scale=1.0):
    
        # load model
        model_xml_str  = mujoco_utils.generate_model_xml_string(robot_gravity_compensation=1,
                                                                fingertip_model=load_fingertip_model,
                                                                use_sim_config=use_sim_config)
        
        # change CWD temporarily to assets folder so relative paths inside included XMLs are resolved
        original_cwd = os.getcwd()
        os.chdir(os.path.join(ur3_push_mujoco.__path__[0], "assets"))
        self.model = mujoco.MjModel.from_xml_string(model_xml_str)
        os.chdir(original_cwd)
        
        self.data = mujoco.MjData(self.model)

        self.use_sim_config = use_sim_config
        self.n_substeps = n_substeps # number of MuJoCo simulation timesteps per Gymnasium step

        # BasePushEnv
        BasePushEnv.__init__(   
            self,
            object_params=object_params,
            object_reset_options=object_reset_options,
            fixed_object_height=fixed_object_height,
            sample_mass_slidfric_from_uniform_dist=sample_mass_slidfric_from_uniform_dist,
            scale_exponential=scale_exponential,
            threshold_pos=threshold_pos,
            threshold_zangle=threshold_zangle,
            consider_object_orientation=consider_object_orientation,
            log_level=log_level)
        
        # init object params
        self._init_object_params()
        
        # rendering
        self.render_mode = render_mode
        if self.render_mode is not None:
            self.mujoco_renderer = CustomMujocoRenderer(self.model, self.data, default_camera_config)

        # camera 
        if "camera_name" in camera_options.keys():
            self.use_camera = True
            self.camera_name = camera_options["camera_name"]
            self.camera_id = self.model.camera(self.camera_name).id
            self.image_width = camera_options.get("width", 64)
            self.image_height = camera_options.get("height", 64)
            self.geomgroup = camera_options.get("geomgroup", {"object": np.array([1,1,0,0,0,0], dtype=np.uint8), 
                                                              "ee": np.array([1,1,0,0,0,0], dtype=np.uint8)})
            self.min_max_hue = camera_options.get("min_max_hue", {"object": [80,103], "ee": [0,10]})
            self.min_max_saturation = camera_options.get("min_max_saturation", {"object": [190, 255], "ee": [170,255]})
            self.min_max_value = camera_options.get("min_max_value", {"object": [0, 255] if self.use_sim_config else [65,255], "ee": [0,255]})
        else:
            self.use_camera = False

        # EE site — UR3 com garra RG2: TCP é o 'pinch_site'
        self.ee_site_name = "pinch_site"
        self.ee_site_id = self.model.site(self.ee_site_name).id

        # set initial ee pos
        self.initial_qpos_dict = initial_qpos_dict
        self.reset_robot_qpos()

        # extract information for sampling goals.
        initial_ee_pos = mujoco_utils.get_site_xpos(self.model, self.data, self.ee_site_name).copy()
        self.initial_ee_xypos = initial_ee_pos[:2]
        self.initial_ee_zpos = initial_ee_pos[2]
        self.height_table = self.model.geom("table").size[2] * 2

        # controller (sets gravity compensation: body.gravcomp = 1)
        if self.use_sim_config:
            min_ee_xy_pos = np.array([0.1, -0.25])
            max_ee_xy_pos = np.array([0.65, 0.25])
        else:
            # min, max y (red): [0.246, 0.5788]
            min_ee_xy_pos = np.array([-0.25, 0.15])
            max_ee_xy_pos = np.array([0.25, 0.65])
            
        self.controller = mujoco_utils.MuJoCoUR3PushController(model = self.model, 
                                                                robot_name = "ur3",
                                                                ee_site_name = self.ee_site_name,
                                                                initial_ee_zpos = self.initial_ee_zpos,
                                                                min_ee_xy_pos=min_ee_xy_pos,
                                                                max_ee_xy_pos=max_ee_xy_pos,
                                                                use_sim_config = use_sim_config,
                                                                safety_dq_scale = safety_dq_scale)

    def reset(self, seed=None, options={}):
        # options: see object_reset_options dictionary in BasePushEnv
        super().reset(seed=seed, options=options)

        # reset object and target (object params + pose)
        self._reset_object_target(options=options)

        # reset joints to initial qpos
        self.reset_robot_qpos()

        self._reset_callback(options=options)

        mujoco.mj_forward(self.model, self.data)

        self.render()

        observation = self._get_obs()
        info = self._get_info(observation["achieved_goal"], observation["desired_goal"])

        return observation, info

    def render(self):
        if self.render_mode is not None:
            self._render_callback()
            return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        if self.render_mode is not None:
            self.mujoco_renderer.close()

    def get_binaryGeomImg(self, visible_geom="object"):
        rgb_image = self.mujoco_renderer.render(render_mode="rgb_array", camera_id=self.camera_id, width=self.image_width if self.use_sim_config else 640, height=self.image_height if self.use_sim_config else 480, geomgroup=self.geomgroup[visible_geom])
        binary_image = image_utils.rgbImg_to_binaryImg( rgb_img=rgb_image.copy(), 
                                                        min_max_hue=self.min_max_hue[visible_geom], 
                                                        min_max_saturation=self.min_max_saturation[visible_geom],
                                                        min_max_val=self.min_max_value[visible_geom])
        
        if not self.use_sim_config:
            binary_image = cv.flip(binary_image, 0)
            # crop
            binary_image = binary_image[130:,210:-80]
            # resize
            binary_image = cv.resize(binary_image, dsize=[self.image_width,self.image_height])
            binary_image[np.bitwise_and(binary_image!=0, binary_image!=1)] = 0

        return binary_image, rgb_image

    def reset_robot_qpos(self):
        for name, value in self.initial_qpos_dict.items():
            mujoco_utils.set_joint_qpos(self.model, self.data, name, value)
        mujoco.mj_forward(self.model, self.data)

    def reload_model(self, use_target_pose_as_obj_pose=False):
        model_xml_str = mujoco_utils.generate_model_xml_string( obj_type=self.obj_type,
                                                                obj_xy_pos=self.target_start_pos[0:2] if use_target_pose_as_obj_pose else self.obj_start_pos[0:2],
                                                                target_xy_pos=self.target_start_pos[0:2],
                                                                obj_quat=self.target_start_quat if use_target_pose_as_obj_pose else self.obj_start_quat,
                                                                target_quat=self.target_start_quat,
                                                                obj_size_0=self.obj_size_0,
                                                                obj_size_1=self.obj_size_1,
                                                                obj_height=self.obj_height,
                                                                obj_sliding_fric=self.obj_sliding_fric,
                                                                obj_torsional_fric=self.obj_torsional_fric,
                                                                robot_gravity_compensation=1,
                                                                fingertip_model=self.use_fingertip_sensor,
                                                                use_red_ee=self.encode_ee_pos,
                                                                use_sim_config=self.use_sim_config)
        
        original_cwd = os.getcwd()
        os.chdir(os.path.join(ur3_push_mujoco.__path__[0], "assets"))
        self.model = mujoco.MjModel.from_xml_string(model_xml_str)
        os.chdir(original_cwd)
        
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        if self.render_mode is not None:
            self.mujoco_renderer.reload_model(self.model, self.data)
        self.controller.load_model_data(self.model) # reload ids, qposadr...

    def _reset_callback(self, options={}):
        pass

    def _render_callback(self):
        pass

    def _init_object_params(self):
        geom = self.model.geom("object_geom")
        self.obj_mass = self.model.body_mass[geom.bodyid][0]
        self.obj_sliding_fric = geom.friction[0]
        self.obj_torsional_fric = geom.friction[1]
        self.obj_size_0 = geom.size[0]
        self.obj_size_1 = geom.size[1]

        self.obj_geom_type_value = geom.type[0]
        if self.obj_geom_type_value == 6:
            self.obj_type = "box"
            self.obj_size_2 = geom.size[2]
            self.obj_height = self.obj_size_2
        elif self.obj_geom_type_value == 5:
            self.obj_type = "cylinder"
            self.obj_size_2 = 0
            self.obj_height = self.obj_size_1
        else:
            logging.error("Object is not a box or a cylinder")
            self.obj_size_2 = geom.size[2]
            self.obj_height = self.obj_size_2

    def _reset_object_target(self, options={}):
        self._reset_object_target_config(options)
        # change quaternion order (x,y,z,w) -> (w,x,y,z) (tf.transformations -> mujoco)
        self.obj_start_quat = rotations_utils.tftansformationsQuat_to_mujocoQuat(self.obj_start_quat)
        self.target_start_quat = rotations_utils.tftansformationsQuat_to_mujocoQuat(self.target_start_quat)
        # normalize quaternions
        mujoco.mju_normalize4(self.obj_start_quat)
        mujoco.mju_normalize4(self.target_start_quat)

        # reload model with new object and target params
        self.reload_model(use_target_pose_as_obj_pose=self.use_camera)

        # set object_pose = target_pose and get new image -> use this image as target image
        if self.use_camera:
            self.target_binary_image, self.target_rgb_image = self.get_binaryGeomImg(visible_geom="object")

        self._set_object_pose(self.obj_start_pos, self.obj_start_quat, "object_joint")

    def _set_object_pose(self, pos, quat, joint_name):
        joint_qpos = mujoco_utils.get_joint_qpos(self.model, self.data, joint_name)
        assert joint_qpos.shape == (7,)
        joint_qpos[:3] = pos.copy()
        joint_qpos[3:] = quat.copy()
        mujoco_utils.set_joint_qpos(self.model, self.data, joint_name, joint_qpos)
        mujoco_utils.set_joint_qvel(self.model, self.data, joint_name, np.zeros(6))
        
    def _get_object_pose(self, site_name, add_noise=True):
        site = self.model.site(site_name)
        # position
        object_pose_m = self.data.site_xpos[site.id][0:2] + 0.001*self.rng_noise.standard_normal(size=2)*int(add_noise) # gaussian noise (approx. +/- 2mm)
        # orientation
        if self.consider_object_orientation:
            object_zangle = rotations.mat2euler(self.data.site_xmat[site.id,:].reshape(3, 3))[2] + (np.pi/180)*self.rng_noise.standard_normal()*int(add_noise) # euler: xyz
            object_zangle = rotations.normalize_angles(object_zangle) # -pi <= object_z_angle <= pi
            object_pose_m = np.append(object_pose_m, object_zangle)
        return object_pose_m
