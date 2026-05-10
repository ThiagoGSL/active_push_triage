from typing import Dict, Tuple, Union
from mujoco import MjData, MjModel
import numpy as np
import os, ur3_push_mujoco

from gymnasium import error
from gymnasium_robotics.utils.mujoco_utils import (get_site_jacp,
                                                   get_site_jacr,
                                                   set_joint_qpos,
                                                   set_joint_qvel,
                                                   get_joint_qpos,
                                                   get_joint_qvel,
                                                   get_site_xpos,
                                                   get_site_xmat,
                                                   MujocoModelNames)

try:
    import mujoco
    from mujoco import MjModel, MjData
except ImportError as e:
    raise error.DependencyNotInstalled(f"{e}. (HINT: you need to install mujoco")

from ur3_push_mujoco.utils import controller_utils, rotations_utils


def get_model_robot_joints(model: MjModel, robot_name: str):
    mj_model_names = MujocoModelNames(model)

    if robot_name == "ur3":
        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    else:
        joint_names = [name for name in mj_model_names.joint_names if robot_name in name]
        
    joint_ids = [mj_model_names.joint_name2id[joint_name] for joint_name in joint_names]
    joint_qposadr = model.jnt_qposadr[joint_ids]
    joint_dofadr = model.jnt_dofadr[joint_ids]

    return joint_names, joint_ids, joint_qposadr, joint_dofadr


def get_model_actuators(model: MjModel, robot_name: str="ur3"):
    mj_model_names = MujocoModelNames(model)

    if robot_name == "ur3":
        actuator_names = ["v_servo_shoulder_pan", "v_servo_shoulder_lift", "v_servo_elbow", "v_servo_wrist_1", "v_servo_wrist_2", "v_servo_wrist_3"]
    else:
        actuator_names = [name for name in mj_model_names.actuator_names]
        
    actuator_ids = [mj_model_names.actuator_name2id[actuator_name] for actuator_name in actuator_names]

    return actuator_names, actuator_ids


def set_geom_material(model: MjModel, geom_name: str, material_name: str):
    material_id = model.material(material_name).id
    geom_id = model.geom(geom_name).id
    model.geom_matid[geom_id] = material_id


import xml.etree.ElementTree as ET

def generate_model_xml_string(
                            obj_type: str="box",
                            obj_xy_pos: np.ndarray=np.array([0.3, -0.1]),
                            target_xy_pos: np.ndarray=np.array([0.5, 0.1]),
                            obj_quat: np.ndarray=np.array([1,0,0,0]),
                            target_quat: np.ndarray=np.array([1,0,0,0]),
                            obj_size_0: float=0.055,
                            obj_size_1: float=0.055,
                            obj_height: float=0.04,
                            obj_mass: float=0.2,
                            obj_sliding_fric: float=0.8,
                            obj_torsional_fric: float=0.005,
                            robot_gravity_compensation: float=1,
                            fingertip_model: bool=False,
                            use_red_ee: bool=False,
                            path_to_assets: str=os.path.join(ur3_push_mujoco.__path__[0], "assets"),
                            include_camera: bool=True, 
                            include_actuator: bool=True,
                            use_sim_config: bool = True):
    
    # Load base XML for UR3
    xml_path = os.path.join(path_to_assets, "push_world_ur3_base.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Convert relative paths to absolute so MuJoCo can find them when parsing from string
    for inc in root.findall(".//include"):
        f = inc.get("file")
        if f:
            inc.set("file", os.path.abspath(os.path.join(path_to_assets, f)))
    for mesh in root.findall(".//mesh"):
        f = mesh.get("file")
        if f:
            mesh.set("file", os.path.abspath(os.path.join(path_to_assets, f)))
    
    # Set texture directory as absolute path
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("texturedir", os.path.abspath(os.path.join(path_to_assets, "textures")))

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    
    # Update table
    if use_sim_config:
        table_surface_z = 0.02      # Z of table TOP surface (where robot pushes objects)
        table_thickness = 0.4       # thick table extending DOWNWARD to prevent collision tunneling
        table_height = table_surface_z  # used for object spawn Z calculation
        table_center_z = table_surface_z - table_thickness / 2  # center is below surface
        table_pos = f"0.525 0.0 {table_center_z}"
        table_size = f"0.35 0.45 {table_thickness / 2}"
    else:
        table_surface_z = 0.02
        table_thickness = 0.4
        table_height = table_surface_z
        table_center_z = table_surface_z - table_thickness / 2
        table_pos = f"0.0 0.595 {table_center_z}"
        table_size = f"0.45 {0.35 + 0.06} {table_thickness / 2}"

    table = worldbody.find("./geom[@name='table']")
    if table is not None:
        table.set("pos", table_pos)
        table.set("size", table_size)
        table.set("friction", f"{obj_sliding_fric} {obj_torsional_fric} 0.0001")

    # Update target
    target_body = worldbody.find("./body[@name='target']")
    if target_body is not None:
        target_body.set("pos", f"{target_xy_pos[0]} {target_xy_pos[1]} {table_height + obj_height + 0.001}")
        target_body.set("quat", f"{target_quat[0]} {target_quat[1]} {target_quat[2]} {target_quat[3]}")
        
        # Remove joint so it stays fixed
        joint = target_body.find("joint")
        if joint is not None:
            target_body.remove(joint)

        target_geom = target_body.find("./geom[@name='target_geom']")
        if target_geom is not None:
            geom_size_str = f"{obj_size_0} {obj_height}" if obj_type == "cylinder" else f"{obj_size_0} {obj_size_1} {obj_height}"
            target_geom.set("size", geom_size_str)
            target_geom.set("type", obj_type)
            target_geom.set("contype", "0")
            target_geom.set("conaffinity", "0")
            target_geom.set("rgba", "0 1 0 0.5") # Green semi-transparent
            target_geom.set("mass", "0")
            target_geom.set("friction", "0 0 0")

    # Update object
    object_body = worldbody.find("./body[@name='object']")
    if object_body is not None:
        object_body.set("pos", f"{obj_xy_pos[0]} {obj_xy_pos[1]} {table_height + obj_height + 0.001}")
        object_body.set("quat", f"{obj_quat[0]} {obj_quat[1]} {obj_quat[2]} {obj_quat[3]}")
        object_geom = object_body.find("./geom[@name='object_geom']")
        if object_geom is not None:
            geom_size_str = f"{obj_size_0} {obj_height}" if obj_type == "cylinder" else f"{obj_size_0} {obj_size_1} {obj_height}"
            object_geom.set("size", geom_size_str)
            object_geom.set("type", obj_type)
            object_geom.set("mass", str(obj_mass))
            object_geom.set("friction", f"{obj_sliding_fric} {obj_torsional_fric} 0.0001")

    # Camera
    if include_camera:
        if use_sim_config:
            camera_pos = "0.4 0.0 -0.1"
            camera_z_angle = "0.0"
            camera_fovy = "65"
        else:
            camera_pos = "-0.08 0.552 0.19"
            camera_z_angle = "0"
            camera_fovy = "117"
        cam_body = ET.SubElement(worldbody, "body", name="camera_link", pos=camera_pos, euler=f"0 0 {camera_z_angle}")
        ET.SubElement(cam_body, "camera", name="rgb_cam", mode="fixed", fovy=camera_fovy, euler="3.141 0 0")
        ET.SubElement(cam_body, "geom", name="cam_geom", pos="0 0 0", type="box", size="0.05 0.05 0.05", rgba="0.8 0 0 0.2", group="0")


    # Actuator (UR3 joints)
    if include_actuator:
        actuator = root.find("actuator")
        if actuator is None:
            actuator = ET.SubElement(root, "actuator")
        ET.SubElement(actuator, "velocity", name="v_servo_shoulder_pan",  joint="shoulder_pan_joint",  kv="30")
        ET.SubElement(actuator, "velocity", name="v_servo_shoulder_lift", joint="shoulder_lift_joint", kv="30")
        ET.SubElement(actuator, "velocity", name="v_servo_elbow",         joint="elbow_joint",         kv="30")
        ET.SubElement(actuator, "velocity", name="v_servo_wrist_1",       joint="wrist_1_joint",       kv="30")
        ET.SubElement(actuator, "velocity", name="v_servo_wrist_2",       joint="wrist_2_joint",       kv="10")
        ET.SubElement(actuator, "velocity", name="v_servo_wrist_3",       joint="wrist_3_joint",       kv="10")
        ET.SubElement(actuator, "velocity", name="v_servo_finger",        joint="finger_joint",        kv="50")

    # Convert tree back to string
    return ET.tostring(root, encoding="utf8").decode("utf8")

class MuJoCoUR3PushController():

    def __init__(self, 
                 model: MjModel, 
                 robot_name: str, 
                 ee_site_name: str, 
                 initial_ee_zpos: float, 
                 min_ee_xy_pos: np.ndarray = np.array([0.1, -0.25]),
                 max_ee_xy_pos: np.ndarray = np.array([0.65, 0.25]),
                 use_sim_config: bool = True,
                 safety_dq_scale: float = 1.0):
        self.robot_name = robot_name
        # EE site and initial zPos
        self.ee_site_name = ee_site_name
        self.initial_ee_zpos = initial_ee_zpos

        self.load_model_data(model)

        # min max x,y EE pos
        self.min_ee_xy_pos = min_ee_xy_pos
        self.max_ee_xy_pos = max_ee_xy_pos

        # position and velocity limits for UR3 robot (6 GDL)
        # max, min joint velocities (rad/s)
        self.dq_max = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14])
        self.dq_min = (-1)*self.dq_max

        # max, min joint positions (rad)
        self.q_max = np.array([6.2831853, 6.2831853, 6.2831853, 6.2831853, 6.2831853, 6.2831853])
        self.q_min = (-1)*self.q_max

        # controller config (sim vs. real)
        self.use_sim_config = use_sim_config

        # safety velocity scale 
        self.safety_dq_scale = safety_dq_scale

    def load_model_data(self, model):
        # remember robot joint names, ids, qposadr and dofadr
        self.panda_joint_names, self.panda_joint_ids, self.panda_joint_qposadr, self.panda_joint_dofadr = get_model_robot_joints(model, self.robot_name)
        # actuator names and ids
        self.panda_actuator_names, self.panda_actuator_ids = get_model_actuators(model, self.robot_name)
        # EE site 
        self.ee_site_id = model.site(self.ee_site_name).id

    def update_desired_pose(self, model: MjModel, data: MjData, action: np.ndarray):
        assert action.shape[0] == 2 or action.shape[0] == 3

        # desired EE pos
        ee_pos = get_site_xpos(model, data, self.ee_site_name)
        self.ee_pos_d = np.array([ee_pos[0] + action[0], ee_pos[1] + action[1], self.initial_ee_zpos])
        self.ee_pos_d[:2] = np.minimum(np.maximum(self.ee_pos_d[:2], self.min_ee_xy_pos), self.max_ee_xy_pos)

        if action.shape[0] == 3:
            ee_rotmat = data.site_xmat[self.ee_site_id,:].reshape(3, 3)
            # align x-axis of base frame with x-axis of ee frame
            self.cone_ref_vec_x = np.array([[1,0,0]])
            ee_x_axis = ee_rotmat[:,0]
            # angle between x-axis (base frame) and projection of x-axis into xy-plane of base frame
            current_angle = np.arctan2(ee_x_axis[1], ee_x_axis[0])
            desired_angle = rotations_utils.add_normalized_angles(current_angle, action[-1])
            self.cone_ref_vec_x = np.array([[np.cos(desired_angle),np.sin(desired_angle),0]])

    def update(self, model: MjModel, data: MjData, action: np.ndarray):

        # get Jacobians and EE pose
        ee_rotmat = data.site_xmat[self.ee_site_id,:].reshape(3, 3)
        ee_pos = get_site_xpos(model, data, self.ee_site_name)
        jacp = get_site_jacp(model, data, self.ee_site_id)[:, self.panda_joint_dofadr]
        jacr = get_site_jacr(model, data, self.ee_site_id)[:, self.panda_joint_dofadr]

        # cone and position error
        task_size = 4 + int(action.shape[0] == 3)
        conepos_error = np.zeros(task_size)

        # align z-axis of ee frame with z-axis*(-1) of base frame
        cone_ref_vec_z = np.array([[0,0,-1]])
        ee_z_axis = ee_rotmat[:,2]
       
        if action.shape[0] == 3:
            conepos_error[0] = (self.cone_ref_vec_x @ self.ee_x_axis.reshape(-1,1) - 1) # error cone task (x-axis)

        idx_cone_z_task = int(action.shape[0] == 3)
        conepos_error[idx_cone_z_task] = (cone_ref_vec_z @ ee_z_axis.reshape(-1,1) - 1) # error cone task (z-axis)
        conepos_error[idx_cone_z_task+1:] = self.ee_pos_d - ee_pos # error position task

        # cone and position task (Jacobiana do UR3 é 6 GDL)
        jac_conepos = np.zeros((task_size, 6))

        if action.shape[0] == 3:
            ee_x_axis = ee_rotmat[:,0]
            jac_conepos[0, 5] = self.cone_ref_vec_x @ controller_utils.vec2SkewSymmetricMat(ee_x_axis) @ jacr[:, 5]
        jac_conepos[idx_cone_z_task, :6] = cone_ref_vec_z @ controller_utils.vec2SkewSymmetricMat(ee_z_axis) @ jacr[:, :6]
        jac_conepos[idx_cone_z_task + 1:, :6] = jacp[:,:6].copy()

        jac_conepos_pinv = controller_utils.pinv(jac_conepos, use_damping=self.use_sim_config)
        dq_d = jac_conepos_pinv @ conepos_error

        # set desired joint velocities and ensure joint position and velocity limits
        data.ctrl[self.panda_actuator_ids] = self.ensure_joint_pos_velo_limits(data, dq_d)
    
    def ensure_joint_pos_velo_limits(self, data: MjData, dq: np.ndarray):
        q = data.qpos[self.panda_joint_qposadr]

        if self.use_sim_config:
            dq = self.safety_dq_scale * np.clip(dq, a_min=np.maximum(self.dq_min, self.q_min - q), a_max=np.minimum(self.dq_max, self.q_max - q))
        else:
            pos_aware_dq_min = np.maximum(self.dq_min, self.q_min - q)
            pos_aware_dq_max = np.minimum(self.dq_max, self.q_max - q)
            # uniformly scale qdot if any limit is exceeded
            scales = np.maximum(dq / pos_aware_dq_min, dq / pos_aware_dq_max)
            scales[np.logical_not(np.isfinite(scales))] = 1.0
            dq = (self.safety_dq_scale * dq) / np.maximum(1.0, np.max(scales))

        assert (dq <= self.dq_max).all()

        return dq