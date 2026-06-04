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
            # Fricção mínima exigida pelo MuJoCo Warp (MJ_MINMU=1e-5).
            # friction="0 0 0" causa NaN na GPU após muitos steps.
            target_geom.set("friction", "1e-5 0 0")

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
        ET.SubElement(actuator, "velocity", name="v_servo_shoulder_pan",  joint="shoulder_pan_joint",  kv="50")
        ET.SubElement(actuator, "velocity", name="v_servo_shoulder_lift", joint="shoulder_lift_joint", kv="50")
        ET.SubElement(actuator, "velocity", name="v_servo_elbow",         joint="elbow_joint",         kv="50")
        ET.SubElement(actuator, "velocity", name="v_servo_wrist_1",       joint="wrist_1_joint",       kv="30")
        ET.SubElement(actuator, "velocity", name="v_servo_wrist_2",       joint="wrist_2_joint",       kv="20")
        ET.SubElement(actuator, "velocity", name="v_servo_wrist_3",       joint="wrist_3_joint",       kv="20")
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

        # Apply proportional gain (Kp) to convert position error to velocity command.
        # Without this, Kp is implicitly 1.0, causing extremely sluggish tracking.
        dq_d = dq_d * 10.0

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


class MuJoCoUR3PushControllerBatched:
    """Controlador IK Jacobiano vetorizado para N mundos em paralelo (MuJoCo Warp).

    Opera sobre arrays batched [N, ...] lidos da GPU via d.xxx.numpy().
    O cálculo do Jacobiano usa o MuJoCo padrão (CPU) com um MjData auxiliar
    por mundo, em um loop Python. O restante (cone error, pinv, clip) é NumPy.

    Fluxo por gym step:
        1. Ler site_xpos, site_xmat, qpos da GPU → NumPy [N, ...]
        2. update_desired_pose_batched(ee_pos_batch, actions) → ee_pos_d [N, 3]
        3. compute_ctrl_batched(qpos_batch, site_xpos_batch, site_xmat_batch, ee_pos_d_batch)
           → ctrl_batch [N, n_arm_joints]
        4. Escrever ctrl_batch na GPU → warp.copy(d.ctrl, wp.array(ctrl_batch))
        5. Executar n_substeps via CUDA graph na GPU
    """

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
        self.ee_site_name = ee_site_name
        self.initial_ee_zpos = initial_ee_zpos
        self.min_ee_xy_pos = min_ee_xy_pos
        self.max_ee_xy_pos = max_ee_xy_pos
        self.use_sim_config = use_sim_config
        self.safety_dq_scale = safety_dq_scale

        # Limites de velocidade e posição de junta (UR3, 6 GDL)
        self.dq_max = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14])
        self.dq_min = -self.dq_max
        self.q_max = np.array([6.2831853] * 6)
        self.q_min = -self.q_max

        # Carrega IDs do modelo CPU (compartilhado entre mundos — modelo é estático)
        self._model = model  # MjModel CPU (referência)
        self._load_model_ids(model)

        # MjData auxiliar CPU para cálculo de Jacobianos por mundo
        # Criamos um único objeto e restauramos o estado de cada mundo nele
        self._cpu_data = mujoco.MjData(model)

    def _load_model_ids(self, model: MjModel):
        """Carrega IDs de joints, actuators e site do modelo."""
        _, _, self.panda_joint_qposadr, self.panda_joint_dofadr = \
            get_model_robot_joints(model, self.robot_name)
        _, self.panda_actuator_ids = get_model_actuators(model, self.robot_name)
        self.ee_site_id = model.site(self.ee_site_name).id
        self.n_arm_joints = len(self.panda_joint_dofadr)  # 6 para UR3
        # Parametros de damping para pseudo-inversa (mesmos do controller_utils.pinv)
        self._pinv_damping  = 0.2    # use_sim_config=True (damped)
        self._pinv_theta    = 0.03   # use_sim_config=False (truncated)

    def reload_model_data(self, model: MjModel):
        """Recarrega IDs após recriação do modelo (ex: mudança de obj type)."""
        self._model = model
        self._cpu_data = mujoco.MjData(model)
        self._load_model_ids(model)

    def compute_ctrl_batched_vectorized(
            self,
            qpos_batch:      np.ndarray,   # [N, nq]
            site_xpos_batch: np.ndarray,   # [N, nsite, 3]
            site_xmat_batch: np.ndarray,   # [N, nsite, 3, 3]
            xanchor_batch:   np.ndarray,   # [N, ndof, 3]  — d.xanchor da GPU
            xaxis_batch:     np.ndarray,   # [N, ndof, 3]  — d.xaxis da GPU
            ee_pos_d_batch:  np.ndarray,   # [N, 3]
    ) -> np.ndarray:
        """IK Jacobiano vetorizado: elimina o loop Python sobre N mundos.

        Usa d.xanchor e d.xaxis do MuJoCo Warp para computar o Jacobiano
        analitico de forma totalmente batcheada (NumPy puro, sem mj_jacSite).
        SVD batcheado sobre [N, 4, 6] elimina overhead de N SVDs sequenciais.

        Equivalencia numerica com compute_ctrl_batched: max diff ~1e-4
        (float32 das arrays GPU vs float64 do mj_jacSite). Aceitavel para IK.

        Returns:
            ctrl_batch: [N, n_arm_joints] — velocidades de junta alvo
        """
        N = qpos_batch.shape[0]
        ctrl_batch = np.zeros((N, self.n_arm_joints), dtype=np.float64)

        # --- 1. Deteccao vetorizada de NaN (mundos com fisica instavel) ---
        valid = (
            np.all(np.isfinite(site_xpos_batch.reshape(N, -1)), axis=1) &
            np.all(np.isfinite(xanchor_batch.reshape(N, -1)), axis=1)  &
            np.all(np.isfinite(qpos_batch), axis=1)
        )  # [N] bool
        if not np.any(valid):
            return ctrl_batch
        idx = np.where(valid)[0]  # indices dos mundos validos
        M   = len(idx)

        # --- 2. Estado do EE [M, 3] ---
        ee_pos    = site_xpos_batch[idx, self.ee_site_id, :]      # [M, 3]
        ee_rotmat = site_xmat_batch[idx, self.ee_site_id, :, :]   # [M, 3, 3]
        ee_z      = ee_rotmat[:, :, 2]                            # [M, 3]

        # --- 3. Jacobiano analitico via xanchor/xaxis [M, 6, 3] ---
        arm_anchor = xanchor_batch[idx][:, self.panda_joint_dofadr, :]  # [M, 6, 3]
        arm_axis   = xaxis_batch[idx][:,   self.panda_joint_dofadr, :]  # [M, 6, 3]

        # Jacobiano translacional: axis x (ee_pos - anchor)  -> [M, 6, 3] -> [M, 3, 6]
        r    = ee_pos[:, None, :] - arm_anchor     # [M, 6, 3]
        jacp = np.cross(arm_axis, r).transpose(0, 2, 1)  # [M, 3, 6]
        # Jacobiano rotacional [M, 3, 6]
        jacr = arm_axis.transpose(0, 2, 1)          # [M, 3, 6]

        # --- 4. Jacobiano da tarefa combinada [M, 4, 6] ---
        # Linha 0: restricao de cone z: cone_ref @ skew(ee_z) @ jacr
        # Matriz skew-simetrica de ee_z: [M, 3, 3]
        S = np.zeros((M, 3, 3), dtype=np.float64)
        S[:, 0, 1] = -ee_z[:, 2];  S[:, 0, 2] =  ee_z[:, 1]
        S[:, 1, 0] =  ee_z[:, 2];  S[:, 1, 2] = -ee_z[:, 0]
        S[:, 2, 0] = -ee_z[:, 1];  S[:, 2, 1] =  ee_z[:, 0]

        cone_ref = np.array([0., 0., -1.], dtype=np.float64)
        # cone_ref @ S: [M, 3] — einsum sobre indice j
        cone_S   = np.einsum('j,mji->mi', cone_ref, S)             # [M, 3]
        # cone_S @ jacr: [M, 3] @ [M, 3, 6] -> [M, 6]
        cone_row = np.einsum('mi,mij->mj', cone_S, jacr)           # [M, 6]

        jac_conepos = np.zeros((M, 4, 6), dtype=np.float64)
        jac_conepos[:, 0, :]  = cone_row
        jac_conepos[:, 1:, :] = jacp

        # --- 5. Erro de tarefa [M, 4] ---
        cone_scalar = np.einsum('j,mj->m', cone_ref, ee_z) - 1.0  # [M]
        pos_error   = ee_pos_d_batch[idx] - ee_pos                 # [M, 3]
        task_error  = np.concatenate([cone_scalar[:, None], pos_error], axis=1)  # [M, 4]

        # --- 6. Pseudo-inversa batcheada via SVD [M, 4, 6] -> [M, 6, 4] ---
        try:
            U, s, Vt = np.linalg.svd(jac_conepos, full_matrices=False)
            # full_matrices=False: U[M,4,4], s[M,4], Vt[M,4,6]
            if self.use_sim_config:
                # Damped pseudo-inverse (lambda = 0.2)
                s_d = s / (s**2 + self._pinv_damping**2)          # [M, 4]
            else:
                # Truncated pseudo-inverse (theta = 0.03)
                s_d = np.where(s >= self._pinv_theta,
                               1.0 / s,
                               s / self._pinv_theta**2)            # [M, 4]
            # J_pinv = Vt^T @ diag(s_d) @ U^T
            jac_pinv = (Vt.transpose(0, 2, 1) * s_d[:, None, :]) @ U.transpose(0, 2, 1)
            # [M, 6, 4]
            # dq = Kp * J_pinv @ error
            dq_d = 10.0 * np.einsum('mij,mj->mi', jac_pinv, task_error)  # [M, 6]
        except np.linalg.LinAlgError:
            # SVD nao convergiu (fallback: sem controle)
            return ctrl_batch

        # --- 7. Clip vetorizado de limites de junta ---
        q_arm = qpos_batch[idx][:, self.panda_joint_qposadr]  # [M, 6]
        if self.use_sim_config:
            dq_min_pw = np.maximum(self.dq_min[None, :], self.q_min[None, :] - q_arm)
            dq_max_pw = np.minimum(self.dq_max[None, :], self.q_max[None, :] - q_arm)
            dq_d = self.safety_dq_scale * np.clip(dq_d, dq_min_pw, dq_max_pw)
        else:
            dq_min_pw = np.maximum(self.dq_min[None, :], self.q_min[None, :] - q_arm)
            dq_max_pw = np.minimum(self.dq_max[None, :], self.q_max[None, :] - q_arm)
            with np.errstate(divide='ignore', invalid='ignore'):
                scales = np.maximum(
                    np.where(dq_min_pw != 0, dq_d / dq_min_pw, 0.0),
                    np.where(dq_max_pw != 0, dq_d / dq_max_pw, 0.0),
                )  # [M, 6]
            scales[~np.isfinite(scales)] = 1.0
            scale_max = np.maximum(1.0, np.max(scales, axis=1, keepdims=True))  # [M, 1]
            dq_d = (self.safety_dq_scale * dq_d) / scale_max

        ctrl_batch[idx] = dq_d
        return ctrl_batch  # [N, 6]

    def update_desired_pose_batched(self,
                                    ee_pos_batch: np.ndarray,
                                    actions_batch: np.ndarray,
                                    action_scaling_factor: float = 1.0) -> np.ndarray:
        """Calcula posição desejada do EE para N mundos.

        Args:
            ee_pos_batch: [N, 3] — posição atual do EE para cada mundo
            actions_batch: [N, 2] — ações do agente (delta x, delta y)
            action_scaling_factor: fator de escala das ações

        Returns:
            ee_pos_d: [N, 3] — posição desejada do EE (com clipping)
        """
        N = ee_pos_batch.shape[0]
        ee_pos_d = ee_pos_batch.copy()  # [N, 3]

        # Incrementa x, y com ação escalada
        ee_pos_d[:, 0] += actions_batch[:, 0] * action_scaling_factor
        ee_pos_d[:, 1] += actions_batch[:, 1] * action_scaling_factor

        # Força altura z fixa (altura de push)
        ee_pos_d[:, 2] = self.initial_ee_zpos

        # Clipping dos limites do workspace [N, 2]
        ee_pos_d[:, :2] = np.clip(ee_pos_d[:, :2],
                                   self.min_ee_xy_pos,
                                   self.max_ee_xy_pos)
        return ee_pos_d  # [N, 3]

    def compute_ctrl_batched(self,
                              qpos_batch: np.ndarray,
                              site_xpos_batch: np.ndarray,
                              site_xmat_batch: np.ndarray,
                              ee_pos_d_batch: np.ndarray) -> np.ndarray:
        """Calcula velocidades de junta para N mundos via IK Jacobiano.

        Para cada mundo i:
        - Restaura qpos no MjData CPU auxiliar
        - Calcula Jacobiano via mujoco.mj_jacSite
        - Resolve dq = pinv(J) @ error * Kp
        - Aplica limites de velocidade e posição

        Args:
            qpos_batch:    [N, nq] — posições de junta de todos os mundos
            site_xpos_batch: [N, nsite, 3] — posições dos sites
            site_xmat_batch: [N, nsite, 9] — matrizes de rotação dos sites
            ee_pos_d_batch: [N, 3] — posição desejada do EE

        Returns:
            ctrl_batch: [N, n_arm_joints] — velocidades de junta alvo
        """
        N = qpos_batch.shape[0]
        ctrl_batch = np.zeros((N, self.n_arm_joints), dtype=np.float64)

        # Buffers pré-alocados para Jacobianos (evita alocação a cada iteração)
        nv = self._model.nv
        jacp_full = np.zeros((3, nv))
        jacr_full = np.zeros((3, nv))

        cone_ref_vec_z = np.array([[0, 0, -1]])

        for i in range(N):
            # --- Verifica NaN no estado do mundo i (physica instavel) ---
            if (not np.all(np.isfinite(site_xpos_batch[i])) or
                    not np.all(np.isfinite(site_xmat_batch[i])) or
                    not np.all(np.isfinite(qpos_batch[i]))):
                # Mundo com NaN: saida zero (robô parado) — evita crash no SVD
                ctrl_batch[i] = 0.0
                continue

            # --- Restaura estado do mundo i no MjData CPU auxiliar ---
            self._cpu_data.qpos[:] = qpos_batch[i]
            self._cpu_data.qvel[:] = 0.0  # velocidades não necessárias para IK
            mujoco.mj_kinematics(self._model, self._cpu_data)
            mujoco.mj_comPos(self._model, self._cpu_data)

            # --- Lê estado do EE para o mundo i ---
            ee_pos_i = site_xpos_batch[i, self.ee_site_id, :]          # [3]
            # site_xmat shape do MJWarp: [N, nsite, 3, 3] (já é matriz 3x3)
            ee_rotmat_i = site_xmat_batch[i, self.ee_site_id, :, :]    # [3, 3]
            ee_z_axis_i = ee_rotmat_i[:, 2]

            # --- Calcula Jacobiano (translacional e rotacional) ---
            jacp_full[:] = 0.0
            jacr_full[:] = 0.0
            mujoco.mj_jacSite(self._model, self._cpu_data,
                               jacp_full, jacr_full, self.ee_site_id)

            # Seleciona apenas as colunas dos joints do braço (6 GDL)
            jacp_arm = jacp_full[:, self.panda_joint_dofadr]  # [3, 6]
            jacr_arm = jacr_full[:, self.panda_joint_dofadr]  # [3, 6]

            # --- Erro de tarefa: cone z + posição (4 DoF total) ---
            conepos_error = np.zeros(4)
            conepos_error[0] = float(cone_ref_vec_z @ ee_z_axis_i.reshape(-1, 1) - 1)
            conepos_error[1:] = ee_pos_d_batch[i] - ee_pos_i

            # --- Jacobiano da tarefa combinada [4, 6] ---
            jac_conepos = np.zeros((4, 6))
            jac_conepos[0, :] = (cone_ref_vec_z @
                                  controller_utils.vec2SkewSymmetricMat(ee_z_axis_i) @
                                  jacr_arm).flatten()
            jac_conepos[1:, :] = jacp_arm

            # --- Pseudoinversa + ganho proporcional ---
            try:
                jac_pinv = controller_utils.pinv(jac_conepos, use_damping=self.use_sim_config)
            except np.linalg.LinAlgError:
                # SVD nao convergiu (Jacobiano singular ou contem NaN)
                ctrl_batch[i] = 0.0
                continue
            dq_d = jac_pinv @ conepos_error
            dq_d = dq_d * 10.0  # Kp

            # --- Clipping de limites de velocidade e posição ---
            q_i = qpos_batch[i][self.panda_joint_qposadr]
            dq_d = self._clip_joint_limits(dq_d, q_i)

            ctrl_batch[i] = dq_d

        return ctrl_batch  # [N, 6]

    def _clip_joint_limits(self, dq: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Aplica limites de velocidade considerando posição atual (sim config)."""
        if self.use_sim_config:
            dq = self.safety_dq_scale * np.clip(
                dq,
                a_min=np.maximum(self.dq_min, self.q_min - q),
                a_max=np.minimum(self.dq_max, self.q_max - q)
            )
        else:
            pos_aware_dq_min = np.maximum(self.dq_min, self.q_min - q)
            pos_aware_dq_max = np.minimum(self.dq_max, self.q_max - q)
            scales = np.maximum(dq / pos_aware_dq_min, dq / pos_aware_dq_max)
            scales[~np.isfinite(scales)] = 1.0
            dq = (self.safety_dq_scale * dq) / np.maximum(1.0, np.max(scales))
        return dq