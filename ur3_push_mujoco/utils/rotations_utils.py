import numpy as np
from gymnasium_robotics.utils.rotations import normalize_angles

def tftansformationsQuat_to_mujocoQuat(tf_quat):
    # change quaternion order (x,y,z,w) -> (w,x,y,z) (tf.transformations -> mujoco)
    return np.roll(tf_quat, 1)

def mujocoQuat_to_tftransformationsQuat(mujoco_quat):
    # change quaternion order (w,x,y,z) -> (x,y,z,w) (mujoco -> tf.transformations)
    return np.roll(mujoco_quat, -1)

def unnormalize_angles(angles):
    # puts angles in [0, 2*pi] range
    if isinstance(angles, np.ndarray):
        angles = angles.copy()
        mask = angles < 0
        angles[mask] = 2*np.pi + angles[mask]
    elif isinstance(angles, float) or isinstance(angles, np.float32) or isinstance(angles, np.float64):
        if angles < 0:
            angles = 2*np.pi + angles
    else:
        raise TypeError("'angles' has unexpected type")
    return angles

def add_normalized_angles(a1, a2):
    a1_u = unnormalize_angles(a1)
    a2_u = unnormalize_angles(a2)

    a3_u = np.mod(a1_u + a2_u, 2*np.pi)
    return normalize_angles(a3_u)