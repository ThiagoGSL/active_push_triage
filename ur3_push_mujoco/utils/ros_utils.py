import numpy as np

import rospy
import tf.transformations
from geometry_msgs.msg import PoseStamped, Pose

def to_msec(stamp):
    return int(stamp.to_nsec()*1e-6)

def pose_to_nparrays(pose):
    pos = point_to_nparray(pose.position)
    quat = quaternion_to_nparray(pose.orientation)
    return pos, quat

def point_to_nparray(point):
    return np.array([point.x, point.y, point.z])

def quaternion_to_nparray(quat):
    return np.array([quat.x, quat.y, quat.z, quat.w])

def nparray_to_pose(np_pos, np_quat):
    pose = Pose()
    pose.position.x = np_pos[0]
    pose.position.y = np_pos[1]
    pose.position.z = np_pos[2]
    pose.orientation.x = np_quat[0]
    pose.orientation.y = np_quat[1]
    pose.orientation.z = np_quat[2]
    pose.orientation.w = np_quat[3]

    return pose

def nparray_to_posestamped(np_pos, np_quat, frame_id="world"):
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = frame_id
    pose.pose.position.x = np_pos[0]
    pose.pose.position.y = np_pos[1]
    pose.pose.position.z = np_pos[2]
    pose.pose.orientation.x = np_quat[0]
    pose.pose.orientation.y = np_quat[1]
    pose.pose.orientation.z = np_quat[2]
    pose.pose.orientation.w = np_quat[3]

    return pose

def FrankaState_to_measured_ee_pose(state_msg):
    # read measured ee pose in base frame
    ee_pose_measured = PoseStamped()
    ee_pose_measured.header = state_msg.header

    measured_quaternion = tf.transformations.quaternion_from_matrix(
        np.transpose(np.reshape(state_msg.O_T_EE, (4, 4))))
    measured_quaternion = measured_quaternion / np.linalg.norm(measured_quaternion)
    ee_pose_measured.pose.orientation.x = measured_quaternion[0]
    ee_pose_measured.pose.orientation.y = measured_quaternion[1]
    ee_pose_measured.pose.orientation.z = measured_quaternion[2]
    ee_pose_measured.pose.orientation.w = measured_quaternion[3]
    ee_pose_measured.pose.position.x = state_msg.O_T_EE[12]
    ee_pose_measured.pose.position.y = state_msg.O_T_EE[13]
    ee_pose_measured.pose.position.z = state_msg.O_T_EE[14]

    return ee_pose_measured

def FrankaState_to_measured_joint_pos(state_msg):
    # read measured joint position (rad)
    return np.array(state_msg.q).copy()

def FrankaState_to_measured_joint_velo(state_msg):
    # read measured joint velocity (rad/s)
    return np.array(state_msg.dq).copy()
