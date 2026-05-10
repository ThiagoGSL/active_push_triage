#!/usr/bin/env python3
import rospy
from franka_msgs.srv import SetEEFrame
import numpy as np
import tf.transformations

rospy.init_node("set_fixed_ee_trafo_node")

# wait until required service is available
required_srv = "/franka_control/set_EE_frame"
rospy.wait_for_service(required_srv)

ee_trafo_tmp = np.transpose(np.reshape(np.array([*rospy.get_param("/fixed_EE_trafo")]), (4, 4)))
ee_trafo = tf.transformations.inverse_matrix(ee_trafo_tmp)
rospy.logwarn(ee_trafo_tmp)

set_EE_frame = rospy.ServiceProxy("/franka_control/set_EE_frame", SetEEFrame)
resp = set_EE_frame(NE_T_EE=tuple((ee_trafo.T).ravel()))  # column-major format
if not resp.success:
    rospy.logerr("Failed to set fixed EE trafo")
