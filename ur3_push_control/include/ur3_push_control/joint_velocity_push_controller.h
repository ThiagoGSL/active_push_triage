// Originalmente Copyright (c) 2017 Franka Emika GmbH
// Refatorado para UR3 (6 GDL) com garra RG2
#pragma once

#include <string>
#include <vector>

#include <ur3_push_control/ControllerConfig.h>
#include <dynamic_reconfigure/server.h>

#include <controller_interface/multi_interface_controller.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <Eigen/Dense>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>

namespace ur3_push_control {

class JointVelocityPushController : public controller_interface::MultiInterfaceController<
                                           hardware_interface::VelocityJointInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void update(const ros::Time&, const ros::Duration& period) override;
  void starting(const ros::Time&) override;
  void stopping(const ros::Time&) override;

 private:
  hardware_interface::VelocityJointInterface* velocity_joint_interface_;
  std::vector<hardware_interface::JointHandle> velocity_joint_handles_;
  ur3_push_control::ControllerConfig config_;
  
  double lambda_;
  double safety_dq_scale_;
  float min_ee_z_pos_;
  Eigen::Vector2d min_ee_xy_pos_, max_ee_xy_pos_;
  bool consider_x_cone_task_;

  Eigen::Vector3d position_d_, cone_ref_vec_x_, ee_x_axis_, cone_ref_vec_z_, ee_z_axis_;
  Eigen::Vector3d position_d_target_;
  Eigen::Quaterniond orientation_d_target_;
  Eigen::Matrix<double, 6, 1> q_init_,dq_d_;

  Eigen::MatrixXd vstack(Eigen::MatrixXd A, Eigen::MatrixXd B);
  Eigen::Matrix<double, 3, 3> vec2skewSymmetricMat(Eigen::Vector3d vec);
  Eigen::Vector3d clipXYZPos(geometry_msgs::Point msg_position);

  // position and velocity limits
  Eigen::Matrix<double, 6, 1> q_max_, q_min_, dq_max_, dq_min_;

  // equilibrium pose subscriber
  ros::Subscriber sub_equilibrium_pose_;
  void equilibriumPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void ReconfigureCallback(ur3_push_control::ControllerConfig &config, uint32_t level);

  // debug
  ros::Publisher pub_cone_error_;
  ros::Publisher pub_position_error_;
  geometry_msgs::PointStamped cone_task_error_msg, position_task_error_msg;

  dynamic_reconfigure::Server<ur3_push_control::ControllerConfig> server_;
};


inline void pseudoInverse(const Eigen::MatrixXd& M_, Eigen::MatrixXd& M_pinv_, double lambda = 0.01) {

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M_, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType sing_vals_ = svd.singularValues();
  Eigen::MatrixXd S_ = M_;  // copying the dimensions of M_, its content is not needed.
  S_.setZero();

  for (int i = 0; i < sing_vals_.size(); i++) {
    if (sing_vals_(i) >= lambda) {
      S_(i, i) = 1/sing_vals_(i);
    } else {
      S_(i, i) = sing_vals_(i) / (lambda * lambda);
    }
  }

  M_pinv_ = Eigen::MatrixXd(svd.matrixV() * S_.transpose() * svd.matrixU().transpose());
}

}  // namespace ur3_push_control
