// Originalmente Copyright (c) 2017 Franka Emika GmbH
// Refatorado para UR3 (6 GDL) com garra RG2
#include <ur3_push_control/joint_velocity_push_controller.h>

#include <cmath>

#include <controller_interface/controller_base.h>
#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>

namespace ur3_push_control {

bool JointVelocityPushController::init(hardware_interface::RobotHW* robot_hw,
                                          ros::NodeHandle& node_handle) {

	sub_equilibrium_pose_ = node_handle.subscribe(
      "equilibrium_pose", 20, &JointVelocityPushController::equilibriumPoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

	// debug
  pub_cone_error_ = node_handle.advertise<geometry_msgs::PointStamped>("cone_task_error", 100);
  pub_position_error_ = node_handle.advertise<geometry_msgs::PointStamped>("position_task_error", 100);

  velocity_joint_interface_ = robot_hw->get<hardware_interface::VelocityJointInterface>();
  if (velocity_joint_interface_ == nullptr) {
    ROS_ERROR(
        "JointVelocityPushController: Error getting velocity joint interface from hardware!");
    return false;
  }

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("JointVelocityPushController: Could not get parameter arm_id");
    return false;
  }

  // if (!node_handle.getParam("safety_dq_scale_", safety_dq_scale_)) {
  //   ROS_ERROR("JointVelocityPushController: Could not get parameter pos_error_scale");
  //   return false;
  // }
  // if (safety_dq_scale_ > 1.) {
  //   ROS_ERROR("JointVelocityPushController: safety_dq_scale_ should not be greater than 1!");
  //   return false;
  // }

  if (!node_handle.getParam("min_ee_z_pos", min_ee_z_pos_)){
    ROS_ERROR(
        "JointVelocityPushController: Could not read parameter min_ee_z_pos");
    return false;
  }

  if (!node_handle.getParam("min_ee_xy_pos/x", min_ee_xy_pos_(0))){
    ROS_ERROR(
        "JointVelocityPushController: Could not read parameter min_ee_xy_pos/x");
    return false;
  }

  if (!node_handle.getParam("min_ee_xy_pos/y", min_ee_xy_pos_(1))){
    ROS_ERROR(
        "JointVelocityPushController: Could not read parameter min_ee_xy_pos/y");
    return false;
  }

  if (!node_handle.getParam("max_ee_xy_pos/x", max_ee_xy_pos_(0))){
    ROS_ERROR(
        "JointVelocityPushController: Could not read parameter max_ee_xy_pos/x");
    return false;
  }

  if (!node_handle.getParam("max_ee_xy_pos/y", max_ee_xy_pos_(1))){
    ROS_ERROR(
        "JointVelocityPushController: Could not read parameter max_ee_xy_pos/y");
    return false;
  }

  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names)) {
    ROS_ERROR("JointVelocityPushController: Could not parse joint names");
  }
  if (joint_names.size() != 6) {
    ROS_ERROR_STREAM("JointVelocityPushController: Wrong number of joint names, got "
                     << joint_names.size() << " instead of 6 names (UR3)!");
    return false;
  }
  velocity_joint_handles_.resize(6);
  for (size_t i = 0; i < 6; ++i) {
    try {
      velocity_joint_handles_[i] = velocity_joint_interface_->getHandle(joint_names[i]);
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "JointVelocityPushController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  // UR3: não utiliza franka_hw model/state interface.
  // A Jacobiana é obtida via ur_kinematics ou KDL a partir do URDF.

	position_d_.setZero();
  dq_d_.setZero();

  consider_x_cone_task_ = false;

  // Limites de posição do UR3: ±2π para todas as juntas
  q_max_ << 6.2832, 6.2832, 6.2832, 6.2832, 6.2832, 6.2832;
  q_min_ << -6.2832, -6.2832, -6.2832, -6.2832, -6.2832, -6.2832;

  // Limites de velocidade do UR3: máx 3.14 rad/s (segurança do laboratório)
  dq_max_ << 3.14, 3.14, 3.14, 3.14, 3.14, 3.14;
  dq_min_ = (-1) * dq_max_;

  server_.setCallback(boost::bind(&JointVelocityPushController::ReconfigureCallback, this, _1, _2));

  return true;
}

void JointVelocityPushController::ReconfigureCallback(ur3_push_control::ControllerConfig &config, uint32_t level) {
  if (level == 0xFFFFFFFF) {
      config.min_height = min_ee_z_pos_;
  }
  
  min_ee_z_pos_ = config.min_height;
  lambda_ = config.lambda;
  safety_dq_scale_ = config.safety_dq_scale;
  if (safety_dq_scale_ > 1.) {
    ROS_ERROR("JointVelocityPushController: safety_dq_scale_ should not be greater than 1!");
    safety_dq_scale_ = 0.2;
  }

  ros::param::set("/joint_velocity_push_controller/min_ee_z_pos", min_ee_z_pos_);

  ROS_INFO_STREAM("Applied new clustering config! New z min: " << min_ee_z_pos_ << "; new lambda: " << lambda_);
  return;
}

void JointVelocityPushController::starting(const ros::Time& /* time */) {
  // UR3: inicializar posição e velocidade com zeros.
  // A leitura de estado real é feita pelo driver ur_robot_driver via joint_states.
  dq_d_.setZero();
  q_init_.setZero();
  position_d_.setZero();

  cone_ref_vec_z_ << 0, 0, -1;
  consider_x_cone_task_ = false;
}

void JointVelocityPushController::update(const ros::Time& /* time */,
                                            const ros::Duration& period) {

  // get state variables via joint_states (UR3 driver)
  // NOTA: substituir estas linhas com a leitura real do ur_robot_driver
  // A Jacobiana deve ser calculada via KDL ou ur_kinematics com base no q atual.
  // Abaixo mantém-se a estrutura algébrica para compilação; adaptar a fonte de dados.

  // Jacobiana da tarefa: 6 (espaço cartesiano) x 6 (juntas UR3)
  // [ADAPTAR]: popular jacobian a partir de ur_kinematics::forward() ou KDL
  Eigen::Matrix<double, 6, 6> jacobian;
  jacobian.setZero();
  Eigen::Matrix<double, 6, 1> q;   // posição das juntas (rad)
  Eigen::Matrix<double, 6, 1> dq;  // velocidade das juntas (rad/s)
  q.setZero();
  dq.setZero();

  // [ADAPTAR]: obter transformação EE via cinematica direta UR3
  Eigen::Affine3d transform = Eigen::Affine3d::Identity();
  Eigen::Vector3d position(transform.translation());

  // === errors ====
  // cone task errors
  Eigen::VectorXd conepos_error(4);
  int idx_z_task = 0;
  // cone tasks
  if (consider_x_cone_task_){
    // align x-axis of base frame with x-axis of ee frame
    ee_x_axis_ << transform(0,0), transform(1,0), transform(2,0); // axis end-effector frame; orientation w.r.t. base frame
    conepos_error(0) = cone_ref_vec_x_.transpose()*ee_x_axis_ - 1;
    conepos_error.conservativeResize(5);
    idx_z_task = 1;
  }
  // align z-axis of ee frame with z-axis*(-1) of base frame
  ee_z_axis_ << transform(0,2), transform(1,2), transform(2,2); // axis end-effector frame; orientation w.r.t. base frame
  conepos_error(idx_z_task) = cone_ref_vec_z_.transpose()*ee_z_axis_ - 1; 
  // position task error 
  conepos_error.bottomRows(3) = (position_d_ - position);// * safety_dq_scale_;

  // === Jacobians ===
  Eigen::MatrixXd J_conepos(conepos_error.size(), 6);
  J_conepos.setZero();
  // posição e rotação: submatrizes 3×6 da Jacobiana do UR3
  Eigen::Matrix<double, 3, 6> J_pos(jacobian.topRows(3));
  Eigen::Matrix<double, 3, 6> J_rot(jacobian.bottomRows(3));
  // cone tasks
  if (consider_x_cone_task_){
    Eigen::Matrix<double,3,3> hat_x(vec2skewSymmetricMat(ee_x_axis_));
    J_conepos.block(0,6,1,1) = cone_ref_vec_x_.transpose() * hat_x * J_rot.block(0,6,3,1);
  }
  Eigen::Matrix<double,3,3> hat_z(vec2skewSymmetricMat(ee_z_axis_));
  J_conepos.block(idx_z_task,0,1,6) = cone_ref_vec_z_.transpose() * hat_z * J_rot.block(0,0,3,6);
  J_conepos.block(idx_z_task + 1,0,3,6) = J_pos.block(0,0,3,6);

  // compute and set desired joint velocity (UR3: vetor de 6)
	Eigen::VectorXd dq_conepos(6);
	Eigen::MatrixXd J_conepos_pinv;
	pseudoInverse(J_conepos, J_conepos_pinv, lambda_); 
	dq_d_ = J_conepos_pinv * conepos_error;

  // ensure position and velocity limits (UR3: 6 GDL)
  Eigen::Matrix<double, 6, 1> pos_aware_dq_min = dq_min_.cwiseMax(q_min_ - q);
  Eigen::Matrix<double, 6, 1> pos_aware_dq_max = dq_max_.cwiseMin(q_max_ - q);

  Eigen::Matrix<double, 6, 1> scales = (dq_d_.array() / pos_aware_dq_min.array()).cwiseMax(dq_d_.array() / pos_aware_dq_max.array());
  scales = scales.unaryExpr([](double v) { return std::isfinite(v)? v : 1.0; });
  dq_d_ = (safety_dq_scale_ * dq_d_) / std::max(1., scales.maxCoeff());
  
	for (size_t i = 0; i < 6; ++i) {
    velocity_joint_handles_[i].setCommand(dq_d_[i]);
  }

  // debug
  cone_task_error_msg.header.stamp = ros::Time::now();
  cone_task_error_msg.point.x = conepos_error(0,0);
  cone_task_error_msg.point.y = cone_task_error_msg.point.z = 0;
  pub_cone_error_.publish(cone_task_error_msg);

  position_task_error_msg.header.stamp = ros::Time::now();
  position_task_error_msg.point.x = conepos_error(1,0);
  position_task_error_msg.point.y = conepos_error(2,0);
  position_task_error_msg.point.z = conepos_error(3,0);
  pub_position_error_.publish(position_task_error_msg);
  cone_task_error_msg.header.stamp = ros::Time::now();
  cone_task_error_msg.point.x = conepos_error(0,0);
  cone_task_error_msg.point.y = cone_task_error_msg.point.z = 0;
  pub_cone_error_.publish(cone_task_error_msg);

  position_task_error_msg.header.stamp = ros::Time::now();
  position_task_error_msg.point.x = conepos_error(1,0);
  position_task_error_msg.point.y = conepos_error(2,0);
  position_task_error_msg.point.z = conepos_error(3,0);
  pub_position_error_.publish(position_task_error_msg);

}

Eigen::MatrixXd JointVelocityPushController::vstack(Eigen::MatrixXd A, Eigen::MatrixXd B){
  Eigen::MatrixXd C(A.rows()+B.rows(), A.cols());
  C << A,B;
  return C;
}

Eigen::Matrix<double, 3, 3> JointVelocityPushController::vec2skewSymmetricMat(Eigen::Vector3d vec){
  Eigen::Matrix<double, 3, 3> hat;
  hat << 0, -vec(2), vec(1), 
         vec(2), 0, -vec(0),
         -vec(1), vec(0), 0;
  return hat;
}

void JointVelocityPushController::equilibriumPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg){
	// desired pose w.r.t. base frame
  // position
  position_d_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  position_d_.topRows(2) = position_d_.topRows(2).cwiseMax(min_ee_xy_pos_).cwiseMin(max_ee_xy_pos_);
  if (position_d_(2) < min_ee_z_pos_){
    position_d_(2) = min_ee_z_pos_;
    ROS_DEBUG_STREAM_NAMED("JointVelocityPushController","Desired target z-pos is smaller than min z position.");
  }

  // orientation
  if (msg->header.frame_id == "z_task") consider_x_cone_task_ = false;
  else consider_x_cone_task_ = true;

  if (consider_x_cone_task_){
    orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z, msg->pose.orientation.w;
    Eigen::Matrix3d rot_mat(orientation_d_target_.toRotationMatrix());

    cone_ref_vec_x_ << rot_mat(0,0), rot_mat(1,0), 0;
    if(abs(rot_mat(2,0)) > 1e-2){
      cone_ref_vec_x_.normalize();
      ROS_INFO_STREAM_NAMED("JointVelocityPushController","z-component of cone reference vector is not zero. Will use normalized projection on xy-plane of base frame: " << cone_ref_vec_x_.transpose());
    }
  }
}

void JointVelocityPushController::stopping(const ros::Time& /*time*/) {
  // WARNING: DO NOT SEND ZERO VELOCITIES HERE AS IN CASE OF ABORTING DURING MOTION
  // A JUMP TO ZERO WILL BE COMMANDED PUTTING HIGH LOADS ON THE ROBOT. LET THE DEFAULT
  // BUILT-IN STOPPING BEHAVIOR SLOW DOWN THE ROBOT.
}

}  // namespace ur3_push_control

PLUGINLIB_EXPORT_CLASS(ur3_push_control::JointVelocityPushController,
                       controller_interface::ControllerBase)
