#include <safe_placement/cartesian_twist_controller.h>

#include <cmath>
#include <iterator>
#include <algorithm>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/Dense>

#include <realtime_tools/realtime_buffer.h>
#include <realtime_tools/realtime_publisher.h>

#include <control_toolbox/pid.h>

#include <geometry_msgs/Twist.h>
#include <franka_hw/franka_model_interface.h>
#include <controller_interface/controller_base.h>
#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>


namespace safe_placement {

bool CartesianTwistController::init(hardware_interface::RobotHW* robot_hardware,
                                    ros::NodeHandle& node_handle) {

  velocity_joint_interface_ = robot_hardware->get<hardware_interface::VelocityJointInterface>();
  
  model_interface = robot_hardware->get<franka_hw::FrankaModelInterface>();
  
  // if (!pid_controller_.init(ros::NodeHandle(node_handle, "gains")))
  //   return false;

  std::string arm_id;

  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("CartesianTwistController: Could not get parameter arm_id");
    return false;
  }

  model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(model_interface->getHandle(arm_id + "_model"));

  if (model_handle_ == nullptr) {
    ROS_ERROR(
        "CartesianTwistController: Error getting model handle from Model interface!");
    return false;
  }

  
  if (model_interface == nullptr) {
    ROS_ERROR(
        "CartesianTwistController: Error getting Franka model interface from hardware!");
    return false;
  }
  if (velocity_joint_interface_ == nullptr) {
    ROS_ERROR(
        "CartesianTwistController: Error getting velocity joint interface from hardware!");
    return false;
  }

  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names)) {
    ROS_ERROR("CartesianTwistController: Could not parse joint names");
  }
  if (joint_names.size() != 7) {
    ROS_ERROR_STREAM("CartesianTwistController: Wrong number of joint names, got "
                     << joint_names.size() << " instead of 7 names!");
    return false;
  }
  velocity_joint_handles_.resize(7);
  for (size_t i = 0; i < 7; ++i) {
    try {
      velocity_joint_handles_[i] = velocity_joint_interface_->getHandle(joint_names[i]);
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "CartesianTwistController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  // effort_joint_handles_.resize(7);
  // for (size_t i = 0; i < 7; ++i) {
  //   try {
  //     effort_joint_handles_[i] = effort_joint_interface_->getHandle(joint_names[i]);
  //   } catch (const hardware_interface::HardwareInterfaceException& ex) {
  //     ROS_ERROR_STREAM(
  //         "CartesianTwistController: Exception getting joint handles: " << ex.what());
  //     return false;
  //   }
  // }

  // Load PID Controller using gains set on parameter server
  // pid_controllers_.resize(7);
  // for (size_t i=0; i<7; ++i) {
  //   pid_controllers_[i].init(ros::NodeHandle(node_handle, "/cartesian_twist_controller/gains/" + joint_names[i]));
  // }



  auto state_interface = robot_hardware->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR("CartesianTwistController: Could not get state interface from hardware");
    return false;
  }

  try {
    auto state_handle = state_interface->getHandle(arm_id + "_robot");

    std::array<double, 7> q_start{{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    for (size_t i = 0; i < q_start.size(); i++) {
      if (std::abs(state_handle.getRobotState().q_d[i] - q_start[i]) > 0.1) {
        ROS_ERROR_STREAM(
            "CartesianTwistController: Robot is not in the expected starting position for "
            "running this example. Run `roslaunch franka_example_controllers move_to_start.launch "
            "robot_ip:=<robot-ip> load_gripper:=<has-attached-gripper>` first.");
        return false;
      }
    }
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "CartesianTwistController: Exception getting state handle: " << e.what());
    return false;
  }


  sub_command_ = node_handle.subscribe("cmd_twist", 1, &CartesianTwistController::cmdTwistCallback, this);

  return true;
}

void CartesianTwistController::starting(const ros::Time& /* time */) {  
  
  // for (auto pid : pid_controllers_) {
  //   pid.reset();
  // }
}

void CartesianTwistController::update(const ros::Time& /* time */,
                                            const ros::Duration& period) {
  
  // 2. Listen to Realtime Buffer to get current control Twist signal
  Commands curr_cmd = *(command_.readFromRT());


  twistCommand = Eigen::Map<Eigen::Matrix<double,6, 1> >(curr_cmd.twist);


 
  std::array<double, 42> jacobianArray = model_handle_ -> getZeroJacobian(franka::Frame::kEndEffector);
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobianArray.data());
  
  // 1. Compute Pseudinverse of Jacobian. Singular Value Decomposition and damping from lecture

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian, Eigen::ComputeFullV | Eigen::ComputeFullU );
  
  // Invert singular values: Currently simple Iversion 1/s; to account for singularities implement methods from lecture!

  Eigen::VectorXd singularValues = svd.singularValues();
  Eigen::VectorXd invertedSingularValues(singularValues.size());

  for(int i=0; i<singularValues.size(); i++){
    invertedSingularValues[i] = 1 / singularValues[i];
  };

  Eigen::MatrixXd diaSiVa = invertedSingularValues.asDiagonal(); 
  Eigen::MatrixXd pseudoInverse = svd.matrixV().transpose() * diaSiVa * svd.matrixU();
  
  // Eigen::MatrixXd pseudoInverse = svd.matrixV().transpose() * svd.singularValues() * svd.matrixU().transpose();

  
  // 3. Compute Joint Velocities from Pseudoinverse & EE-Twist
  
  Eigen::VectorXd jointVels = pseudoInverse * twistCommand;
  
  double x;
  for (int i =0; i<jointVels.size(); i++) {
    
    // double vel_error = jointVels[i] - velocity_joint_handles_[i].getVelocity();

    // double commanded_velocity = pid_controllers_[i].computeCommand(vel_error, period);

    velocity_joint_handles_[i].setCommand(jointVels[i]);
  }


}

void CartesianTwistController::stopping(const ros::Time& /*time*/) {
  // WARNING: DO NOT SEND ZERO VELOCITIES HERE AS IN CASE OF ABORTING DURING MOTION
  // A JUMP TO ZERO WILL BE COMMANDED PUTTING HIGH LOADS ON THE ROBOT. LET THE DEFAULT
  // BUILT-IN STOPPING BEHAVIOR SLOW DOWN THE ROBOT.
}

void CartesianTwistController::cmdTwistCallback(const geometry_msgs::Twist& command) {

      
      command_struct_.twist[0] = command.angular.x;
      command_struct_.twist[1] = command.angular.y;
      command_struct_.twist[2] = command.angular.z;
      command_struct_.twist[3] = command.linear.x;
      command_struct_.twist[4] = command.linear.y;
      command_struct_.twist[5] = command.linear.z;
      command_struct_.stamp = ros::Time::now();
      command_.writeFromNonRT(command_struct_);
      
      // command_struct_.zTranslation   = command.angular.z;
      // command_struct_.lin   = command.linear.x;
      // command_struct_.stamp = ros::Time::now();
      // command_.writeFromNonRT (command_struct_);
      ROS_DEBUG_STREAM("Added values to command. " << "Twist: " << command_struct_.twist << ", Timestamp: " << command_struct_.stamp); 
  }


}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(safe_placement::CartesianTwistController,
                       controller_interface::ControllerBase)