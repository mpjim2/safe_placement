#pragma once

#include <string>
#include <vector>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/Dense>

#include <realtime_tools/realtime_buffer.h>
#include <realtime_tools/realtime_publisher.h>

#include <control_toolbox/pid.h>

#include <geometry_msgs/Twist.h>
#include <controller_interface/multi_interface_controller.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>

namespace safe_placement {

class CartesianTwistController : public controller_interface::MultiInterfaceController<
                                           franka_hw::FrankaModelInterface,
                                           franka_hw::FrankaStateInterface,
                                           hardware_interface::VelocityJointInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) override;
  void update(const ros::Time&, const ros::Duration& period) override;
  void starting(const ros::Time&) override;
  void stopping(const ros::Time&) override;
  

 private:
  hardware_interface::VelocityJointInterface* velocity_joint_interface_;
  franka_hw::FrankaModelInterface* model_interface;

  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::vector<hardware_interface::JointHandle> velocity_joint_handles_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  ros::Duration elapsed_time_;
  Eigen::VectorXd twistCommand;
  struct Commands
    {
      double twist[6];
      ros::Time stamp;

      Commands() : twist{}, stamp(0.0) {}
    };
  
  realtime_tools::RealtimeBuffer<Commands> command_; // command_ is name of realtime buffer
  Commands command_struct_;
  ros::Subscriber sub_command_; //subscriber to listen to commands

  Eigen::Vector3d position_d_;
  Eigen::Quaterniond orientation_d_;
  std::mutex position_and_orientation_d_target_mutex_;
  Eigen::Vector3d position_d_target_;
  Eigen::Quaterniond orientation_d_target_;
  // std::shared_ptr<realtime_tools::RealtimePublisher<tf::tfMessage> > tf_odom_pub_;

  std::vector<control_toolbox::Pid> pid_controllers_;

  private:
    void cmdTwistCallback(const geometry_msgs::Twist& command);
};

}  // namespace franka_example_controllers