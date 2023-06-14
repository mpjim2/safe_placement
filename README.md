# safe_placement

Code written for the master's thesis "Learning to safely place objects based on tactile feedback using Deep Reinforcement Learning". 

# Installation

Pull this repository into the src folder of your ROS-Workspace, that has mujoco_ros_pkgs, mujoco_contact_surfaces, tactile_toolbox and franka_ros_mujoco installed and run `catkin build`.  

# Running trained models

To run the trained models provided in this repository, source the ws and navigate to `YOUR_ROS_WS/src/safe_placement`. From there run `python scripts/evaluate.py --model=MODEL_ID`. 
There are four models available: 
  - MODEL_ID=0 : Tactile-only 
  - MODEL_ID=1 : Tactile+Torques+EEorientation
  - MODEL_ID=2 : Tactile+Full Robot state
  - MODEL_ID=3 : Flat 16x16 tactile array + EEorientation 
