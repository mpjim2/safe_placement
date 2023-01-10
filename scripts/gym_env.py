import os 
import numpy as np 
from scipy.spatial.transform import Rotation as R

import gym
from gym.spaces import Box, Discrete, Dict

import rospy, rosgraph, roslaunch, rosservice
from geometry_msgs import PoseStamped, PointStamped, QuaternionStamped, Twist
from franka_msgs.msg import FrankaState
from franka_msgs.srv import SetEEFrame
from tactile_msgs.msg import TactileState
from franka_gripper.msg import GraspAction, GraspEpsilon, GraspGoal, MoveAction, MoveGoal

from mujoco_ros_msgs.msg import GeomType, GeomProperties, BodyState
from mujoco_ros_msgs.srv import SetGeomProperties, SetBodyState
import tf.transformations
import message_filters, actionlib
import rospkg


class SafePlacementEnv(gym.Env):
    metadata = {}

    def __init__(self):

        super().__init__()

        rospy.init_node("SafePlacementEnvNode")

        self.franka_state_sub = message_filters.Subscriber("franka_state_controller/franka_states", FrankaState)
        self.franka_state_cache = message_filters.Cache(self.franka_state_sub, cache_size=1, allow_headerless=False)
        self.franka_state_cache.registerCallback(self._new_msg_callback)

        self.current_msg = {"franka_state" : None}
        self.last_timestamps = {"franka_state" : 0}


        self.action_space = Box(low=-1, high=1, shape=(3), dtype=np.int8)

        self.observation_space = Dict({
            "observation" : Dict({
                "ee_pose" : Box( low=np.array([-np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi]), 
                                                high=np.array([np.inf, np.inf, np.inf, np.pi, np.pi, np.pi]),
                                                dtype=np.float64),
                "joint_positions" : Box(low=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]), 
                                       high=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
                                       dtype=np.float64),
                "joint_torques" : Box(low=np.array([-87, -87, -87, -87, -12, -12, -12]),
                                        high=np.array([87, 87, 87, 87, 12, 12, 12]),
                                        dtype=np.float64),
                "joint_velocities" : Box(low=np.array([]),
                                         high=np.array([]),
                                         dtype=np.float64)
            })
        })


        all_services_available = False
        required_srv = ["/franka_control/set_EE_frame", "/mujoco_server/set_geom_properties", "/mujoco_server/set_body_state"]
        while not all_services_available:
            service_list = rosservice.get_service_list()
            if not [srv for srv in required_srv if srv not in service_list]:
                all_services_available = True
                print("All Services Available!")

        self.set_geom_properties = rospy.ServiceProxy("/mujoco_server/set_geom_properties", SetGeomProperties)
        self.set_body_state = rospy.ServiceProxy("/mujoco_server/set_body_state", SetBodyState)
        
        #client for MoveAction ie open gripper
        self.release_client = actionlib.SimpleActionClient("/franka_gripper/move", MoveAction)
        rospy.loginfo("Waiting for MoveAction action server...")
        self.release_client.wait_for_server()

        #client for grasp action (only needed when resetting the environment)
        self.grasp_client = actionlib.SimpleActionClient("/franka_gripper/grasp", GraspAction)
        rospy.loginfo("Waiting for GraspAction action server...")
        self.grasp_client.wait_for_server()
        rospy.loginfo("Action server started!")


    def _pose_quat_from_trafo(self, transformationEE):
        
        transformationEE = np.array(transformationEE).reshape((4,4), order='F')
        
        orientation = transformationEE[:3, :3]
        position    = transformationEE[:3, -1]
        
        orientation = R.from_matrix(orientation)

        orientation_q = orientation.as_quat()
        orientation_xyz = orientation.as_eurler('xyz')
        return position, orientation_q, orientation_xyz

    def _get_observation(self):
        
        current_obs = self.current_msg['franka_state']
        ee_pos, ee_quat, ee_xyz = self._pose_quat_from_trafo(current_obs.O_T_EE)
        pose = np.array((ee_pos, ee_xyz)).reshape(6)

        joint_positions = np.array(current_obs.q)
        joint_torques = np.array(current_obs.tau_J)
        joint_velocities = np.array(current_obs.dq)

        observation = {"ee_pose" : pose, 
                       "joint_positions" : joint_positions,
                       "joint_torques" : joint_torques,
                       "joint_velocities" : joint_velocities}

        return observation

    def step(self):
        pass

    def reset(self):
        pass

    def close(self):
        pass