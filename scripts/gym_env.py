#!/usr/bin/env python3

from __future__ import print_function
from six.moves import input

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

from math import pi, tau, dist, fabs, cos, sqrt, sin

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

import numpy as np
import rospy

import gym
from gym.spaces import Box, Dict 

from mujoco_ros_msgs.srv import ResetBodyQPos
from franka_gripper.msg import StopActionFeedback

from franka_msgs.msg import FrankaState
import message_filters, actionlib



def step_callback(data):

    rospy.

class CustomEnv(gym.Env):
    
    """Custom Environment that follows gym interface"""
    metadata = {}

    def __init__(self, cache_size):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = None
        
        #message data for observation
        self.current_msg = {'franka_state':None}
        self.last_timestep = {'franka_state':0}
        
        self.observation_space = Dict({
            "joint_positions" : Box(low=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175]),
                                    high=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525])
                                    dtype=np.float64),
            "joint_torques" : Box(low=np.array([0, 0, 0, 0, 0, 0]),
                                  high=np.array([87, 87, 87, 87, 12, 12]),
                                  dtype=np.float64),
            "ee_position" : Box(low=np.array([0, 0, 0]), 
                                high=np.array([np.inf, np.inf, np.inf]),
                                dtype=np.float64),
            "ee_orientation" :Box(low=np.array([-np.pi, -np.pi]),
                                  high=np.array([np.pi, np.pi]),
                                  dtype=np.float64)
        })

        self.action_space = Box(low=np.array([-0.1, -np.pi, -np.pi]),
                                high=np.array([0.1, np.pi, np.pi]),
                                dtype=np.float64)
        
        
        rospy.init_node("safe_object_placement")
        
        #subscribers for state information
        self.franka_state_sub   = message_filters.Subscriber("franka_state_controller/franka_states", FrankaState)
        self.franka_state_cache = message_filters.Cache(self.franka_state_sub, cache_size=cache_size, allow_headerless=False) 
        self.franka_state_cache.registerCallback(self._new_msg_callback)
        



    def step(self, action=None):
        
        pass

    def reset(self):
        pass 

    def render(self, mode='human'):
        pass

    def close (self):
        pass

    def _new_msg_callback(self, msg):
        topic = msg._connection_header["topic"]

        key = [k for k in self.current_msg.keys() if k in topic]
        if not key:
            rospy.logerr(f"No key found for current topic {topic} - current message is ignored")
        elif len(key) > 1:
            rospy.logerr(f"More than one key found for current topic {topic} - current message is ignored")
        else:
            if self.current_msg[key[0]] is not None:
                if msg.header.stamp < self.current_msg[key[0]].header.stamp:
                    rospy.logerr(f"Detected jump back in time - current message is ignored (topic: {topic})")
                    return
                self.last_timestamps[key[0]] = self.current_msg[key[0]].header.stamp # remember timestamp of old message
            self.current_msg[key[0]] = msg


if __name__=='__main__':

    env = CustomEnv()

    env.moveItPlanner.move_down(0.1)
    env.reset()

