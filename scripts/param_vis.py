#!/usr/bin/env python3

import os 
import sys
import numpy as np 
from scipy.spatial.transform import Rotation as R
import pyquaternion as pq
import copy

import gym
from gym.spaces import Box, Discrete, Dict

import rospy, rosgraph, roslaunch, rosservice
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseStamped, PointStamped, QuaternionStamped, Twist, Vector3Stamped
from franka_msgs.msg import FrankaState
from franka_msgs.srv import SetEEFrame, SetLoad
from sensor_msgs.msg import Image
from tactile_msgs.msg import TactileState
from franka_gripper.msg import GraspAction, GraspEpsilon, GraspGoal, MoveAction, MoveGoal, StopAction, StopGoal

from mujoco_ros_msgs.msg import GeomType, GeomProperties, BodyState, ScalarStamped, StepAction, StepGoal
from mujoco_ros_msgs.srv import SetGeomProperties, SetBodyState, SetPause, SetGravity, GetGravity
import tf.transformations
import message_filters, actionlib
import rospkg

import math
import time

from datetime import datetime
import cv2 as cv
global tactile

def NormalizeData(data, high, low):
    return (data - low) / (high - low)

def _new_msg_callback(msg):
    topic = msg._connection_header["topic"]
    global tactile
    if topic == '/myrmex_r':
        if msg is not None:
            tactile = msg

def increase_pixel_size(image, n):


    bigger = np.zeros((image.shape[0] * n, image.shape[1]*n))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            bigger[(i*n):((i+1)*n), (j*n):((j+1)*n)] = image[i][j] 

    return bigger

def vis_nonzero(vals):
    
    vals = vals.reshape((16,16))

    h = np.max(vals)
    l = np.min(vals)

    vals = NormalizeData(vals, h, l)
    
    vals *= 255

    vals = increase_pixel_size(vals, 16)
    # vals = cv.cvtColor(vals,cv.COLOR_GRAY2RGB)
    
    

    img2 = np.zeros((vals.shape[0], vals.shape[1], 3))
    img2[:,:,0] = vals
    img2[:,:,1] = vals
    img2[:,:,2] = vals

    vals = img2.astype(np.uint8)
    cv.applyColorMap(vals, cv.COLORMAP_JET)
    # vals = cv.resize(vals, (256, 256), interpolation= cv.INTER_NEAREST)

    cv.imshow('activations', vals)
    cv.waitKey(0)        
        

if __name__ == '__main__':
    
    tactile_right_sub = message_filters.Subscriber("/myrmex_r", TactileState)
    tactile_right_cache = message_filters.Cache(tactile_right_sub, cache_size=1, allow_headerless=False)
    tactile_right_cache.registerCallback(_new_msg_callback)

    tactile = None
    launch_file = "parameter_test.launch"
    if not rosgraph.is_master_online(): # check if ros master is already running
        print("starting launch file...")

        uuid = roslaunch.rlutil.get_or_generate_uuid(options_runid=None, options_wait_for_master=False)
        roslaunch.configure_logging(uuid)
        pkg_path = rospkg.RosPack().get_path("safe_placement")
        launch = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_files=[os.path.join(pkg_path, "launch", launch_file)], is_core=True)
        launch.start()

    rospy.init_node("testyNode")

    for i in range(5):

        if tactile is not None:
            vals = np.array(tactile.sensors[0].values)
     
            
            vis_nonzero(vals)

            vals = vals.reshape((16,16))
            # print(vals)
           
            nonzero = np.count_nonzero(vals)
            print(nonzero)
            nonzero_idcs = np.nonzero(vals)

            xs = nonzero_idcs[0]
            ys = nonzero_idcs[1]

            if len(nonzero_idcs[0]) > 0:
                
                xu = np.unique(xs)
                yu = np.unique(ys)
                
                x_length = len(xu)
                y_length = len(yu)

                xz = list(zip(xu, [i for i in range(len(xu))]))
                yz = list(zip(yu, [i for i in range(len(xu))]))
                #print(nonzero_idcs)

                grid = np.zeros((x_length, y_length))

                test = []
                c = 0
                for e_x, i_x in xz:
                    for e_y, i_y in xz:
                        # print(c)
                        # test.append(((x,y), (xs[c], ys[c])))
                        # grid[x][y] = vals[xs[c]][ys[c]]
                        # c+=1

                        grid[i_x][i_y] = vals[e_x][e_y]
                print(grid)
                nonzero = vals[nonzero_idcs]
                # min = np.min(nonzero)
                max = np.max(nonzero)
                mean = np.mean(nonzero)

                print( mean, max)


        time.sleep(1)
