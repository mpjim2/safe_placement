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

def quaternion_multiply(Q0,Q1):

    # Extract the values from Q0
    
    x0 = Q0[0]
    y0 = Q0[1]
    z0 = Q0[2]
    w0 = Q0[3]
    # Extract the values from Q1
    
    x1 = Q1[0]
    y1 = Q1[1]
    z1 = Q1[2]
    w1 = Q1[3]
    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([Q0Q1_x, Q0Q1_y, Q0Q1_z, Q0Q1_w])
    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32) 
    return final_quaternion

def axis_angle_to_quaternion(axis, angle):

    qx = axis[0] * sin(angle/2)
    qy = axis[1] * sin(angle/2)
    qz = axis[2] * sin(angle/2)
    qw = cos(angle/2)

    return [qx, qy, qz, qw]

def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True


class MoveItMetaPlanner(object):
    
    def __init__(self):
        super(MoveItMetaPlanner, self).__init__()


        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("MoveItMetaActionPlanner", anonymous=True)

        
        robot = moveit_commander.RobotCommander()

        scene = moveit_commander.PlanningSceneInterface()

        group_name = 'panda_arm'
        move_group = moveit_commander.MoveGroupCommander(group_name)


        planning_frame = move_group.get_planning_frame()

        eef_link = move_group.get_end_effector_link()

        group_names = robot.get_group_names()

        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names


    def move_down(self, x):
        
        current_pose = self.move_group.get_current_pose().pose 

        pose_goal = self.move_group.get_current_pose().pose
        
        
        pose_goal.position.z -= x 

        self.move_group.set_pose_target(pose_goal)


        success = self.move_group.go(wait=True)


        self.move_group.stop()

        self.move_group.clear_pose_targets()

 
        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)
        

    
    def rotate_EE(self, theta, axis):
        
        current_pose = self.move_group.get_current_pose().pose
        pose_goal    = self.move_group.get_current_pose().pose

        quaternion      = axis_angle_to_quaternion(axis=axis, angle=theta)
        orientation_cur = [current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]
        
        orientation_new = quaternion_multiply(orientation_cur, quaternion)
        
        pose_goal.orientation.x = orientation_new[0]
        pose_goal.orientation.y = orientation_new[1]
        pose_goal.orientation.z = orientation_new[2]
        pose_goal.orientation.w = orientation_new[3]

        self.move_group.set_pose_target(pose_goal)

        success = self.move_group.go(wait=True)

        self.move_group.stop()
    
        self.move_group.clear_pose_targets()
    
        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    
    def open_gripper(self):
        pass
    
    
      



if __name__=='__main__':

    test = MoveItMetaPlanner()
    

    print('rotate around Y')
    test.rotate_EE(theta=0.3*pi, axis=[0,1,0])
    print('rotate around X')
    test.rotate_EE(theta=0.3*pi, axis=[1,0,0])


  



    print('rotate around Z')
    test.rotate_EE(theta=1.5*pi, axis=[0,0,1])
    
    print("move down")
    test.move_down(-0.1)

