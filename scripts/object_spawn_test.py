#!/usr/bin/env python3

import os
import numpy as np
from copy import deepcopy
import gym 
from gym.spaces import Box, Dict
import pyquaternion as pq


from math import sin, cos, pi


from scipy.spatial.transform import Rotation as R


from franka_gripper.msg import GraspAction, GraspEpsilon, GraspGoal, MoveAction, MoveGoal
from franka_msgs.srv import SetLoad

import rospy, rosgraph, roslaunch, rosservice
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseStamped, PointStamped, QuaternionStamped, Twist
from franka_msgs.msg import FrankaState
from franka_msgs.srv import SetEEFrame
from tactile_msgs.msg import TactileState
from mujoco_ros_msgs.msg import GeomType, GeomProperties, BodyState, StepAction, StepGoal
from mujoco_ros_msgs.srv import SetGeomProperties, SetBodyState
import tf.transformations
import message_filters, actionlib
import rospkg 
import time

def sample_obj_params(object_params=dict()):

    obj_types = object_params.get("types", ["box"])
    range_obj_radius = object_params.get("range_radius", np.array([0.04, 0.055])) # [m]
    range_obj_height = object_params.get("range_height", np.array([0.15/2, 0.2/2])) # [m]
    range_obj_wl = object_params.get("range_wl", np.array([0.07/2, 0.12/2])) # [m]; range width and length (requires object type "box")
    range_obj_mass = object_params.get("range_mass", np.array([0.01, 0.1])) # [kg]
    range_obj_sliding_fric = object_params.get("range_sliding_fric", np.array([0.69, 0.7])) # ignore torsional friction, because object cannot be a sphere
    
    # sample new object/target type
    idx_type = np.random.randint(low=0, high=len(obj_types))
    obj_type = obj_types[idx_type]

    # sample new object/target mass
    obj_mass = np.random.uniform(low=range_obj_mass[0], high=range_obj_mass[1])

    # sample new sliding friction parameter
    obj_sliding_fric = np.random.uniform(low=range_obj_sliding_fric[0], high=range_obj_sliding_fric[1])

    # sample new object/target size depending on object type
    if obj_type == "box":
        obj_geom_type_value = 6
        # sample object width/2, length/2 and height/2 
        obj_size_0 = np.random.uniform(low=range_obj_wl[0], high=range_obj_wl[1]) # width/2
        obj_size_1 = np.random.uniform(low=range_obj_wl[0], high=range_obj_wl[1]) # length/2
        obj_size_2 = np.random.uniform(low=range_obj_height[0], high=np.amin(np.array([range_obj_height[1], obj_size_0, obj_size_1]))) # height/2
        obj_height = obj_size_2
    else:
        # obj_type == "cylinder"
        obj_geom_type_value = 5
        # sample object height/2 and radius
        obj_size_0 = np.random.uniform(low=range_obj_radius[0], high=range_obj_radius[1]) # radius
        obj_size_1 = np.random.uniform(low=range_obj_height[0], high=range_obj_height[1]) # height/2
        obj_size_2 = 0
        obj_height = obj_size_1

    return obj_geom_type_value,  obj_size_0, obj_size_1, obj_size_2, obj_sliding_fric, obj_mass

def sample_object_pose(obj_height, object_params=dict()):
    range_obj_x_pos = object_params.get("range_x_pos", np.array([0.275,0.525])) #[m]
    range_obj_y_pos = object_params.get("range_y_pos", np.array([-0.2,0.2])) #[m]
    # sample rotation about Y-axis
    y_angle = np.random.uniform(low=-np.pi, high=np.pi)
    # sample x-pos
    x_pos = np.random.uniform(low=range_obj_x_pos[0], high=range_obj_x_pos[1])
    # sample y-pos 
    y_pos = np.random.uniform(low=range_obj_y_pos[0], high=range_obj_y_pos[1])

    pos = np.array([x_pos, y_pos, obj_height + 5])
    quat = tf.transformations.quaternion_from_euler(0,y_angle,0)

    return pos, quat


def arr_to_pose_msg(pos, quat):

    pose = PoseStamped()
    
    pose.header.stamp =  rospy.Time.now()
    
    pose.pose.position.x = pos[0]
    pose.pose.position.y = pos[1]
    pose.pose.position.z = pos[2]

    pose.pose.orientation.x = quat[0]
    pose.pose.orientation.y = quat[1]
    pose.pose.orientation.z = quat[2]
    pose.pose.orientation.w = quat[3]

    return pose

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm


def axis_angle_to_quaternion(axis, angle):

    qx = axis[0] * sin(angle/2)
    qy = axis[1] * sin(angle/2)
    qz = axis[2] * sin(angle/2)
    qw = cos(angle/2)

    return normalize(np.array([qx, qy, qz, qw]))


def quaternion_multiplication(q1, q2):

    q_x = q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1]
    q_y = q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0]
    q_z = q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3]
    q_w = q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2]

    q = np.array([q_x, q_y, q_z, q_w])
   
    return q

def quat_dot(q1, q2):

    return q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]

class ObjectPoseSpawn:

    def __init__(self):
       
        rospy.init_node("object_spawn_test_node")

        self.franka_state_sub = message_filters.Subscriber("franka_state_controller/franka_states", FrankaState)
        self.franka_state_cache = message_filters.Cache(self.franka_state_sub, cache_size=1, allow_headerless=False)
        self.franka_state_cache.registerCallback(self._new_msg_callback)

        self.current_msg = {"franka_state" : None}
        self.last_timestamps = {"franka_state" : 0}
        
        all_services_available = False
        required_srv = ["/franka_control/set_EE_frame", 
                        "/franka_control/set_load",
                        "/mujoco_server/set_geom_properties", 
                        "/mujoco_server/set_body_state", 
                        "/mujoco_server/reset",
                        "/mujoco_server/pause"]

        while not all_services_available:
            service_list = rosservice.get_service_list()
            if not [srv for srv in required_srv if srv not in service_list]:
                all_services_available = True
                print("All Services Available!")

       
        self.set_geom_properties = rospy.ServiceProxy("/mujoco_server/set_geom_properties", SetGeomProperties)
        self.set_body_state      = rospy.ServiceProxy("/mujoco_server/set_body_state", SetBodyState)
        self.reset_world         = rospy.ServiceProxy("/mujoco_server/reset", Empty)
        self.set_load            = rospy.ServiceProxy("/franka_control/set_load", SetLoad)
    

        self.twist_pub = rospy.Publisher('/cartesian_impedance_controller/twist_cmd', Twist, queue_size=1)

        #client for MoveAction
        self.release_client = actionlib.SimpleActionClient("/franka_gripper/move", MoveAction)
        rospy.loginfo("Waiting for MoveAction action server...")
        self.release_client.wait_for_server()

        #client for grasp action (only needed when resetting the environment)
        self.grasp_client = actionlib.SimpleActionClient("/franka_gripper/grasp", GraspAction)
        rospy.loginfo("Waiting for GraspAction action server...")
        self.grasp_client.wait_for_server()
        rospy.loginfo("Action server started!")



    def set_object_params(self):
        obj_geom_type_value,  obj_size_0, obj_size_1, obj_size_2, obj_sliding_fric, obj_mass = sample_obj_params()

        obj_geom_type = GeomType(value=obj_geom_type_value)
        obj_geom_properties = GeomProperties(env_id=0, name="object_geom", type=obj_geom_type, size_0=obj_size_0, size_1=obj_size_1, size_2=obj_size_2, friction_slide=1, friction_spin=0.005, friction_roll=0.0001)
        resp = self.set_geom_properties(properties=obj_geom_properties, set_type=True, set_mass=False, set_friction=True, set_size=True)
        if not resp.success:
            rospy.logerr("SetGeomProperties:failed to set object parameters")
        
        ee_pos, ee_quat = self.compute_EE_pose()

        _, obj_quat = sample_object_pose(obj_size_1)

        ee_pos[-1] -= obj_size_2/2
        body_state = BodyState(env_id=0, name="object", pose=arr_to_pose_msg(ee_pos, obj_quat), mass=obj_mass)
        resp = self.set_body_state(state=body_state, set_pose=True, set_twist=False, set_mass=True, reset_qpos=False)
        if not resp.success:
            rospy.logerr("SetBodyState: failed to set object pose or object mass")

        return obj_size_1

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

    def compute_goal_pose(self):
        pass
        
    def compute_EE_pose(self):
        
        transformationEE = self.current_msg['franka_state'].O_T_EE
        transformationEE = np.array(transformationEE).reshape((4,4), order='F')
        
        orientation = transformationEE[:3, :3]
        position    = transformationEE[:3, -1]
        
        orientation = R.from_matrix(orientation)

        orientation_q = orientation.as_quat()
        
        return position, orientation_q

    def open_gripper(self):
        goal = MoveGoal(width=0.08, speed=0.05)

        self.release_client.send_goal(goal)
        self.release_client.wait_for_result(rospy.Duration(3))

        success = self.release_client.get_result()
        if not success:
            rospy.logerr("Open Action failed!")

        return success

    def close_gripper(self, width=0.05, eps=0.0001, speed=0.1, force=20):
        
        epsilon = GraspEpsilon(inner=eps, outer=eps)
        goal_ = GraspGoal(width=width, epsilon=epsilon, speed=speed, force=force)
        
        self.grasp_client.send_goal(goal_)
        self.grasp_client.wait_for_result(rospy.Duration(3))

        success = self.grasp_client.get_result()

        if not success:
            rospy.logerr("Grasp Action failed!")

        return success

    def getMass(self):

        mass = self.current_msg['franka_state'].m_load


        return mass

    def setLoad(self): 
        
        mass = 0.078*2 #mass of two ubi fingertips
        F_x_center_load = [0,0,0.17]
        load_inertia = [0, 0, 0.000651, 0, 0, 0.000651, 0, 0, 0.000896]

        #{0,0,0.17} translation from flange to in between fingers; default for f_x_center
        response = self.set_load(mass=mass, F_x_center_load=F_x_center_load, load_inertia=load_inertia)
        
        return response.success  

    def moveArm(self, rotate_X=0, rotate_Y=0, translateZ=0):

        pos_init, _ = self.compute_EE_pose()

        quat = pq.Quaternion()

        # quat = deepcopy(quat_init)
        if rotate_X > 0:
            quat = quat * pq.Quaternion(axis=[1, 0, 0], angle=0.04)
        elif rotate_X < 0:
            quat = quat * pq.Quaternion(axis=[1, 0, 0], angle=-0.04)
        if rotate_Y > 0:
            quat = quat * pq.Quaternion(axis=[0, 1, 0], angle=0.04)
        elif rotate_Y < 0:
            quat = quat * pq.Quaternion(axis=[0, 1, 0], angle=-0.04)
        
        # if quat_dot(quat_init, quat) < 0:
        # quat *= -1
        
        # twist_quat = quat_init.inverse * quat
        
        xyz = quat.vector 

        pos = np.zeros(3)
        if translateZ > 0:
            pos[2] -= 0.005
        elif translateZ < 0:
            pos[2] += 0.005

        twist_msg = Twist()

        twist_msg.linear.x = pos[0]
        twist_msg.linear.y = pos[1]
        twist_msg.linear.z = pos[2]

        twist_msg.angular.x = xyz[0]
        twist_msg.angular.y = xyz[1]
        twist_msg.angular.z = xyz[2]

        self.twist_pub.publish(twist_msg)


if __name__=='__main__':

    test = ObjectPoseSpawn()

    s = False

    while not s:
        s = test.open_gripper()
        w = test.set_object_params()
        s = test.close_gripper(width=w*2)
        

    # measured_mass = test.getMass()

    # time.sleep(1)
    # print("Measured Mass at Endeffector: ", measured_mass)

    # w = test.set_object_params()
    # test.setLoad()

    # test.moveArm(translateZ=True)
    # time.sleep(5)

    # total_X = 0
    # total_Y = 0
    # total_Z = 0
    for i in range(100):

        r_x = np.random.choice([-1, 0, 1])
        r_y = np.random.choice([-1, 0, 1])
        t_z = np.random.choice([-1, 0])

        print("r_x: ", r_x, " r_y: ", r_y, " t_z: ", t_z)
        test.moveArm(rotate_X=r_x, rotate_Y=r_y, translateZ=t_z)

        pos, q = test.compute_EE_pose()

        print(pos)
        time.sleep(0.2)

    test.moveArm(0,0,0)

    test.reset_world()

    # print("Total Rotation around X: ", total_X)
    # print("Total Rotation around Y: ", total_Y)
    # print("Total Translation along Z: ", total_Z)


    # test.moveArm(rotate_X=0, rotate_Y=0, translateZ=1)
    # time.sleep(5)
    # test.moveArm(rotate_Y=True)

