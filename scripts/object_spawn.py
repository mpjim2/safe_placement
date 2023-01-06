#!/usr/bin/env python3

import os
import numpy as np
from copy import deepcopy
import gym 
from gym.spaces import Box, Dict

from scipy.spatial.transform import Rotation as R


from franka_gripper.msg import GraspAction, GraspEpsilon, GraspGoal, MoveAction, MoveGoal

import rospy, rosgraph, roslaunch, rosservice
from geometry_msgs.msg import PoseStamped, PointStamped, QuaternionStamped
from franka_msgs.msg import FrankaState
from franka_msgs.srv import SetEEFrame
from tactile_msgs.msg import TactileState
from mujoco_ros_msgs.msg import GeomType, GeomProperties, BodyState, StepAction, StepGoal
from mujoco_ros_msgs.srv import SetGeomProperties, SetBodyState
import tf.transformations
import message_filters, actionlib
import rospkg 


def sample_obj_params(object_params=dict()):

    obj_types = object_params.get("types", ["box", "cylinder"])
    range_obj_radius = object_params.get("range_radius", np.array([0.04, 0.055])) # [m]
    range_obj_height = object_params.get("range_height", np.array([0.05/2, 0.1/2])) # [m]
    range_obj_wl = object_params.get("range_wl", np.array([0.05/2, 0.1/2])) # [m]; range width and length (requires object type "box")
    range_obj_mass = object_params.get("range_mass", np.array([0.01, 0.3])) # [kg]
    range_obj_sliding_fric = object_params.get("range_sliding_fric", np.array([0.3, 0.7])) # ignore torsional friction, because object cannot be a sphere

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
    # sample rotation about z-axis
    z_angle = np.random.uniform(low=-np.pi, high=np.pi)
    # sample x-pos
    x_pos = np.random.uniform(low=range_obj_x_pos[0], high=range_obj_x_pos[1])
    # sample y-pos 
    y_pos = np.random.uniform(low=range_obj_y_pos[0], high=range_obj_y_pos[1])

    pos = np.array([x_pos, y_pos, obj_height + 5])
    quat = tf.transformations.quaternion_from_euler(0,0,z_angle)

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

class ObjectPoseSpawn:

    def __init__(self):
       
        rospy.init_node("object_spawn_test_node")

        self.franka_state_sub = message_filters.Subscriber("franka_state_controller/franka_states", FrankaState)
        self.franka_state_cache = message_filters.Cache(self.franka_state_sub, cache_size=1, allow_headerless=False)
        self.franka_state_cache.registerCallback(self._new_msg_callback)

        self.current_msg = {"franka_state" : None}
        self.last_timestamps = {"franka_state" : 0}
        
        all_services_available = False
        required_srv = ["/franka_control/set_EE_frame", "/mujoco_server/set_geom_properties", "/mujoco_server/set_body_state"]
        while not all_services_available:
            service_list = rosservice.get_service_list()
            if not [srv for srv in required_srv if srv not in service_list]:
                all_services_available = True
                print("All Services Available!")

        self.set_geom_properties = rospy.ServiceProxy("/mujoco_server/set_geom_properties", SetGeomProperties)
        self.set_body_state = rospy.ServiceProxy("/mujoco_server/set_body_state", SetBodyState)

        self.actionClient = actionlib.SimpleActionClient("/franka_gripper/grasp", GraspAction)
        rospy.loginfo("Waiting for action server...")
        self.actionClient.wait_for_server()
        rospy.loginfo("Action server started!")


    def set_object_params(self):
        obj_geom_type_value,  obj_size_0, obj_size_1, obj_size_2, obj_sliding_fric, obj_mass = sample_obj_params()

        obj_geom_type = GeomType(value=obj_geom_type_value)
        obj_geom_properties = GeomProperties(env_id=0, name="object_geom", type=obj_geom_type, size_0=obj_size_0, size_1=obj_size_1, size_2=obj_size_2, friction_slide=obj_sliding_fric)
        resp = self.set_geom_properties(properties=obj_geom_properties, set_type=True, set_mass=False, set_friction=True, set_size=True)
        if not resp.success:
            rospy.logerr("SetGeomProperties:failed to set object parameters")
        
        ee_pos, _ = self.compute_EE_pose()

        _, obj_quat = sample_object_pose(obj_size_1)

        body_state = BodyState(env_id=0, name="object", pose=arr_to_pose_msg(ee_pos, obj_quat), mass=obj_mass)
        resp = self.set_body_state(state=body_state, set_pose=True, set_twist=False, set_mass=True, reset_qpos=False)
        if not resp.success:
            rospy.logerr("SetBodyState: failed to set object pose or object mass")

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

    def compute_EE_pose(self):

        transformationEE = self.current_msg['franka_state'].O_T_EE

        transformationEE = np.array(transformationEE).reshape((4,4), order='F')
        
        orientation = transformationEE[:3, :3]
        position    = transformationEE[:3, -1]
        
        orientation = R.from_matrix(orientation)

        orientation_q = orientation.as_quat()
        
        return position, orientation_q

    def open_gripper(self, width):

        pass

    def close_gripper(self, width=0.005, epsilon=0.0002, speed=0.1, force=1):
        
        epsilon = GraspEpsilon(inner=epsilon-0.000001, outer=epsilon+0.000001)
        goal_ = GraspGoal(width=width, epsilon=epsilon, speed=speed, force=force)
        
        #id_ = GoalID(stamp=rospy.Time.now(), id='grasp')
                
        #goal = GraspActionGoal(goal=goal_)

        self.actionClient.send_goal(goal_)
        self.actionClient.wait_for_result()

        success = self.actionClient.get_result()
        print(success)
        if not success:
            rospy.logerror("Grasp Action failed!")
        

if __name__=='__main__':

    test = ObjectPoseSpawn()

    test.set_object_params()    

    test.compute_EE_pose()
    test.close_gripper()