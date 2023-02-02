#!/usr/bin/env python3

import os 
import numpy as np 
from scipy.spatial.transform import Rotation as R
import pyquaternion as pq

import gym
from gym.spaces import Box, Discrete, Dict

import rospy, rosgraph, roslaunch, rosservice
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseStamped, PointStamped, QuaternionStamped, Twist, Vector3Stamped
from franka_msgs.msg import FrankaState
from franka_msgs.srv import SetEEFrame, SetLoad
from tactile_msgs.msg import TactileState
from franka_gripper.msg import GraspAction, GraspEpsilon, GraspGoal, MoveAction, MoveGoal

from mujoco_ros_msgs.msg import GeomType, GeomProperties, BodyState, ScalarStamped
from mujoco_ros_msgs.srv import SetGeomProperties, SetBodyState, SetPause
import tf.transformations
import message_filters, actionlib
import rospkg

import time


def rotate_vec(v, q):
    return q.rotate(v)

def quat_dot(q1, q2):
    return q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]

def to_msec(stamp):
    return int(stamp.to_nsec()*1e-6)
    
def nparray_to_posestamped(np_pos, np_quat, frame_id="world"):
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = frame_id
    pose.pose.position.x = np_pos[0]
    pose.pose.position.y = np_pos[1]
    pose.pose.position.z = np_pos[2]
    pose.pose.orientation.x = np_quat[0]
    pose.pose.orientation.y = np_quat[1]
    pose.pose.orientation.z = np_quat[2]
    pose.pose.orientation.w = np_quat[3]

    return pose

def quaternion_to_numpy(quaternion):
    return np.array([quaternion.w, quaternion.x, quaternion.y, quaternion.z])

def point_to_numpy(point):
    return np.array([point.x, point.y, point.z])

class TactileObjectPlacementEnv(gym.Env):
    metadata = {}

    def __init__(self, object_params=dict()):

        super().__init__()

        self.myrmex_max_val = 0.8

        self.franka_state_sub = message_filters.Subscriber("franka_state_controller/franka_states", FrankaState)
        self.franka_state_cache = message_filters.Cache(self.franka_state_sub, cache_size=1, allow_headerless=False)
        self.franka_state_cache.registerCallback(self._new_msg_callback)

        self.obj_quat_sub = message_filters.Subscriber("object_quat", QuaternionStamped)
        self.obj_quat_cache = message_filters.Cache(self.obj_quat_sub, cache_size=1, allow_headerless=False)
        self.obj_quat_cache.registerCallback(self._new_msg_callback)

        self.obj_pos_sub = message_filters.Subscriber("object_pos", PointStamped)
        self.obj_pos_cache = message_filters.Cache(self.obj_pos_sub, cache_size=1, allow_headerless=False)
        self.obj_pos_cache.registerCallback(self._new_msg_callback)

        self.tactile_left_sub = message_filters.Subscriber("/myrmex_l", TactileState)
        self.tactile_left_cache = message_filters.Cache(self.tactile_left_sub, cache_size=1, allow_headerless=False)
        self.tactile_left_cache.registerCallback(self._new_msg_callback)

        self.tactile_right_sub = message_filters.Subscriber("/myrmex_r", TactileState)
        self.tactile_right_cache = message_filters.Cache(self.tactile_right_sub, cache_size=1, allow_headerless=False)
        self.tactile_right_cache.registerCallback(self._new_msg_callback)

        self.current_msg = {"myrmex_l" : None, "myrmex_r" : None, "franka_state" : None, "object_pos" : None, "object_quat" : None}
        self.last_timestamps = {"myrmex_l" : 0, "myrmex_r" : 0, "franka_state" : 0, "object_pos" : 0, "object_quat" : 0}

        # self.current_msg = {"franka_state" : None, "object_pos" : None, "object_quat" : None}
        # self.last_timestamps = {"franka_state" : 0, "object_pos" : 0, "object_quat" : 0}

        self.action_space = Dict({
            "move_down" : Discrete(2),
            "rotate_X" : Discrete(3),
            "rotate_Y" : Discrete(3),
            "open_gripper" : Discrete(2)
        })

        self.observation_space = Dict({
            "observation" : Dict({
                "ee_pose" : Box( low=np.array([-np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi]), 
                                 high=np.array([np.inf, np.inf, np.inf, np.pi, np.pi, np.pi]),
                                 dtype=np.float64),
                
                "joint_positions" : Box(low=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]), 
                                       high=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
                                       dtype=np.float64),
                
                "joint_torques" : Box(low=np.array([-87, -87, -87, -87, -12, -12, -12]),
                                      high=np.array([87,  87,  87,  87,  12,  12,  12]),
                                      dtype=np.float64),
                
                "joint_velocities" : Box(low=np.array([-2.175, -2.175, -2.175, -2.175, -2.61, -2.61, -2.61]),
                                         high=np.array([  2.175,  2.175,  2.175,  2.175,  2.61,  2.61,  2.61]),
                                         dtype=np.float64),
                
                "myrmex_l" : Box(low=0, high=1, shape=(16*16,), dtype=np.float64),
                
                "myrmex_r" : Box(low=0, high=1, shape=(16*16,), dtype=np.float64),

                "time_diff": Box(low=0, high=np.inf, shape=(len(self.current_msg),), dtype=np.int64)
            })
        })


        if not rosgraph.is_master_online(): # check if ros master is already running
            print("starting launch file...")
            uuid = roslaunch.rlutil.get_or_generate_uuid(options_runid=None, options_wait_for_master=False)
            roslaunch.configure_logging(uuid)
            pkg_path = rospkg.RosPack().get_path("safe_placement")
            self.launch = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_files=[os.path.join(pkg_path, "launch","test.launch")], is_core=True)
            self.launch.start()

        rospy.init_node("SafePlacementEnvNode")

        all_services_available = False
        required_srv = ["/franka_control/set_EE_frame", 
                        "/franka_control/set_load",
                        "/mujoco_server/set_geom_properties", 
                        "/mujoco_server/set_body_state", 
                        "/mujoco_server/reset",
                        "/mujoco_server/set_pause"]

        while not all_services_available:
            service_list = rosservice.get_service_list()
            if not [srv for srv in required_srv if srv not in service_list]:
                all_services_available = True
                print("All Services Available!")

        self.set_geom_properties = rospy.ServiceProxy("/mujoco_server/set_geom_properties", SetGeomProperties)
        self.set_body_state      = rospy.ServiceProxy("/mujoco_server/set_body_state", SetBodyState)
        self.reset_world         = rospy.ServiceProxy("/mujoco_server/reset", Empty)
        self.pause_sim           = rospy.ServiceProxy("/mujoco_server/set_pause", SetPause)

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

        
        self.obj_height = 0
        self.obj_types = object_params.get("types", ["box"])
        self.range_obj_radius = object_params.get("range_radius", np.array([0.04, 0.055])) # [m]
        self.range_obj_height = object_params.get("range_height", np.array([0.1/2, 0.1/2])) # [m]
        self.range_obj_l = object_params.get("range_l", np.array([0.04/2, 0.07/2])) # [m]; range width and length (requires object type "box")
        self.range_obj_w = object_params.get("range_w", np.array([0.04/2, 0.15  /2])) # [m]; range width and length (requires object type "box")
        self.range_obj_mass = object_params.get("range_mass", np.array([0.01, 0.1])) # [kg]

        self.range_obj_x_pos = object_params.get("range_x_pos", np.array([0.275,0.525])) #[m]
        self.range_obj_y_pos = object_params.get("range_y_pos", np.array([-0.2,0.2])) #[m]

        self.sensor_thickness = 0.003


    def _sample_obj_params(self):
        # sample new object/target type
        idx_type = np.random.randint(low=0, high=len(self.obj_types))
        self.obj_type = self.obj_types[idx_type]

        # sample new object/target mass
        self.obj_mass = np.random.uniform(low=self.range_obj_mass[0], high=self.range_obj_mass[1])

        # sample new object/target size depending on object type
        if self.obj_type == "box":
            self.obj_geom_type_value = 6
            # sample object width/2, length/2 and height/2 
            self.obj_size_0 = np.random.uniform(low=self.range_obj_w[0], high=self.range_obj_w[1]) # width/2
            self.obj_size_1 = np.random.uniform(low=self.range_obj_l[0], high=self.range_obj_l[1]) # length/2
            self.obj_size_2 = np.random.uniform(low=self.range_obj_height[0], high=self.range_obj_height[1] ) # height/2
            self.obj_height = self.obj_size_2
        else:
            # self.obj_type == "cylinder"
            self.obj_geom_type_value = 5
            # sample object height/2 and radius
            self.obj_size_0 = np.random.uniform(low=self.range_obj_radius[0], high=self.range_obj_radius[1]) # radius
            self.obj_size_1 = np.random.uniform(low=self.range_obj_height[0], high=self.range_obj_height[1]) # height/2
            self.obj_size_2 = 0
            self.obj_height = self.obj_size_1

    def _sample_object_pose(self):

        transformationEE = self.current_msg['franka_state'].O_T_EE
        transformationEE = np.array(transformationEE).reshape((4,4), order='F')

        position    = transformationEE[:3, -1]

        #self.obj_size_0, self.obj_size_1, self.obj_size_2

        z_shift = np.random.uniform(low=-0.035, high=0.03)
        pos_z = position[2] - self.obj_height + z_shift

        x_shift = np.random.uniform(low=-self.obj_size_0, high=self.obj_size_0)
        position[0] += x_shift
        # x_shift = np.random.uniform(low=-self.obj_size_0/3, high=self.obj_size_0/3)
        # position[0] += x_shift  
        position[2] = pos_z

        return position

    def _pose_quat_from_trafo(self, transformationEE):
        
        transformationEE = np.array(transformationEE).reshape((4,4), order='F')
        
        orientation = transformationEE[:3, :3]
        position    = transformationEE[:3, -1]
        
        orientation = R.from_matrix(orientation)

        orientation_q = orientation.as_quat()
        orientation_xyz = orientation.as_euler('xyz')
        
        return position, orientation_q, orientation_xyz

    def _get_obs(self):
        
        current_obs = self.current_msg['franka_state']
        
        ee_pos, ee_quat, ee_xyz = self._pose_quat_from_trafo(current_obs.O_T_EE)
        pose = np.array((ee_pos, ee_xyz)).reshape(6)

        # ee_obj_dist = np.linalg.norm(ee_pos - obj_pos)

        joint_positions = np.array(current_obs.q, dtype=np.float64)
        joint_torques = np.array(current_obs.tau_J, dtype=np.float64)
        joint_velocities = np.array(current_obs.dq, dtype=np.float64)
        
        myrmex_data_noise_l = np.array(self.current_msg["myrmex_l"].sensors[0].values)/self.myrmex_max_val + 0.000001*np.random.randn(16*16)
        myrmex_data_l = np.clip(myrmex_data_noise_l, a_min=0, a_max=1)

        myrmex_data_noise_r = np.array(self.current_msg["myrmex_r"].sensors[0].values)/self.myrmex_max_val + 0.000001*np.random.randn(16*16)
        myrmex_data_r = np.clip(myrmex_data_noise_r, a_min=0, a_max=1)

        time_diff = np.array([to_msec(self.current_msg[k].header.stamp) - to_msec(self.last_timestamps[k]) for k in self.current_msg.keys()]) # milliseconds

        observation = {"ee_pose" : pose, 
                       "joint_positions" : joint_positions,
                       "joint_torques" : joint_torques,
                       "joint_velocities" : joint_velocities,
                       "myrmex_l" : myrmex_data_l,
                       "myrmex_r" : myrmex_data_r,
                       "time_diff" : time_diff
                       }


        return {"observation" : observation}

    def _compute_twist(self, rotate_X, rotate_Y, translate_Z):

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
        if translate_Z > 0:
            pos[2] -= 0.005
        elif translate_Z < 0:
            pos[2] += 0.005

        twist_msg = Twist()

        twist_msg.linear.x = pos[0]
        twist_msg.linear.y = pos[1]
        twist_msg.linear.z = pos[2]

        twist_msg.angular.x = xyz[0]
        twist_msg.angular.y = xyz[1]
        twist_msg.angular.z = xyz[2]

        return twist_msg

    def _open_gripper(self, width=0.08):
        goal = MoveGoal(width=width, speed=0.05)

        self.release_client.send_goal(goal)
        self.release_client.wait_for_result()

        success = self.release_client.get_result()
        if not success:
            rospy.logerr("Open Action failed!")

        return success

    def _close_gripper(self, width, eps, speed, force):
        
        epsilon = GraspEpsilon(inner=eps, outer=eps)
        goal_ = GraspGoal(width=width, epsilon=epsilon, speed=speed, force=force)
        
        self.grasp_client.send_goal(goal_)
        self.grasp_client.wait_for_result()

        success = self.grasp_client.get_result()

        if not success:
            rospy.logerr("Grasp Action failed!")

        return success

    def _setLoad(self, mass, load_inertia): 
        
        obj_pos = point_to_numpy(self.current_msg['object_pos'].point)
        
        transformationEE = self.current_msg['franka_state'].O_T_EE
        transformationEE = np.array(transformationEE).reshape((4,4), order='F')
        ee_pos           = transformationEE[:3, -1]
        
        #subtract EE to flange transformation
        ee_pos -= np.array([0, 0, 0.17])

        F_x_center_load = list(ee_pos - obj_pos)
        #Fingertip params
        # mass = 0.078*2 #mass of two ubi fingertips
        # F_x_center_load = [0,0,0.17]
        # load_inertia = [0, 0, 0.000651, 0, 0, 0.000651, 0, 0, 0.000896]

        #{0,0,0.17} translation from flange to in between fingers; default for f_x_center
        response = self.set_load(mass=mass, F_x_center_load=F_x_center_load, load_inertia=load_inertia)
        
        return response.success

    def _compute_reward(self):
        
        #object Y-Value in World Frame
        obj_h = point_to_numpy(self.current_msg['object_pos'].point)[1]        

        #object Y-Axis in World frame
        quat = quaternion_to_numpy(self.current_msg['object_quat'].quaternion)
        quat = pq.Quaternion(quat)

        y_axis = rotate_vec(np.array([0, 1, 0]), quat)

        # dot product of world Y and object Y axis
        uprightnes = np.dot(np.array([0, 1, 0]), y_axis)

        # distance of object to floor
        floor_closeness = self.obj_height/2 - obj_h
        if uprightnes == 0:
            return -1
        else:
            return uprightnes - floor_closeness
        #Reward = 1 if object upright
        #Reward = -1 if no contact between ground and object at release time
        #Reward = 0.5 * difference from upright position at moment of release
        #if object tips over
        

    def step(self, action):
        
        done = False
        if action['open_gripper'] == 1:
            #Stop movement of arm
            stop_ = self._compute_twist(0, 0, 0)
            self.twist_pub.publish(stop_)
            
            observation = self._get_obs()
            
            self._open_gripper()
            done = True

            reward = self._compute_reward()
        
        else:
            
            reward = 0

            twist = self._compute_twist(action['rotate_X']  -1, 
                                        action['rotate_Y']  -1, 
                                        action['move_down'] -1)

            self.twist_pub.publish(twist)
            
            observation = self._get_obs()
        
        return observation, reward, done, False, {'info' : None}

    def _initial_grasp(self):
        
        self._open_gripper()
        # self.pause_sim(paused=True)
        
        self.set_object_params()

        grasp_success = self._close_gripper(width=self.obj_size_1*2, force=5.0, eps=0.005, speed=0.01)
        
        if grasp_success:
            self._setLoad(mass=self.obj_mass, load_inertia=list(np.eye(3).flatten()))
            
            #move upward to pick object up
            twist = self._compute_twist(0, 0, 1)
            self.twist_pub.publish(twist)
            
            time.sleep(1)
            twist = self._compute_twist(0, 0, 0)
            self.twist_pub.publish(twist)

            scaf_geom_type       = GeomType(value=6)
            scaf_geom_properties = GeomProperties(env_id=0, name="scaffold_geom", type=scaf_geom_type, size_0=0, size_1=0, size_2=0, friction_slide=1, friction_spin=0.005, friction_roll=0.0001) 
            resp = self.set_geom_properties(properties=scaf_geom_properties, set_type=True, set_mass=False, set_friction=True, set_size=True)
            if not resp.success:
                rospy.logerr("SetGeomProperties:failed to set scaffold parameters")

        return grasp_success

    def set_object_params(self):
        # sample object type, mass, sliding friction, height, radius, length, width
        self._sample_obj_params()        

        # sample object start pose
        obj_pos = self._sample_object_pose()

        #compute and spawn scaffold
        scaf_size_2 = obj_pos[-1] - self.obj_size_2

        scaf_geom_type       = GeomType(value=6)
        scaf_geom_properties = GeomProperties(env_id=0, name="scaffold_geom", type=scaf_geom_type, size_0=0.2, size_1=0.3, size_2=scaf_size_2, friction_slide=1, friction_spin=0.005, friction_roll=0.0001) 
        resp = self.set_geom_properties(properties=scaf_geom_properties, set_type=True, set_mass=False, set_friction=True, set_size=True)
        if not resp.success:
            rospy.logerr("SetGeomProperties:failed to set scaffold parameters")

        # set object properties
        obj_geom_type = GeomType(value=self.obj_geom_type_value)
        obj_geom_properties = GeomProperties(env_id=0, name="object_geom", type=obj_geom_type, size_0=self.obj_size_0, size_1=self.obj_size_1, size_2=self.obj_size_2, friction_slide=1, friction_spin=0.005, friction_roll=0.0001)
        resp = self.set_geom_properties(properties=obj_geom_properties, set_type=True, set_mass=False, set_friction=True, set_size=True)
        if not resp.success:
            rospy.logerr("SetGeomProperties:failed to set object parameters")

        # set object pose with mass=0
        body_state = BodyState(env_id=0, name="object", pose=nparray_to_posestamped(obj_pos, np.array([1, 0, 0, 0])), mass=self.obj_mass)
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
          
    def reset(self, seed=None):

        super().reset(seed=seed)

        success = False

        while not success:

            self._setLoad(mass=0, load_inertia=list(np.eye(3).flatten()))
            twist = self._compute_twist(0, 0, 0)
            self.twist_pub.publish(twist)

            self.reset_world()

            self._open_gripper()

            success = self._initial_grasp()
            time.sleep(0.1)

        observation = self._get_obs() 
    
        info = {'X' : None}
        return observation, info 
    

    def close(self):
        if "self.launch" in locals():
            self.launch.stop()

