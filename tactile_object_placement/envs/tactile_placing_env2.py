class TactileObjectPlacementEnv_v2(gym.Env):
    metadata = {}

    def __init__(self, object_params=dict(), curriculum=False, sensor="fingertip", continuous=False, grid_size=4, action_space='full', timesteps=10):

        super().__init__()
        
        self.timesteps = timesteps

        pkg_path = rospkg.RosPack().get_path("safe_placement")

        self.franka_state_sub = message_filters.Subscriber("franka_state_controller/franka_states", FrankaState)
        self.franka_state_cache = message_filters.Cache(self.franka_state_sub, cache_size=1, allow_headerless=False)
        self.franka_state_cache.registerCallback(self._franka_state_callback)

        self.obj_quat_sub = message_filters.Subscriber("object_quat", QuaternionStamped)
        self.obj_quat_cache = message_filters.Cache(self.obj_quat_sub, cache_size=1, allow_headerless=False)
        self.obj_quat_cache.registerCallback(self._new_msg_callback)

        self.obj_contact_sub = message_filters.Subscriber("object_contact_GT", ScalarStamped)
        self.obj_contact_cache = message_filters.Cache(self.obj_contact_sub, cache_size=1, allow_headerless=False)
        self.obj_contact_cache.registerCallback(self._new_msg_callback)

        self.obj_pos_sub = message_filters.Subscriber("object_pos", PointStamped)
        self.obj_pos_cache = message_filters.Cache(self.obj_pos_sub, cache_size=1, allow_headerless=False)
        self.obj_pos_cache.registerCallback(self._new_msg_callback)

        self.tactile_left_sub = message_filters.Subscriber("/myrmex_l", TactileState)
        self.tactile_left_cache = message_filters.Cache(self.tactile_left_sub, cache_size=1, allow_headerless=False)
        self.tactile_left_cache.registerCallback(self._new_msg_callback)

        self.tactile_right_sub = message_filters.Subscriber("/myrmex_r", TactileState)
        self.tactile_right_cache = message_filters.Cache(self.tactile_right_sub, cache_size=1, allow_headerless=False)
        self.tactile_right_cache.registerCallback(self._new_msg_callback)
        
        self.current_msg = {"myrmex_l" : deque([np.zeros(4) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                            "myrmex_r" : deque([np.zeros(4) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                            "joint_positions" : deque([np.zeros(7) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                            "joint_velocities" : deque([np.zeros(7) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                            "joint_torques" : deque([np.zeros(7) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                            "ee_pose" : deque([np.zeros(7) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                            "object_pos" : None, 
                            "object_quat" : None, 
                            "object_contact_GT" : None}
        
        self.last_timestamps = {"myrmex_l" : None, "myrmex_r" : None, "franka_state" : None, "object_pos" : None, "object_quat" : None, "object_contact_GT" : None}
        


        self.record = False

        print("All Subs Defined")
        # self.current_msg = {"franka_state" : None, "object_pos" : None, "object_quat" : None}
        # self.last_timestamps = {"franka_state" : 0, "object_pos" : 0, "object_quat" : 0}
        
        self.sensor = sensor
        
        self.max_episode_steps = 300

        self.DISCRETE_ACTIONS = []

        self.action_space_mode = action_space
        if action_space =='full':
            for x in [-1, 1]:
                self.DISCRETE_ACTIONS.append([0,x,0,0])
            for z in [1, -1]:
                self.DISCRETE_ACTIONS.append([z,0,0,0])
            self.DISCRETE_ACTIONS.append([0, 0, 0, 1])
            self.DISCRETE_ACTIONS.append([0, 0, 0, 0])
        else:
            self.DISCRETE_ACTIONS.append([-1,0,0,0])
            self.DISCRETE_ACTIONS.append([0, 0, 0, 1])
            # self.DISCRETE_ACTIONS.append([0, 0, 0, 0])

        # for x in [-1, 0, 1]:
        #     for z in [-1, 0, 1]:
        #         self.DISCRETE_ACTIONS.append([z,x,0,0])
                
        
        self.action_space = Discrete(len(self.DISCRETE_ACTIONS))


        if sensor == 'fingertip': 
            self.myrmex_max_val = 0.1
            tactile_obs = Box(low=0, high=1, shape=(20,), dtype=np.float64)        
            launch_file = "panda_fingertip.launch"    
            self.num_taxels = 32

        elif sensor == 'plate':
            self.myrmex_max_val = 1.3
            tactile_obs = Box(low=0, high=1, shape=(grid_size*grid_size,), dtype=np.float64)
            launch_file = "panda.launch"
            self.num_taxels = grid_size*grid_size

        self.observation_space = Dict({
            "observation" : Dict({
                "ee_pose" : Sequence(Box( low=np.array([-0.5, -0.5, 0, -1, -1, -1, -1]), 
                                 high=np.array([0.5, 0.5, 0.6, 1, 1, 1, 1]),
                                 dtype=np.float64)),
                
                "joint_positions" : Sequence(Box(low=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]), 
                                       high=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
                                       dtype=np.float64)),
                
                "joint_torques" : Sequence(Box(low=np.array([-87, -87, -87, -87, -12, -12, -12]),
                                      high=np.array([87,  87,  87,  87,  12,  12,  12]),
                                      dtype=np.float64)),
                
                "joint_velocities" : Sequence(Box(low=np.array([-2.175, -2.175, -2.175, -2.175, -2.61, -2.61, -2.61]),
                                         high=np.array([  2.175,  2.175,  2.175,  2.175,  2.61,  2.61,  2.61]),
                                         dtype=np.float64)),
                

                "myrmex_l" : Sequence(tactile_obs),
                
                "myrmex_r" : Sequence(tactile_obs)

                # "contact" : Box(low=np.array([0]),
                #                 high=np.array([1]),
                #                 dtype=np.uint8)

                # "time_diff": Box(low=0, high=np.inf, shape=(len(self.current_msg),), dtype=np.int64)
            })
        })

        if not rosgraph.is_master_online(): # check if ros master is already running
            print("starting launch file...")

            uuid = roslaunch.rlutil.get_or_generate_uuid(options_runid=None, options_wait_for_master=False)
            roslaunch.configure_logging(uuid)
            pkg_path = rospkg.RosPack().get_path("safe_placement")
            self.launch = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_files=[os.path.join(pkg_path, "launch", launch_file)], is_core=True)
            self.launch.start()

        rospy.init_node("SafePlacementEnvNode")

        all_services_available = False
        required_srv = ["/franka_control/set_EE_frame", 
                        "/franka_control/set_load",
                        "/mujoco_server/set_geom_properties", 
                        "/mujoco_server/set_body_state", 
                        "/mujoco_server/reset",
                        "/mujoco_server/set_pause",
                        "/mujoco_server/set_gravity",
                        "/mujoco_server/get_gravity"]

        while not all_services_available:
            service_list = rosservice.get_service_list()
            if not [srv for srv in required_srv if srv not in service_list]:
                all_services_available = True
                print("All Services Available!")

        self.set_geom_properties = rospy.ServiceProxy("/mujoco_server/set_geom_properties", SetGeomProperties)
        self.set_body_state      = rospy.ServiceProxy("/mujoco_server/set_body_state", SetBodyState)
        self.reset_world         = rospy.ServiceProxy("/mujoco_server/reset", Empty)
        self.pause_sim           = rospy.ServiceProxy("/mujoco_server/set_pause", SetPause)
        self.set_gravity         = rospy.ServiceProxy("mujoco_server/set_gravity", SetGravity)
        self.get_gravity         = rospy.ServiceProxy("mujoco_server/get_gravity", GetGravity)
        self.set_load            = rospy.ServiceProxy("/franka_control/set_load", SetLoad)
        
        #Action Client for Simulation Step

        self.continuous = continuous
        if not continuous:
            self.action_client = actionlib.SimpleActionClient("/mujoco_server/step", StepAction)
            rospy.loginfo("Waiting for action server...")
            self.action_client.wait_for_server()
            rospy.loginfo("Action server started")
            self._perform_sim_steps(num_sim_steps=100)
            rospy.loginfo("Initial simulation steps finished")
        
        # resp = self.pause_sim(paused=False)
     
        self.twist_pub = rospy.Publisher('/cartesian_impedance_controller/twist_cmd', Twist, queue_size=1)

        #client for MoveAction
        self.release_client = actionlib.SimpleActionClient("/franka_gripper/move", MoveAction)
        rospy.loginfo("Waiting for MoveAction action server...")
        self.release_client.wait_for_server()

        #client for grasp action (only needed when resetting the environment)
        self.grasp_client = actionlib.SimpleActionClient("/franka_gripper/grasp", GraspAction)
        rospy.loginfo("Waiting for GraspAction action server...")
        self.grasp_client.wait_for_server()

        self.stop_client = actionlib.SimpleActionClient("/franka_gripper/grasp", StopAction)
        rospy.loginfo("Waiting for GraspAction action server...")
        self.stop_client.wait_for_server()
        rospy.loginfo("Action server started!")

      
        self.obj_height = 0
        self.obj_types = object_params.get("types", ["box"])
        self.range_obj_radius = object_params.get("range_radius", np.array([0.04, 0.055])) # [m]
        self.range_obj_height = object_params.get("range_height", np.array([0.1/2, 0.1/2])) # [m]
        self.range_obj_l = object_params.get("range_l", np.array([0.04/2, 0.06/2])) # [m]; range width and length (requires object type "box")
        self.range_obj_w = object_params.get("range_w", np.array([0.04/2, 0.06/2])) # [m]; range width and length (requires object type "box")
        self.range_obj_mass = object_params.get("range_mass", np.array([0.1, 0.1])) # [kg]

        self.range_obj_x_pos = object_params.get("range_x_pos", np.array([0.275,0.525])) #[m]
        self.range_obj_y_pos = object_params.get("range_y_pos", np.array([-0.2,0.2])) #[m]

        self.sensor_thickness = 0.005

        
        self.init_quat = None
        self.min_gapsize = 0.002
        self.table_height = 0
        self.curriculum = curriculum
        self.max_timesteps = 1000
        self.angle_range = 0.17


        self.noop_counter = 0
        self.contact_counter = 0
        self.last_command = None
        while True:
            self._perform_sim_steps(5)
            if None not in self.current_msg.values():
                break
        
        #Fingertips are located at the franka gripper fingers; no transformation of EE Frame necessary
        if sensor == 'plate':
            self._set_EE_frame_to_gripcenter()

    # def _update_space_limits(self):
        
    #     self.observation_space["observation"]["ee_pose"] = Box( low=np.array([-np.inf, -np.inf, self.table_height*2 + 0.022, -np.pi, -np.pi/2, -np.pi]), 
    #                                                             high=np.array([np.inf, np.inf, 0.5, np.pi, np.pi/2, np.pi]),
    #                                                             dtype=np.float64)

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
            
            self.obj_size_2 = np.random.uniform(low=self.range_obj_height[0], high=self.range_obj_height[1] ) # height/2

            self.obj_size_0 = np.random.uniform(low=self.range_obj_w[0], high=self.range_obj_w[1]) # width/2
            self.obj_size_1 = np.random.uniform(low=self.range_obj_l[0], high=self.range_obj_l[1]) # length/2
            
            # self.obj_size_2 = self.range_obj_height[1]

            # self.obj_size_0 = self.range_obj_w[1]
            # self.obj_size_1 = self.range_obj_l[1]
            

            self.obj_height = self.obj_size_2
        else:
            # self.obj_type == "cylinder"
            self.obj_geom_type_value = 5
            # sample object height/2 and radius
            self.obj_size_0 = np.random.uniform(low=self.range_obj_radius[0], high=self.range_obj_radius[1]) # radius
            self.obj_size_1 = np.random.uniform(low=self.range_obj_height[0], high=self.range_obj_height[1]) # height/2
            self.obj_size_2 = 0
            self.obj_height = self.obj_size_1

    def _sample_object_pose(self, testing=False):

        # current_obs = self.current_msg['franka_state']        
        # pos, quat, _ = self._pose_quat_from_trafo(current_obs.F_T_EE)

        position = self.current_msg['ee_pose'][-1][:3]


        if testing:
            y_angle = np.random.choice([-self.angle_range, self.angle_range])
        else:
            y_angle = np.random.uniform(low=-self.angle_range, high=self.angle_range)

        quat = tf.transformations.quaternion_from_euler(0,y_angle,0)

        py_quat = pq.Quaternion(quat)

        obj_axis = np.array(rotate_vec(v=[0,0,1], q=py_quat))
            
        obj_axis_n = (obj_axis / np.linalg.norm(obj_axis)) * self.obj_height/2
        
        if self.sensor == 'fingertip':
            
            # 1. compute corner points relative to bject center 
            corners = compute_corner_coords(np.zeros(3), self.obj_size_0, self.obj_size_1, self.obj_size_2, quat)
            
            # 2. find maximum of corner Z coordinates
            max_z = np.max(corners[:, -1])

            # 3. Compute minimum shift along Object axis that avoids contact of hand & object
            # 3.1 Compute shift along z
            z_shift = max_z - 0.05
            
            if z_shift >= -0.005:
                # 3.2 compute shift along object axis that shifts z by that amount
                if y_angle == 0:
                    min_shift = 0.03
                else:
                    projected = np.array([obj_axis_n[0], obj_axis_n[1], 0])
                
                    magnitude = np.linalg.norm(projected)
                    
                    theta = math.atan2(projected[1], projected[0])
                    hrztl_shift = z_shift * math.cos(theta)
                    
                    min_shift = abs(hrztl_shift * (np.linalg.norm(obj_axis_n) / magnitude))
             
            else: 
                min_shift = 0
        else:

            m_diag = math.sqrt((0.04**2) * 2)/2

            diff = m_diag - np.linalg.norm(obj_axis_n) 
            if diff >= 0:
                min_shift = diff + 0.005
            else:
                min_shift = 0

        ax_shift = np.random.uniform(low=min_shift, high=0.5*self.obj_height)

        position -= (obj_axis * ax_shift)

        return position, quat, y_angle

    def _pose_quat_from_trafo(self, transformationEE):
        
        transformationEE = np.array(transformationEE).reshape((4,4), order='F')
        
        orientation = transformationEE[:3, :3]
        position    = transformationEE[:3, -1]
        
        orientation = R.from_matrix(orientation)

        orientation_q = orientation.as_quat()
        orientation_xyz = orientation.as_euler('xyz')
        
        return position, orientation_q, orientation_xyz

    def _set_EE_frame_to_gripcenter(self): 

        current_obs = self.current_msg['franka_state']        
        gripper_pos, gripper_quat, _ = self._pose_quat_from_trafo(current_obs.F_T_EE)

        gripper_pos[-1] -= 0.076     
        ee_trafo = tf.transformations.quaternion_matrix(np.array([1,0,0,0]))
        ee_trafo[0:3,-1] = gripper_pos

        set_EE_frame = rospy.ServiceProxy("/franka_control/set_EE_frame", SetEEFrame)
        resp = set_EE_frame(NE_T_EE=tuple((ee_trafo.T).ravel()))  # column-major format
        if not resp.success:
            rospy.logerr("SetEEFrame: failed to set fixed EE trafo")

        # wait for a new franka_state msg
        self._perform_sim_steps(100)
        self.current_msg = {"myrmex_l" : deque([np.zeros(4) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                            "myrmex_r" : deque([np.zeros(4) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                            "joint_positions" : deque([np.zeros(7) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                            "joint_velocities" : deque([np.zeros(7) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                            "joint_torques" : deque([np.zeros(7) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                            "ee_pose" : deque([np.zeros(7) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                            "object_pos" : None, 
                            "object_quat" : None, 
                            "object_contact_GT" : None}
        
        self.last_timestamps = {"myrmex_l" : None, "myrmex_r" : None, "franka_state" : None, "object_pos" : None, "object_quat" : None, "object_contact_GT" : None}
        


        while True:
            self._perform_sim_steps(1)
            if None not in self.current_msg.values():
                break
            
    def _get_obs(self, return_quat=False, initial_obs=False):
        

        observation = {
                    "ee_pose" : np.array(self.current_msg['ee_pose']), 
                    "joint_positions" : np.array(self.current_msg['joint_positions']),
                    "joint_torques" : np.array(self.current_msg['joint_torques']),
                    "joint_velocities" : np.array(self.current_msg['joint_velocities']),
                    "myrmex_l" : np.array(self.current_msg['myrmex_l']),
                    "myrmex_r" : np.array(self.current_msg['myrmex_r'])
                    # "contact" : self.current_msg['ee_pose']
                    #    "time_diff" : time_diff
                    }

            
        if not return_quat:
            return {"observation" : observation}
        else:
            return {"observation" : observation}
        
    def _compute_twist(self, rotate_X, rotate_Y, translate_Z):

        quat = pq.Quaternion()

        noop = True
        current_pose = self._get_obs()
        current_pose = current_pose["observation"]["ee_pose"]
        
        cmd = np.array([0, 0])
        if rotate_X > 0:
            quat = quat * pq.Quaternion(axis=[1, 0, 0], angle=0.15)
            cmd[0] += 1
            noop = False
        elif rotate_X < 0:
            quat = quat * pq.Quaternion(axis=[1, 0, 0], angle=-0.15)
            cmd[0] += 1
            noop = False

    # if euler_diff[1] < math.pi/2:
        if rotate_Y > 0:
            quat = quat * pq.Quaternion(axis=[0, 1, 0], angle=0.15)
            noop = False
        elif rotate_Y < 0:
            quat = quat * pq.Quaternion(axis=[0, 1, 0], angle=-0.15)
            noop = False
        
        xyz = quat.vector 

        pos = np.zeros(3)
        if translate_Z < 0:  
            # if current_pose[2] > self.table_height*2 + 0.022:      
            pos[2] += 0.02
            cmd[1] += 1
            noop = False

        elif translate_Z > 0:
            if current_pose[2] < 0.5:
                pos[2] -= 0.02
                cmd[1] -= 1
                noop = False
            else:
                noop = True
                
        twist_msg = Twist()

        twist_msg.linear.x = pos[0]
        twist_msg.linear.y = pos[1]
        twist_msg.linear.z = pos[2]

        twist_msg.angular.x = xyz[0]
        twist_msg.angular.y = xyz[1]
        twist_msg.angular.z = xyz[2]
        smooth_reward = 0
        
        if not self.last_command is None:
            diff = self.last_command - cmd
            if max(diff) == 2:
                smooth_reward = -0.0001

        self.last_command = cmd
        return twist_msg, smooth_reward, noop

    def _check_object_grip(self, obj_pos=None):
        

        if obj_pos is None:
            obj_pos = point_to_numpy(self.current_msg['object_pos'].point)   

        # if self.last_timestamps['object_pos'] - self.last_timestamps['franka_state'] < :

       
        ee_pos = self.current_msg['ee_pose'][-1][:3]
        check = np.linalg.norm(ee_pos - obj_pos) <= self.obj_height/2 + 0.02 

        return check

    def _open_gripper(self, width=0.08):

        self.release_client.cancel_all_goals()
        goal = MoveGoal(width=width, speed=0.03)

        self.release_client.send_goal(goal)
     
        if self.continuous:
            self.release_client.wait_for_result()
            success = self.release_client.get_result()
            
        else:

            success = None
            while success is None:
                self._perform_sim_steps(10)
                success = self.release_client.get_result()
        
        if not success:
            rospy.logerr("Open Action failed!")
        return success

    def _close_gripper(self, width=0.01, eps=0.08, speed=0.03, force=3):
        
        self.grasp_client.cancel_all_goals()
        epsilon = GraspEpsilon(inner=eps, outer=eps)
        goal = GraspGoal(width=width, epsilon=epsilon, speed=speed, force=force)
        
        self.grasp_client.send_goal(goal)

        # self.pause_sim(paused=False)
        if self.continuous:
            self.grasp_client.wait_for_result()
            success = self.grasp_client.get_result()
            
        else:

            success = None
            while success is None: 
                self._perform_sim_steps(10)
                success = self.grasp_client.get_result()
        # self.pause_sim(paused=True)
        if not success:
            rospy.logerr("Grasp Action failed!")
       
        return success

    def _setLoad(self, mass, load_inertia): 
        
        # transformationEE = self.current_msg['franka_state'].O_T_EE
        # transformationEE = np.array(transformationEE).reshape((4,4), order='F')
        
        
        ee_pos           = self.current_msg['ee_pose'][-1][:3]
        
        #subtract EE to flange transformation
        ee_pos -= np.array([0, 0, 0.17])
        if mass != 0:
            obj_pos = point_to_numpy(self.current_msg['object_pos'].point)
        
            F_x_center_load = list(ee_pos - obj_pos)
        else:
            F_x_center_load = list(ee_pos)
        #Fingertip params
        # mass = 0.078*2 #mass of two ubi fingertips
        # F_x_center_load = [0,0,0.17]
        # load_inertia = [0, 0, 0.000651, 0, 0, 0.000651, 0, 0, 0.000896]
    
        #{0,0,0.17} translation from flange to in between fingers; default for f_x_center
        response = None

        while response is None:
            response = self.set_load(mass=mass, F_x_center_load=F_x_center_load, load_inertia=load_inertia)
            self._perform_sim_steps(1)

        return response.success

    def _compute_reward(self, compute_final_reward=False):
        #object Y-Value in World Frame
        reward = -0.01
      
        info = {'cause' : 0}
        obj_pos = point_to_numpy(self.current_msg['object_pos'].point)
        #check if object is still in hand
        if self._check_object_grip(obj_pos) == False:
            reward = -1
            info['cause'] = 'lostObject'
        
        if self.max_episode_steps == 0:
            info['cause'] = 'timeout'
            if self.contact_counter >= 1:
                reward = 0.25
                
            else:
                reward = -1
                
        else:

            if compute_final_reward or self.reward_fn=='close_gap':
                
                contact = self.current_msg['object_contact_GT'].value
                 
                if contact > 0:
                    if self.reward_fn=='close_gap':
                        self.contact_counter += 1
                        if self.contact_counter >= 5:
                            info['cause'] = 'Contact'
                            compute_final_reward = True

                    if compute_final_reward:

                        quat = quaternion_to_numpy(self.current_msg['object_quat'].quaternion)
                        quat = pq.Quaternion(quat)
                        z_axis = rotate_vec(np.array([0, 0, 1]), quat)
                                
                        reward = np.dot(np.array([0, 0, 1]), z_axis)
                    
                else:
                    if self.reward_fn == 'close_gap':
                        self.contact_counter = 0
                    if compute_final_reward:
                        reward = -1

        return reward, info    
    
    def _perform_sim_steps(self, num_sim_steps):
        goal = StepGoal(num_steps=num_sim_steps)
        self.action_client.send_goal(goal)
        self.action_client.wait_for_result()
        success = self.action_client.get_result()
        if not success:
            rospy.logerr("Step action failed")
        # rospy.rostime.wallsleep(0.1)

   
    def step(self, action):
        self.max_episode_steps -= 1

        done = False

  
        action = self.DISCRETE_ACTIONS[action]
        info = {'cause' : 0}
        if action[-1] == 1:
            #DONE
            stop_, smooth_reward, _ = self._compute_twist(0, 0, 0)
            self.twist_pub.publish(stop_)

            self._perform_sim_steps(1)
            
            reward, info = self._compute_reward(compute_final_reward=True)
            
        else:
            twist, smooth_reward, noop = self._compute_twist(action[2], 
                                                       action[1], 
                                                       action[0])

            self.twist_pub.publish(twist)
            self._perform_sim_steps(self.sim_steps)

            observation = self._get_obs()

            if noop:
                self.noop_counter += 1
               
                if self.noop_counter >= 100:
                    info['cause'] = 'Noop'
                    done = True
                    reward = -1
                    return observation, reward, done, False, info
            
            else:
                self.noop_counter = 0

            reward , info = self._compute_reward()

        if reward != -0.01:
            done = True

            if action[-1] == 1:
                info['cause'] = 'OpenedGripper'
                self._open_gripper()
                twist, _, _ = self._compute_twist(0,0,1)

                self.twist_pub.publish(twist)
                self._perform_sim_steps(10)
                
                observation = self._get_obs()
                if reward > 0:
                    stable, _ = self._compute_reward(compute_final_reward=True)

                    if stable >= 0.98:
                        reward = 1
                    else:
                        if self.action_space_mode=='full':
                            reward -1
                        else:
                            reward /= 2
                            
        reward += smooth_reward


        return observation, reward, done, False, info


    def _initial_grasp(self, testing=False):
        open = False
        while not open:
            open = self._open_gripper()
        # self.pause_sim(paused=True)
        resp = self.set_gravity(env_id=0, gravity=[0, 0, 0])
        if resp:    
            obj_pos, obj_quat, angle = self.set_object_params(testing=testing)
        grasp_success = self._close_gripper()

 
        if grasp_success:

            self._setLoad(mass=self.obj_mass, load_inertia=list(np.eye(3).flatten()))

            resp = self.set_gravity(env_id=0, gravity=[0, 0, -9.81])
        
            self._perform_sim_steps(10)
            
            if not self._check_object_grip():
                grasp_success = False

        if not grasp_success:
            print("Grasp unsuccessful. Retry...")
        return grasp_success, (obj_pos, obj_quat, angle)

    def set_object_params(self, testing=False):
        
        # sample object type, mass, sliding friction, height, radius, length, width
        self._sample_obj_params() 


        # sample object start pose
        obj_pos, obj_quat, angle = self._sample_object_pose(testing=testing)
    
  
        # set object properties
        obj_geom_type = GeomType(value=self.obj_geom_type_value)
        obj_geom_properties = GeomProperties(env_id=0, name="object_geom", type=obj_geom_type, size_0=self.obj_size_0, size_1=self.obj_size_1, size_2=self.obj_size_2, friction_slide=1, friction_spin=0.005, friction_roll=0.0001)
        resp = self.set_geom_properties(properties=obj_geom_properties, set_type=True, set_mass=False, set_friction=False, set_size=True)
        if not resp.success:
            rospy.logerr("SetGeomProperties:failed to set object parameters")

        # set object pose with mass=0
        body_state = BodyState(env_id=0, name="object", pose=nparray_to_posestamped(obj_pos, obj_quat), mass=self.obj_mass)
        resp = self.set_body_state(state=body_state, set_pose=True, set_twist=True, set_mass=True, reset_qpos=False)
        if not resp.success:
            rospy.logerr("SetBodyState: failed to set object pose or object mass")

        return obj_pos, obj_quat, angle
    

    def _franka_state_callback(self, msg):
        topic = msg._connection_header["topic"]

        if self.last_timestamps["franka_state"] is not None:
            if msg.header.stamp < self.last_timestamps["franka_state"]:
                rospy.logerr(f"Detected jump back in time - current message is ignored (topic: {topic})")
                return
        
        self.last_timestamps["franka_state"] = msg.header.stamp
            
        ee_pos, ee_quat, ee_xyz = self._pose_quat_from_trafo(msg.O_T_EE)
        
        pose = np.concatenate([ee_pos, ee_quat])

        joint_positions = np.array(msg.q, dtype=np.float64)
        joint_torques = np.array(msg.tau_J, dtype=np.float64)
        joint_velocities = np.array(msg.dq, dtype=np.float64)
        
        if sum(joint_torques - self.current_msg["joint_torques"][-1]) != 0:
            self.current_msg["joint_positions"].append(joint_positions)
            self.current_msg["joint_torques"].append(joint_torques)
            self.current_msg["joint_velocities"].append(joint_velocities)
            self.current_msg["ee_pose"].append(pose)

         
    def _new_msg_callback(self, msg):
        topic = msg._connection_header["topic"]

        key = [k for k in self.current_msg.keys() if k in topic]
        if not key:
            rospy.logerr(f"No key found for current topic {topic} - current message is ignored")
        elif len(key) > 1:
            rospy.logerr(f"More than one key found for current topic {topic} - current message is ignored")
        else:
            if self.last_timestamps[key[0]] is not None:
                if msg.header.stamp < self.last_timestamps[key[0]]:
                    rospy.logerr(f"Detected jump back in time - current message is ignored (topic: {topic})")
                    return
                
            self.last_timestamps[key[0]] = msg.header.stamp # remember timestamp of old message
            
            if key[0] == 'myrmex_r' or key[0] == 'myrmex_l':
                
                raw = np.array(msg.sensors[0].values)
                
                myrmex_data = np.array([np.max(raw[i*5:(i+1)*5]) for i in range(int(len(raw)/5))])  / self.myrmex_max_val#+ 0.0000001*np.random.randn(self.num_taxels)

                myrmex_data = np.clip(myrmex_data, a_min=0, a_max=1)
                
                if sum(myrmex_data - self.current_msg[key[0]][-1]) != 0:
                    self.current_msg[key[0]].append(myrmex_data)

            if key[0] == 'object_contact_GT':
                self.current_msg[key[0]] = msg
            if key[0] == 'object_pos':
                self.current_msg[key[0]] = msg
            if key[0] == 'object_quat':
                self.current_msg[key[0]] = msg

    
    def _set_table_params(self, tableheight):
        
        tableheight -= 0.01
        obj_geom_type = GeomType(value=6)
        obj_geom_properties = GeomProperties(env_id=0, name="table_geom", type=obj_geom_type, size_0=0.2, size_1=0.5, size_2=tableheight, friction_slide=1, friction_spin=0.005, friction_roll=0.0001)
        resp = self.set_geom_properties(properties=obj_geom_properties, set_type=True, set_mass=False, set_friction=False, set_size=True)
        if not resp.success:
            rospy.logerr("SetGeomProperties:failed to set object parameters")        
        
        body_state = BodyState(env_id=0, name="table", pose=nparray_to_posestamped(np.array([0.3, 0, tableheight]), np.array([1,0,0,0])), mass=50)
        resp = self.set_body_state(state=body_state, set_pose=True, set_twist=False, set_mass=False, reset_qpos=False)
        if not resp.success:
            rospy.logerr("SetBodyState: failed to set object pose or object mass")

        obj_geom_type = GeomType(value=6)
        obj_geom_properties = GeomProperties(env_id=0, name="tabletop_geom", type=obj_geom_type, size_0=0.2, size_1=0.5, size_2=0.01, friction_slide=1, friction_spin=0.005, friction_roll=0.0001)
        resp = self.set_geom_properties(properties=obj_geom_properties, set_type=True, set_mass=False, set_friction=False, set_size=True)
        if not resp.success:
            rospy.logerr("SetGeomProperties:failed to set object parameters")        
        
        body_state = BodyState(env_id=0, name="tabletop", pose=nparray_to_posestamped(np.array([0.3, 0, tableheight + tableheight + 0.01]), np.array([1,0,0,0])), mass=50)
        resp = self.set_body_state(state=body_state, set_pose=True, set_twist=False, set_mass=False, reset_qpos=False)
        if not resp.success:
            rospy.logerr("SetBodyState: failed to set object pose or object mass")

    def _set_world_to_initial_state(self, options):
        
        self.reset_world()

        #stop all movement
        self._setLoad(mass=0, load_inertia=list(np.eye(3).flatten()))
        twist, _, _ = self._compute_twist(0, 0, 0)
        self.twist_pub.publish(twist)

        #Delete Old messages
        self.current_msg = {"myrmex_l" : deque([np.zeros(4) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                    "myrmex_r" : deque([np.zeros(4) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                    "joint_positions" : deque([np.zeros(7) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                    "joint_velocities" : deque([np.zeros(7) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                    "joint_torques" : deque([np.zeros(7) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                    "ee_pose" : deque([np.zeros(7) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                    "object_pos" : None, 
                    "object_quat" : None, 
                    "object_contact_GT" : None}
        
        self.last_timestamps = {"myrmex_l" : None, "myrmex_r" : None, "franka_state" : None, "object_pos" : None, "object_quat" : None, "object_contact_GT" : None}
        #wait for all messages to arrive

        while True:
            self._perform_sim_steps(1)
            if None not in self.current_msg.values():
                break
        
        if not options is None:
            if options['testing'] == False:
                self.table_height = np.random.uniform(options['min_table_height'], self.max_table_height)
            else:
                self.table_height = options['min_table_height']

            obj_geom_type = GeomType(value=6)
            obj_geom_properties = GeomProperties(env_id=0, name="table_geom", type=obj_geom_type, size_0=0.2, size_1=0.5, size_2=self.table_height, friction_slide=1, friction_spin=0.005, friction_roll=0.0001)
            resp = self.set_geom_properties(properties=obj_geom_properties, set_type=True, set_mass=False, set_friction=True, set_size=True)
            if not resp.success:
                rospy.logerr("SetGeomProperties:failed to set object parameters")        
                
        #reset step counter
        self.max_episode_steps = 300
        return 0
    

    def _reset_robot(self):

        twist, _, _ = self._compute_twist(0, 0, 0)
        self.twist_pub.publish(twist)
        if not self.continuous:
            self._perform_sim_steps(10)

        self._open_gripper()
        self._set_table_params(0.01)

        self._setLoad(mass=0, load_inertia=list(np.eye(3).flatten()))
            
        self.reset_world()
        if not self.continuous:
            self._perform_sim_steps(10)
    
        return 0
    
    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed)

        self._reset_robot()

        success = False

        self.last_command = None
        self.sim_steps = options['sim_steps']
        self.max_episode_steps = options['max_steps']

        self.noop_counter = 0
        self.contact_counter = 0
        self.reward_fn = options['reward_fn']
        regrasp_counter = 0

        while not success:
            
            gap = options['gap_size']
            
            self.angle_range = options['angle_range']

            self.current_msg = {"myrmex_l" : deque([np.zeros(4) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                    "myrmex_r" : deque([np.zeros(4) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                    "joint_positions" : deque([np.zeros(7) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                    "joint_velocities" : deque([np.zeros(7) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                    "joint_torques" : deque([np.zeros(7) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                    "ee_pose" : deque([np.zeros(7) for _ in range(self.timesteps)], maxlen=self.timesteps), 
                    "object_pos" : None, 
                    "object_quat" : None, 
                    "object_contact_GT" : None}
    
            self.last_timestamps = {"myrmex_l" : None, "myrmex_r" : None, "franka_state" : None, "object_pos" : None, "object_quat" : None, "object_contact_GT" : None}

            #wait for messages CAUSE FOR ERROR/WARNING MESSAGES
            if self.continuous:
                n = time.time()
                while True:
                    if None not in self.current_msg.values():
                        print("All Msgs arrived after: " , time.time() - n)
                        break
            else:
                while True:
                    self._perform_sim_steps(1)
                    if None not in self.current_msg.values():
                        break

            success, (obj_pos, obj_quat, angle) = self._initial_grasp(testing=options['testing'])
            if not success:
                self._reset_robot()
                regrasp_counter += 1
                          
                if regrasp_counter >= 10:
                    break
            
                
        if not regrasp_counter >= 10:

            corners = compute_corner_coords(obj_pos, self.obj_size_0, self.obj_size_1, self.obj_size_2, obj_quat)
 
            lowpoint = np.min(corners[:, -1])
  
            self.table_height = (lowpoint/2) - gap
         
            self._set_table_params(self.table_height)

            info = {'info' : {'sampled_gap' : gap, 'obj_angle' : angle, 'success' : True}}
        else:
            info = {'info' : {'sampled_gap' : None, 'obj_angle' : None, 'success' : False}}

    
        observation    = self._get_obs()
        return observation, info 
    
    def close(self):
        if "self.launch" in locals():
            self.launch.stop()