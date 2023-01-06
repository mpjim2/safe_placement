cmake_minimum_required(VERSION 3.0.2)
project(safe_placement)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  controller_interface
  franka_hw
  franka_gripper
  geometry_msgs
  hardware_interface
  joint_limits_interface
  message_generation
  pluginlib
  realtime_tools
)

find_package(Franka 0.9.0 QUIET)
if(NOT Franka_FOUND)
  find_package(Franka 0.8.0 REQUIRED)
endif()

find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES safe_placement
  CATKIN_DEPENDS controller_interface 
                 franka_hw 
                 franka_gripper 
                 geometry_msgs 
                 hardware_interface 
                 joint_limits_interface 
                 message_generation 
                 pluginlib 
                 realtime_tools 
                 roscpp 
                 rospy
  DEPENDS Franka
          EIGEN3
)

include_directories(include ${catkin_INCLUDE_DIRS}
                            ${EIGEN3_INDLUDE_DIRS})

add_library(${PROJECT_NAME} src/cartesian_twist_controller.cpp)

target_link_libraries(${PROJECT_NAME} 
                      ${catkin_LIBRARIES}
                      ${EIGEN3_LIBRARIES})


install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
        )

install(FILES custom_controllers_plugin.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )