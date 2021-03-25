cmake_minimum_required(VERSION 3.0.2)
project(ros_3d_pointing_detection)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
)

add_message_files(
  FILES
  DetectedObject.msg
)

catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime) # declares dependencies for packages that depend on this package.

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/3d_pointing_detection.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
