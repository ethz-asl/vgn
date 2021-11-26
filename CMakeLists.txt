cmake_minimum_required(VERSION 3.1)
project(vgn)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  GraspConfig.msg
)

add_service_files(
  FILES
  GetMapCloud.srv
  GetSceneCloud.srv
  PredictGrasps.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)
