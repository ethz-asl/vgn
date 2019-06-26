cmake_minimum_required(VERSION 3.1)
project(vgn_ros)

find_package(catkin_simple REQUIRED)

catkin_python_setup()

catkin_simple(ALL_DEPS_REQUIRED)

cs_install()
cs_export()
