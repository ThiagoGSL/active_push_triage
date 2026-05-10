cmake_minimum_required(VERSION 3.4)
project(ur3_push_mujoco)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY assets
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

catkin_install_python(PROGRAMS
  scripts/interactive_marker.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
