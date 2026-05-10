cmake_minimum_required(VERSION 3.4)
project(ur3_push_sim2real)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(PROGRAMS
  scripts/interactive_marker.py
  scripts/set_fixed_EE_trafo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(
  PROGRAMS
    nodes/dt_vicon_scene_spawner
    nodes/collision_monitor
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)