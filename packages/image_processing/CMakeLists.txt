cmake_minimum_required(VERSION 2.8.3)
project(image_processing)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  duckietown_msgs
  sensor_msgs
  pluginlib
  nodelet
  nodelet_topic_tools
  cv_bridge
  image_transport
)

include_directories(${catkin_INCLUDE_DIRS})

catkin_package(
  LIBRARIES image_processing
  CATKIN_DEPENDS nodelet roscpp sensor_msgs
)

# Nodelet library
add_library(image_processing src/decoder.cpp)
target_link_libraries(image_processing ${catkin_LIBRARIES})
if(catkin_EXPORTED_LIBRARIES)
  add_dependencies(image_processing ${catkin_EXPORTED_LIBRARIES})
endif()

## Mark the nodelet library for installations
install(TARGETS image_processing
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
