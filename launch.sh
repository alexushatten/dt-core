#!/bin/bash

set -e

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------
#roslaunch duckietown_demos indefinite_navigation.launch

#TODO: revert this
roslaunch pose_estimation_test pose_estimation_test.launch veh:=watchtower01
#roslaunch apriltag_ros apriltag_detector_node.launch veh:=watchtower01
#TODO: revert this
