#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
sys.dont_write_bytecode = True

import rospy
from geometry_msgs.msg import Vector3, TransformStamped
import numpy as np
from visualization_msgs.msg import Marker
import PyKDL
from tf_conversions import posemath

from graspEnv import GraspEnv
# from cameraEnv import CameraEnv
from utils_geom import *


def main():
  # try:
  print('============ Press Enter to initialize grasping environment...')
  raw_input()
  graspEnv = GraspEnv()
#   cameraEnv = CameraEnv()

  print("============ Press Enter to move ee for capturing depth...")
  raw_input()
  depth_pose = PyKDL.Frame(PyKDL.Rotation.Quaternion(-0.9239554, 0.3824994, 0.0003046, 0.0007358), 
                              PyKDL.Vector(0.25, -0.30, 0.40))
  graspEnv.move_to_goal(posemath.toMsg(depth_pose))

#   print("============ Press Enter to take depth image and infer grasp pose...")

#   print("============ Press Enter to execute pose...")
#   print("============ Press Enter to reach down...")
#   print("============ Press Enter to grasp...")
#   print("============ Press Enter to lift...")

#   print("============ Press Enter to move ee to idle position...")
#   raw_input()
#   idle_pose = PyKDL.Frame(PyKDL.Rotation.Quaternion(-0.9239554, 0.3824994, 0.0003046, 0.0007358), 
#                               PyKDL.Vector(0.30, 0.00, 0.40))
#   graspEnv.move_to_goal(posemath.toMsg(idle_pose))


  # print("============ Press Enter to sample a contact point and a grasp ...")
  # idx = int(np.random.choice(np.arange(len(contact_points_list))))
  # print('Chosen idx: ', idx)

  # contact_point = np.asarray(pcl.points)[idx]
  # contact_normal = np.asarray(pcl.normals)[idx]
  # print(contact_point)
  # print(contact_normal)

  # graspOrn = vecs2quat(np.array([0,0,1]), contact_normal)
  # offsetBefore = 0.20
  # graspPosBefore = contact_point - contact_normal*offsetBefore
  # print(graspPosBefore)

  while 1:
    continue


if __name__ == '__main__':
  main()


  # def plan_cartesian_path(self, scale=1):
  #   ## You can plan a Cartesian path directly by specifying a list of waypoints
  #   ## for the end-effector to go through. If executing interactively in a
  #   ## Python shell, set scale = 1.0.
  #   ##
  #   waypoints = []

  #   wpose = self.move_group.get_current_pose().pose
  #   wpose.position.z -= scale * 0.1  # First move up (z)
  #   wpose.position.y += scale * 0.2  # and sideways (y)
  #   waypoints.append(copy.deepcopy(wpose))

  #   wpose.position.x += scale * 0.1  # Second move forward/backwards in (x)
  #   waypoints.append(copy.deepcopy(wpose))

  #   wpose.position.y -= scale * 0.1  # Third move sideways (y)
  #   waypoints.append(copy.deepcopy(wpose))

  #   # We want the Cartesian path to be interpolated at a resolution of 1 cm
  #   # which is why we will specify 0.01 as the eef_step in Cartesian
  #   # translation.  We will disable the jump threshold by setting it to 0.0,
  #   # ignoring the check for infeasible jumps in joint space, which is sufficient
  #   # for this tutorial.
  #   (plan, fraction) = self.move_group.compute_cartesian_path(
  #                                      waypoints,   # waypoints to follow
  #                                      0.01,        # eef_step
  #                                      0.0)         # jump_threshold

  #   # Note: We are just planning, not asking move_group to actually move the robot yet:
  #   return plan, fraction


  # def execute_plan(self, plan):
  #   ## Use execute if you would like the robot to follow
  #   ## the plan that has already been computed:
  #   self.move_group.execute(plan, wait=True)

  #   ## **Note:** The robot's current joint state must be within some tolerance of the
  #   ## first waypoint in the `RobotTrajectory`_ or ``execute()`` will fail
