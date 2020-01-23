#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Vector3, TransformStamped
import numpy as np

from moveGroupInterface import MoveGroupInterface
from perceptionInterface import PerceptionInterface
from utils_geom import *
from visualization_msgs.msg import Marker


def main():
  # try:
  print('============ Press Enter to start')
  raw_input()
  moveGroup = MoveGroupInterface()
  perception = PerceptionInterface()


  print("============ Press Enter to move ee for capturing point cloud ...")
  raw_input()
  moveGroup.move_to_start()

  print("============ Press Enter to capture point cloud ...")
  raw_input()
  trans = moveGroup.look_up_transform('/panda_link0', '/camera_depth_optical_frame')
  
  pcl, contact_points_list = perception.capture_pcl(trans)
  print("Point cloud saved!")

  print("============ Press Enter to sample a contact point and a grasp ...")
  idx = int(np.random.choice(np.arange(len(contact_points_list))))
  print('Chosen idx: ', idx)

  contact_point = np.asarray(pcl.points)[idx]
  contact_normal = np.asarray(pcl.normals)[idx]
  print(contact_point)
  print(contact_normal)

  graspOrn = vecs2quat(np.array([0,0,1]), contact_normal)
  offsetBefore = 0.20
  graspPosBefore = contact_point - contact_normal*offsetBefore
  print(graspPosBefore)

  moveGroup.move(graspPosBefore, graspOrn)

  while 1:
    continue

  # print("============ Press Enter to plan and display a Cartesian path ...")
  # raw_input()
  # cartesian_plan, fraction = moveGroup.plan_cartesian_path()

  # print("============ Press Enter to execute a saved path ...")
  # raw_input()
  # moveGroup.execute_plan(cartesian_plan)

  # except rospy.ROSInterruptException:
  #   return
  # except KeyboardInterrupt:
  #   return

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
