#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
sys.dont_write_bytecode = True

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from moveit_commander import MoveGroupCommander
from actionlib_msgs.msg import GoalStatusArray
import tf
import PyKDL

from geometry_msgs.msg import Vector3, Quaternion, TransformStamped
from tf_conversions import posemath
from rviz_tools import RvizMarkers


class MoveGroupInterface(object):
    def __init__(self):
        super(MoveGroupInterface, self).__init__()

        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_interface', anonymous=True)


        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        group_name = "panda_arm"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)

        ## Create a `DisplayTrajectory`_ ROS publisher
        self.display_trajectory_publisher = rospy.Publisher(
            					'/move_group/display_planned_path',
                                moveit_msgs.msg.DisplayTrajectory,
                                queue_size=20)

        # For visualization in Rviz
        self.markers = RvizMarkers('/panda_link0', 'visualization_marker')

        # We can get the name of the reference frame for this robot:
        self.planning_frame = self.move_group.get_planning_frame()
        # print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        self.eef_link = self.move_group.get_end_effector_link()
        # print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        self.group_names = self.robot.get_group_names()
        # print("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        # cur_states = robot.get_current_state()
        # print("============ Current state: ", cur_states)

        # TF listener and broadcaster
        self.listener = tf.TransformListener()
        self.br = tf.TransformBroadcaster()


    def move_to_goal(self, pose_goal):
        # cur_pose = self.move_group.get_current_pose().pose

        # while 1:
        #     self.br.sendTransform((pose_goal.position.x, pose_goal.position.y, pose_goal.position.z),
        #                     (pose_goal.orientation.x, pose_goal.orientation.y, pose_goal.orientation.z, pose_goal.orientation.w),
        #                     rospy.Time.now(),
        #                     "target",
        #                     "panda_link0")
        #     rospy.sleep(0.01)

        # Send goal
        self.move_group.set_pose_target(pose_goal)

        # Visualize trajectory
        # plan = self.move_group.plan()
        # self.display_trajectory(plan)
        # while 1:
        #     continue

        ## Now, we call the planner to compute the plan and execute it.
        plan = self.move_group.go(wait=True)

        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()
        
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        self.move_group.clear_pose_targets()

        # Check
        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)



    def display_trajectory(self, plan):
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        self.display_trajectory_publisher.publish(display_trajectory)


    def look_up_transform(self, target_frame, source_frame):
        transform = TransformStamped()
        try:
            (trans, rot) = self.listener.lookupTransform(target_frame, source_frame, rospy.Duration(0))
            transform.transform.rotation = Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
            transform.transform.translation = Vector3(x=trans[0], y=trans[1], z=trans[2])
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print('Error when looking up for transform!')
        return transform


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True
