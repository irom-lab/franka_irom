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

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        self.robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        self.scene = moveit_commander.PlanningSceneInterface()

        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        group_name = "panda_arm"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)

        ## Create a `DisplayTrajectory`_ ROS publisher
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
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

        ################################################################
        self.z_command = 0
        rospy.Subscriber('/spacenav/offset', Vector3, self.joy_callback)

        # TF listener and broadcaster
        self.listener = tf.TransformListener()
        self.br = tf.TransformBroadcaster()

    def joy_callback(self, msg):
        self.z_command = msg.z
        # print(self.z_command)

    def move_to_start(self):
        # cur_pose = self.move_group.get_current_pose().pose

        # # Desired ee pose
        # pose_goal = geometry_msgs.msg.Pose()
        # pose_goal.orientation = cur_pose.orientation
        # pose_goal.position = cur_pose.position
    
        # Transform from panda_link0 to camera_depth_frame, PyKDL uses x,y,z,w for quaternion
        base_depth_pose = PyKDL.Frame(PyKDL.Rotation.Quaternion(0.1463059, 0.3534126, -0.3534126, 0.8536941), PyKDL.Vector(0.217, 0.283, 0.283+0.05))

        # Transform from camera_depth_frame to panda_link8
        trans = self.look_up_transform('/camera_depth_frame', '/panda_link8')
        depth_ee_pose = PyKDL.Frame(PyKDL.Rotation.Quaternion(trans.transform.rotation.x, trans.transform.rotation.y, 
            trans.transform.rotation.z, trans.transform.rotation.w), 
            PyKDL.Vector(trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z))

        # Desired transfrom from link0 to link8
        base_ee_pose = base_depth_pose*depth_ee_pose
        pose_goal = posemath.toMsg(base_ee_pose)
        # print(pose_goal.position)
        # print(pose_goal.orientation)

        # while 1:
        #     self.br.sendTransform((pose_goal.position.x, pose_goal.position.y, pose_goal.position.z),
        #                     (pose_goal.orientation.x, pose_goal.orientation.y, pose_goal.orientation.z, pose_goal.orientation.w),
        #                     rospy.Time.now(),
        #                     "target",
        #                     "panda_link0")
        #     rospy.sleep(0.01)

        self.move_group.set_pose_target(pose_goal)

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

    def move(self, pos, quat):
        # Set target pose based on inputs
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        pose_goal.position = Vector3(x=pos[0], y=pos[1], z=pos[2])
        
        # Visualize pose in Rviz
        axis_length = 0.4
        axis_radius = 0.05
        self.markers.publishAxis(pose_goal, axis_length, axis_radius, 30.0) # pose, axis length, radius, lifetime

        # Set target
        self.move_group.set_pose_target(pose_goal)

        # Plan and visualize trajectory
        plan = self.move_group.plan()
        self.display_trajectory(plan)


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
