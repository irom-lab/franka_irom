#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
sys.dont_write_bytecode = True

import numpy as np
from numpy import array
import time
import copy

import rospy
import tf
import PyKDL
import ropy as rp
# from std_msgs.msg import String
from geometry_msgs.msg import Vector3, Quaternion, TransformStamped, Twist, Pose
from franka_msgs.msg import FrankaState, Errors as FrankaErrors
# from tf_conversions import posemath
from rviz_tools import RvizMarkers
from utils_geom import quatMult, euler2quat

from franka_irom.srv import GraspInfer 
from franka_irom_controllers.panda_commander import PandaCommander


class GraspEnv(object):
	def __init__(self):
		super(GraspEnv, self).__init__()

		# Initialize rospy node
		rospy.init_node('grasp_env', anonymous=True)

		# Initialize service from cameraEnv
		service_name = '/camera_env/infer_grasp'
		rospy.wait_for_service(service_name)
		self.grasp_infer_srv = rospy.ServiceProxy(service_name, GraspInfer)

		# Set up panda moveit commander, pose control
		self.pc = PandaCommander(group_name='panda_arm')

		# Set up ropy and joint velocity controller
		self.panda = rp.Panda()
		self.joint_velocity_pub = rospy.Publisher('/cartesian_velocity_controller/cartesian_velocity', Twist, queue_size=1)

		# Subscribe to robot state
		self.robot_state = None
		rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.__robot_state_callback, queue_size=1)

		# For visualization in Rviz
		self.markers = RvizMarkers('/panda_link0', 'visualization_marker')

		# Errors
		self.ROBOT_ERROR_DETECTED = False

		# Add table as collision
		self.pc.add_table()


	def __recover_robot_from_error(self):
		rospy.logerr('Recovering')
		self.pc.recover()
		rospy.logerr('Done')
		self.ROBOT_ERROR_DETECTED = False


	def __robot_state_callback(self, msg):
		self.robot_state = msg
		if any(self.robot_state.cartesian_collision):
			if not self.ROBOT_ERROR_DETECTED:
				rospy.logerr('Detected Cartesian Collision')
			self.ROBOT_ERROR_DETECTED = True
		for s in FrankaErrors.__slots__:
			if getattr(msg.current_errors, s):
				self.pc.stop()  # stop movement
				if not self.ROBOT_ERROR_DETECTED:
					rospy.logerr('Robot Error Detected')
				self.ROBOT_ERROR_DETECTED = True


	def go(self):

		# startQuat = quatMult(array([1.0, 0.0, 0.0, 0.0]), euler2quat([np.pi/4,0,0]))
		# print(startQuat)

		print("============ Press Enter to move to initial pose...")
		raw_input()
		start_joint_angles = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
		self.pc.goto_joints(start_joint_angles)

		print("============ Press Enter to move away from center...")
		raw_input()
		joint_angles = [-0.797, -0.137, 0.181, -2.608, 0.039, 2.472, -0.649]
		self.pc.set_gripper(0.1)
		self.pc.goto_joints(joint_angles)

		print("============ Press Enter to ask for grasp pose...")
		raw_input()
		if self.ROBOT_ERROR_DETECTED:
			self.__recover_robot_from_error()
		res = self.grasp_infer_srv()
		target_pos = array([res.pos.x, res.pos.y, res.pos.z])
		# target_pos = array([res.pos.x, res.pos.y, res.pos.z+0.03]) # account for longer finger
		target_yaw = res.yaw
		print(target_pos, target_yaw)

		print("============ Press Enter to move to above target...")
		raw_input()
		if self.ROBOT_ERROR_DETECTED:
			self.__recover_robot_from_error()
		target_pos_above = target_pos + array([0.,0.,0.10])
		target_quat = quatMult(array([1.0, 0.0, 0.0, 0.0]), euler2quat([np.pi/4-target_yaw,0,0]))
		target_above_pose =list(np.concatenate((target_pos_above, target_quat)))
		self.pc.goto_pose(target_above_pose, velocity=0.20)

		print(target_pos_above, array(self.robot_state.O_T_EE).reshape(4,4,order='F')[:3,-1])

		print("============ Press Enter to reach down...")
		raw_input()
		if self.ROBOT_ERROR_DETECTED:
			self.__recover_robot_from_error()
		target_pose =list(np.concatenate((target_pos, target_quat)))
		self.pc.goto_pose(target_pose, velocity=0.05)

		if self.ROBOT_ERROR_DETECTED:
			self.__recover_robot_from_error()

		print("============ Press Enter to grasp...")
		raw_input()
		if self.ROBOT_ERROR_DETECTED:
			self.__recover_robot_from_error()
		self.pc.grasp(width=-0.01, e_inner=-0.01, e_outer=0.005, speed=0.03, force=10)

		print("============ Press Enter to lift...")
		raw_input()
		if self.ROBOT_ERROR_DETECTED:
			self.__recover_robot_from_error()
		self.pc.goto_pose(target_above_pose, velocity=0.05)

		print("============ Press Enter to put down...")
		raw_input()
		if self.ROBOT_ERROR_DETECTED:
			self.__recover_robot_from_error()
		self.pc.goto_pose(target_pose, velocity=0.05)		
		self.pc.set_gripper(0.1)

		print("============ Press Enter to lift and move back...")
		raw_input()
		if self.ROBOT_ERROR_DETECTED:
			self.__recover_robot_from_error()
		self.pc.goto_pose(target_above_pose, velocity=0.05)
		# start_pose = [0.30, -0.30, 0.30, 1.0, 0.0, 0.0, 0.0]
		joint_angles = [-0.797, -0.137, 0.181, -2.608, 0.039, 2.472, -0.649]
		# self.pc.goto_pose(start_pose, velocity=0.25)
		self.pc.goto_joints(joint_angles)

		if self.ROBOT_ERROR_DETECTED:
			self.__recover_robot_from_error()

		# print("============ Press Enter to home...")
		# raw_input()
		# start_joint_angles = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
		# self.pc.goto_joints(start_joint_angles)
		# self.pc.set_gripper(0.1)

		# if not grasp_ret or self.ROBOT_ERROR_DETECTED:
		# 	rospy.logerr('Something went wrong, aborting this run')
		# 	if self.ROBOT_ERROR_DETECTED:
		# 		self.__recover_robot_from_error()
		# 	continue

		return 1


if __name__ == '__main__':
	graspEnv = GraspEnv()
	graspEnv.go()
