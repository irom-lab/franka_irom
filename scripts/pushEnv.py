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
# import tf
import tf.transformations as tft
import PyKDL
import ropy as rp
from std_msgs.msg import Empty, String, Float64MultiArray
from std_srvs.srv import Empty as EmptySrv
from geometry_msgs.msg import Vector3, Quaternion, TransformStamped, Twist, Pose
from franka_msgs.msg import FrankaState, Errors as FrankaErrors
import tf_helpers as tfh
from rviz_tools import RvizMarkers
from utils_geom import quatMult, euler2quat, log_rot

from franka_irom_controllers.panda_commander import PandaCommander
from franka_irom_controllers.control_switcher import ControlSwitcher


class PushEnv(object):
	def __init__(self):
		super(PushEnv, self).__init__()

		# Initialize rospy node
		rospy.init_node('grasp_env', anonymous=True)

		# Set up panda moveit commander, pose control
		self.pc = PandaCommander(group_name='panda_arm')

		# Initialize
		rospy.Subscriber('/push_infer', Vector3, self.__push_infer_callback, queue_size=1)
		self.delta_x = 0.0
		self.delta_y = 0.0
		self.delta_x_buffer = 0.0
		self.delta_y_buffer = 0.0

		# Set up ropy and joint velocity controller
		self.panda = rp.Panda()
		self.curr_velocity_publish_rate = 100.0  # for libfranka
		self.curr_velo_pub = rospy.Publisher('/joint_velocity_node_controller/joint_velocity', Float64MultiArray, queue_size=1)
		self.curr_velo = Float64MultiArray()

		# Set up switch between moveit and velocity, start with moveit
		self.cs = ControlSwitcher({
	  		'moveit': 'position_joint_trajectory_controller',
			'velocity': 'joint_velocity_node_controller'})
		self.cs.switch_controller('moveit')

		# Set up update with inference
		self.update_rate = 5.0  # for inference
		update_topic_name = '~/update'
		self.update_pub = rospy.Publisher(update_topic_name, Empty, queue_size=1)
		rospy.Subscriber(update_topic_name, Empty, self.__update_callback, queue_size=1)

		# Subscribe to robot state
		self.robot_state = None
		rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.__robot_state_callback, queue_size=1)

		# For visualization in Rviz
		self.markers = RvizMarkers('/panda_link0', 'visualization_marker')

		# Errors
		self.ROBOT_ERROR_DETECTED = False
		self.BAD_UPDATE = False

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
				self.stop()
				if not self.ROBOT_ERROR_DETECTED:
					rospy.logerr('Robot Error Detected')
				self.ROBOT_ERROR_DETECTED = True

	def __push_infer_callback(self, msg):
		self.delta_x_buffer = msg.x
		self.delta_y_buffer = msg.y

	def __trigger_update(self):
		# Let ROS handle the threading for me.
		self.update_pub.publish(Empty())
	def __update_callback(self, msg):
		self.delta_x = self.delta_x_buffer
		self.delta_y = self.delta_y_buffer

		self.wTep = array(self.robot_state.O_T_EE).reshape(4,4,order='F')
		self.wTep[0,3] += self.delta_x
		self.wTep[1,3] += self.delta_y
		print(self.delta_x, self.delta_y)

	def stop(self):
		self.curr_velo = Float64MultiArray()
		self.curr_velo.data = [0., 0., 0., 0., 0., 0., 0.]
		self.curr_velo_pub.publish(self.curr_velo)

	def go(self):
		# startQuat = quatMult(array([1.0, 0.0, 0.0, 0.0]), euler2quat([np.pi/4,0,0]))
		# print(startQuat)

		print("============ Press Enter to move to initial pose...")
		raw_input()

		# straight down, z=0.155 hits table
		# start_pose = [0.35, 0.0, 0.18, 0.89254919, -0.36948312,  0.23914479, -0.09822433]  # for pushing
		# start_pose = [0.60, 0.0, 0.155, 0.92387953, -0.38268343, 0., 0.]  # straight down
		# start_pose = [0.75, 0.15, 0.16, 0.92387953, -0.38268343, 0., 0.]  # straight down
		# self.pc.goto_pose(start_pose, velocity=0.1)
		# self.pc.set_gripper(0.0)
		start_joint_angles = [-0.011, 0.261, 0.014, -2.941, 0.010, 3.725, 0.776]
		self.pc.goto_joints(start_joint_angles)
		self.pc.set_gripper(0.03)
		rospy.sleep(1.0)

		# loop actions
		while not rospy.is_shutdown():

			print("============ Press Enter to switch to velocity control...")
			raw_input()

			# Initialize target pose
			self.wTep = array(self.robot_state.O_T_EE).reshape(4,4,order='F')

			# START
			self.cs.switch_controller('velocity')
			r = rospy.Rate(self.curr_velocity_publish_rate)
			arrived = False
			ctr = 0
			start_time = 0
			while 1:

				# Get current joint angles
				self.panda.q = np.array(self.robot_state.q)

				# Update current ee pose
				wTe = array(self.robot_state.O_T_EE).reshape(4,4,order='F')

				# Desired end-effecor spatial velocity
				v, arrived = rp.p_servo(wTe, self.wTep, gain=2.0, threshold=0.1)

				# Solve for the joint velocities dq
				dq = np.matmul(np.linalg.pinv(self.panda.Je), v)

				# Stopping criteria: check ee pos, offset bt tip and ee is 7.7cm
				if self.robot_state.O_T_EE[-4] > 0.67 or abs(self.robot_state.O_T_EE[-3]) > 0.25:
					break

				# Update wTep in 5Hz
				ctr += 1
				if ctr >= self.curr_velocity_publish_rate/self.update_rate:
					ctr = 0
					self.__trigger_update()
					print(time.time()-start_time)
					start_time = time.time()
				# # Check cartesian contact
				# if any(self.robot_state.cartesian_contact):
				# 	self.stop()
				# 	rospy.logerr('Detected cartesian contact during velocity control loop.')
				# 	break

				# Send joint velocity cmd
				v = Float64MultiArray()
				v.data = dq
				self.curr_velo_pub.publish(v)

				r.sleep()

			# Send zero velocities for a few seconds
			ctr = 0
			while ctr < 100:
				self.stop()
				ctr += 1		
				r.sleep()

			# Back
			print("============ Press Enter to home...")
			raw_input()
			start_joint_angles = [-0.011, 0.261, 0.014, -2.941, 0.010, 3.725, 0.776]
			self.cs.switch_controller('moveit')
			self.pc.goto_joints(start_joint_angles)
			self.pc.set_gripper(0.03)


if __name__ == '__main__':

	pushEnv = PushEnv()
	pushEnv.go()
