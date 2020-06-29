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

		# Set up ropy and joint velocity controller
		self.panda = rp.Panda()
		self.max_velo = 1.0  # not using rn
		self.curr_velocity_publish_rate = 100.0  # for libfranka
		self.curr_velo_pub = rospy.Publisher('/joint_velocity_node_controller/joint_velocity', Float64MultiArray, queue_size=1)
		self.curr_velo = Float64MultiArray()
		self._in_velo_loop = False

		# Set up switch between moveit and velocity, start with moveit
		self.cs = ControlSwitcher({
	  		'moveit': 'position_joint_trajectory_controller',
			'velocity': 'joint_velocity_node_controller'})
		self.cs.switch_controller('moveit')

		# Set up update with inference
		self.update_rate = 100.0  # for inference
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

		# # TF listener and broadcaster
		# self.listener = tf.TransformListener()
		# self.br = tf.TransformBroadcaster()

		# self.__recover_robot_from_error()


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


	def __update_callback(self, msg):
		# Update the MVP Controller asynchronously
		if not self._in_velo_loop:
			# Stop the callback lagging behind
			return

		# TODO: subscribe to inference node

		# res = self.entropy_srv()
		# if not res.success:
		# 	# Something has gone wrong, 0 velocity.
		# 	self.BAD_UPDATE = True
		# 	self.curr_velo = Twist()
		# 	return

		# self.viewpoints = res.no_viewpoints

		# # Calculate the required angular velocity to match the best grasp.
		# q = tfh.quaternion_to_list(res.best_grasp.pose.orientation)
		# curr_R = np.array(self.robot_state.O_T_EE).reshape((4, 4)).T
		# cpq = tft.quaternion_from_matrix(curr_R)
		# dq = tft.quaternion_multiply(q, tft.quaternion_conjugate(cpq))
		# d_euler = tft.euler_from_quaternion(dq)
		# res.velocity_cmd.angular.z = d_euler[2]

		# self.best_grasp = res.best_grasp
		# self.curr_velo = res.velocity_cmd

		# tfh.publish_pose_as_transform(self.best_grasp.pose, 'panda_link0', 'G', 0.05)


	def __trigger_update(self):
		# Let ROS handle the threading for me.
		self.update_pub.publish(Empty())


	def stop(self):
		self.curr_velo = Float64MultiArray()
		self.curr_velo.data = [0., 0., 0., 0., 0., 0., 0.]
		self.curr_velo_pub.publish(self.curr_velo)


	def go(self):
		
		print("============ Press Enter to move to initial pose...")
		raw_input()
		# startQuat = quatMult(array([1.0, 0.0, 0.0, 0.0]), euler2quat([np.pi/4,0,0]))
		# print(startQuat)

		# straight down, z=0.155 hits table

		start_pose = [0.35, 0.0, 0.18, 0.89254919, -0.36948312,  0.23914479, -0.09822433]  # for pushing
		# start_pose = [0.30, 0.0, 0.155, 0.92387953, -0.38268343, 0., 0.]  # straight down
		self.pc.goto_pose(start_pose, velocity=0.1)
		self.pc.set_gripper(0.015)
		rospy.sleep(3.0)

		print("============ Press Enter to switch to velocity control...")
		raw_input()
		self.cs.switch_controller('velocity')
		ctr = 0
		r = rospy.Rate(self.curr_velocity_publish_rate)
		self._in_velo_loop = True

		# # Get current joint angle, tested
		# self.panda.q = np.array(self.robot_state.q)

		# Get current ee pose
		wTe = array(self.robot_state.O_T_EE).reshape(4,4,order='F')  # column major

		# Set desired pose
		wTep = np.copy(wTe)
		wTep[0,3] += 0.10  # move 5cm in x

		arrived = False
		while not arrived:
			# if self.ROBOT_ERROR_DETECTED or self.BAD_UPDATE:
			# 	return False

			# Get current joint angles
			self.panda.q = np.array(self.robot_state.q)

			# Update current ee pose
			wTe = array(self.robot_state.O_T_EE).reshape(4,4,order='F')  # column major

			# Desired end-effecor spatial velocity, threshold 0.01 - 0.493 final as 0.50 target
			v, arrived = rp.p_servo(wTe, wTep, gain=0.5, threshold=0.01)
			
			# Solve for the joint velocities dq
			dq = np.matmul(np.linalg.pinv(self.panda.Je), v)

			# Stopping criteria: check ee_x pos
			if self.robot_state.O_T_EE[-4] > 0.7:
				self.stop()
				rospy.sleep(0.1)
				break

			# Trigger update
			ctr += 1
			if ctr >= self.curr_velocity_publish_rate/self.update_rate:
				ctr = 0
				self.__trigger_update()

			# Check cartesian contact
			if any(self.robot_state.cartesian_contact):
				self.stop()
				rospy.logerr('Detected cartesian contact during velocity control loop.')
				break

			# Send joint velocity cmd
			v = Float64MultiArray()
			v.data = dq
			self.curr_velo_pub.publish(v)
			r.sleep()
		
		# Send zero joint velocity
		self.stop()
		self.cs.switch_controller('moveit')  # need to switch right away, otherwise communication reflex error?
		rospy.sleep(0.1)

		print("============ Press Enter to home...")
		raw_input()
		start_pose = [0.30, 0.0, 0.40, -0.9239554, 0.3824994, 0.0003046, 0.0007358]
		self.pc.goto_pose(start_pose, velocity=0.2)
		self.pc.set_gripper(0.1)
		self.pc.stop()

		# if not grasp_ret or self.ROBOT_ERROR_DETECTED:
		# 	rospy.logerr('Something went wrong, aborting this run')
		# 	if self.ROBOT_ERROR_DETECTED:
		# 		self.__recover_robot_from_error()
		# 	continue


	# def traj_time_scaling(self, startPos, endPos, numSteps):
	# 	trajPos = np.zeros((numSteps, 3))
	# 	for step in range(numSteps):
	# 		s = 3 * (1.0 * step / numSteps) ** 2 - 2 * (1.0 * step / numSteps) ** 3
	# 		trajPos[step] = (endPos-startPos)*s+startPos
	# 	return trajPos


	# def traj_tracking_vel(self, targetPos, targetQuat, posGain=20, velGain=5):
	# 	eePos, eeQuat = self._panda.get_ee()

	# 	eePosError = targetPos - eePos
	# 	# eeOrnError = log_rot(quat2rot(targetQuat)@(quat2rot(eeQuat).T))  # in spatial frame
	# 	eeOrnError = log_rot(quat2rot(targetQuat).dot((quat2rot(eeQuat).T)))  # in spatial frame

	# 	jointPoses = self._panda.get_arm_joints() + [0,0,0]  # add fingers
	# 	eeState = p.getLinkState(self._pandaId,
	# 						self._panda.pandaEndEffectorLinkIndex,
	# 						computeLinkVelocity=1,
	# 						computeForwardKinematics=1)
	# 	# Get the Jacobians for the CoM of the end-effector link. Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn. The localPosition is always defined in terms of the link frame coordinates.
	# 	zero_vec = [0.0] * len(jointPoses)
	# 	jac_t, jac_r = p.calculateJacobian(self._pandaId, 
	# 								 	self._panda.pandaEndEffectorLinkIndex, 
	# 								  	eeState[2], 
	# 								   	jointPoses, 
	# 									zero_vec, 
	# 									zero_vec)  # use localInertialFrameOrientation
	# 	jac_sp = full_jacob_pb(jac_t, jac_r)[:, :7]  # 6x10 -> 6x7, ignore last three columns
		
	# 	try:
	# 		# jointDot = np.linalg.pinv(jac_sp)@(np.hstack((posGain*eePosError, velGain*eeOrnError)).reshape(6,1))  # pseudo-inverse
	# 		jointDot = np.linalg.pinv(jac_sp).dot((np.hstack((posGain*eePosError, velGain*eeOrnError)).reshape(6,1)))  # pseudo-inverse
	# 	except np.linalg.LinAlgError:
	# 		jointDot = np.zeros((7,1))

	# 	return jointDot



if __name__ == '__main__':
	pushEnv = PushEnv()
	pushEnv.go()

	# def move_to_goal(self, pose_goal, horizon):
	# 	"""
	# 	Use resolved rate control
	# 	"""
	# 	# r = rospy.Rate(self.curr_velocity_publish_rate)
	# 	while not rospy.is_shutdown():
	# 		print(self.robot_state.O_T_EE)
	# 		time.sleep(0.1)
		
	# 	# Current joint angle, TODO
	# 	panda.q = np.array([0, -3, 0, -2.3, 0, 2, 0])

	# 	# Current pose, 4x4, TODO
	# 	wTe = panda.T

	# 	# Desired pose, apply ee offset?
	# 	wTep = np.copy(wTe)
	# 	wTep[0,3] += 0.2

	# 	while 1:
	# 		continue
	# 		# Desired end-effecor spatial velocity
	# 		v, arrived = rp.p_servo(wTe, wTep)
			
	# 		# Solve for the joint velocities dq
	# 		dq = np.matmul(np.linalg.pinv(panda.Je), v)
	
	# 		# Send command, TODO


	# 		# # Check
	# 		# current_pose = self.move_group.get_current_pose().pose
	# 		# return all_close(pose_goal, current_pose, 0.01)



