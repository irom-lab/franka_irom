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
# import moveit_commander
# from moveit_commander.conversions import pose_to_list
# from moveit_commander import MoveGroupCommander
# import moveit_msgs.msg
from std_msgs.msg import String
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import Vector3, Quaternion, TransformStamped, Twist, Pose
from franka_msgs.msg import FrankaState, Errors as FrankaErrors
from tf_conversions import posemath
from rviz_tools import RvizMarkers

from franka_irom_controllers.panda_commander import PandaCommander


class GraspEnv(object):
	def __init__(self):
		super(GraspEnv, self).__init__()

		# Initialize rospy node
		rospy.init_node('grasp_env', anonymous=True)

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
        self.BAD_UPDATE = False

		# # TF listener and broadcaster
		# self.listener = tf.TransformListener()
		# self.br = tf.TransformBroadcaster()


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


	def go(self):
		
		print("============ Press Enter to move to initial pose...")
		raw_input()
		start_pose = [0.20, -0.30, 0.40, -0.9239554, 0.3824994, 0.0003046, 0.0007358]
		self.pc.goto_pose(start_pose, velocity=0.2)
		self.pc.set_gripper(0.1)

		#   print("============ Press Enter to take depth image and infer grasp pose...")
		#   print("============ Press Enter to execute pose...")
		#   print("============ Press Enter to reach down...")
		#   print("============ Press Enter to grasp...")
		#   print("============ Press Enter to lift...")

		print("============ Press Enter to home...")
		raw_input()
		start_pose = [0.30, 0.0, 0.40, -0.9239554, 0.3824994, 0.0003046, 0.0007358]
		self.pc.goto_pose(start_pose, velocity=0.2)
		self.pc.set_gripper(0.1)

		# if not grasp_ret or self.ROBOT_ERROR_DETECTED:
		# 	rospy.logerr('Something went wrong, aborting this run')
		# 	if self.ROBOT_ERROR_DETECTED:
		# 		self.__recover_robot_from_error()
		# 	continue


if __name__ == '__main__':
	graspEnv = GraspEnv()
	graspEnv.go()

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
