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
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from moveit_commander import MoveGroupCommander
from actionlib_msgs.msg import GoalStatusArray
import tf
import PyKDL

from geometry_msgs.msg import Vector3, Quaternion, TransformStamped, Twist, Pose
from tf_conversions import posemath
from rviz_tools import RvizMarkers
import ropy as rp


class GraspEnv(object):
    def __init__(self):
        super(GraspEnv, self).__init__()

        # Initialize rospy node
        rospy.init_node('grasp_env', anonymous=True)

		# Set up joint velocity controller
		self.joint_velocity_pub = rospy.Publisher('/cartesian_velocity_controller/cartesian_velocity', Twist, queue_size=1)

		# Initialize ropy
		self.panda = rp.Panda()

		# Initialze gripper control

		# Subscribe to robot state
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.__robot_state_callback, queue_size=1)

        # For visualization in Rviz
        self.markers = RvizMarkers('/panda_link0', 'visualization_marker')


        # # TF listener and broadcaster
        # self.listener = tf.TransformListener()
        # self.br = tf.TransformBroadcaster()


    def move_to_goal(self, pose_goal, horizon):
		"""
		Use resolved rate control
		"""
        # r = rospy.Rate(self.curr_velocity_publish_rate)

		# Current joint angle, TODO
		panda.q = np.array([0, -3, 0, -2.3, 0, 2, 0])

		# Current pose, 4x4, TODO
		wTe = panda.T

		# Desired pose, apply ee offset?
		wTep = np.copy(wTe)
		wTep[0,3] += 0.2

		# Desired end-effecor spatial velocity
		v, arrived = rp.p_servo(wTe, wTep)
		
		# Solve for the joint velocities dq
		dq = np.matmul(np.linalg.pinv(panda.Je), v)
  
		# Send command, TODO
  
        # # Check
        # current_pose = self.move_group.get_current_pose().pose
        # return all_close(pose_goal, current_pose, 0.01)


	def traj_time_scaling(self, startPos, endPos, numSteps):
		trajPos = np.zeros((numSteps, 3))
		for step in range(numSteps):
			s = 3 * (1.0 * step / numSteps) ** 2 - 2 * (1.0 * step / numSteps) ** 3
			trajPos[step] = (endPos-startPos)*s+startPos
		return trajPos


	def traj_tracking_vel(self, targetPos, targetQuat, posGain=20, velGain=5):
		eePos, eeQuat = self._panda.get_ee()

		eePosError = targetPos - eePos
		# eeOrnError = log_rot(quat2rot(targetQuat)@(quat2rot(eeQuat).T))  # in spatial frame
		eeOrnError = log_rot(quat2rot(targetQuat).dot((quat2rot(eeQuat).T)))  # in spatial frame

		jointPoses = self._panda.get_arm_joints() + [0,0,0]  # add fingers
		eeState = p.getLinkState(self._pandaId,
							self._panda.pandaEndEffectorLinkIndex,
							computeLinkVelocity=1,
							computeForwardKinematics=1)
		# Get the Jacobians for the CoM of the end-effector link. Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn. The localPosition is always defined in terms of the link frame coordinates.
		zero_vec = [0.0] * len(jointPoses)
		jac_t, jac_r = p.calculateJacobian(self._pandaId, 
									 	self._panda.pandaEndEffectorLinkIndex, 
									  	eeState[2], 
									   	jointPoses, 
										zero_vec, 
										zero_vec)  # use localInertialFrameOrientation
		jac_sp = full_jacob_pb(jac_t, jac_r)[:, :7]  # 6x10 -> 6x7, ignore last three columns
		
		try:
			# jointDot = np.linalg.pinv(jac_sp)@(np.hstack((posGain*eePosError, velGain*eeOrnError)).reshape(6,1))  # pseudo-inverse
			jointDot = np.linalg.pinv(jac_sp).dot((np.hstack((posGain*eePosError, velGain*eeOrnError)).reshape(6,1)))  # pseudo-inverse
		except np.linalg.LinAlgError:
			jointDot = np.zeros((7,1))

		return jointDot


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
