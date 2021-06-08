#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
sys.dont_write_bytecode = True

import numpy as np
from numpy import array
import matplotlib
import matplotlib.pyplot as plt
import time
import torch
import copy
import datetime

import rospy
import ros_numpy  # for converting Image to np array
from drpl_policy import FCN

from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from franka_irom.srv import GraspInfer, GraspInferResponse


def rotate_tensor(orig_tensor, theta):
	"""
	Rotate images clockwise
	"""
	affine_mat = np.array([[np.cos(theta), np.sin(theta), 0],
							[-np.sin(theta), np.cos(theta), 0]])
	affine_mat.shape = (2,3,1)
	affine_mat = torch.from_numpy(affine_mat).permute(2,0,1).float()
	flow_grid = torch.nn.functional.affine_grid(affine_mat, orig_tensor.size(), align_corners=False)
	return torch.nn.functional.grid_sample(orig_tensor, flow_grid, mode='nearest', align_corners=False)

class CameraEnv(object):
	def __init__(self):
		super(CameraEnv, self).__init__()

		# Subscribe to depth topic
		rospy.Subscriber('/depth/image_rect', Image, self.depth_callback, queue_size=1)

		# Service to graspEnv
		rospy.Service('~infer_grasp', GraspInfer, self.capture_depth)

		# Raw depth image from camera
		self.depth_rect = None
		self.depth_normalized = None
		self.depth_binned = None

		# Params
		self.img_size = 128
		self.inner_channels = 24
		num_theta = 6
		self.thetas = np.linspace(0, 1, num=num_theta, endpoint=False)*np.pi
		self.delta_z = 0.03	# from height at the pixel
		self.min_ee_z = 0.15  # EE height when fingers contact the table

		# Set up policy
		self.fcn = FCN(inner_channels=self.inner_channels, 
                 		out_channels=1,
                        img_size=self.img_size).to('cpu')
		# model_path = '/home/allen/data/wasserstein/3D/ad_grasp_8c2/policy_model/epoch_13000_step_5_acc_0.727.pt'
		# model_path = '/home/allen/data/wasserstein/3D/no_grasp_8c1/policy_model/epoch_14_step_3_acc_0.820.pt'
		model_path = '/home/allen/data/wasserstein/3D/dr_grasp_8c1/policy_model/epoch_13_step_6_acc_0.812.pt'
		self.fcn.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
		for _, param in self.fcn.named_parameters():
			param.requires_grad = False
		self.fcn.eval()

		# Pixel to xy
		pixel_xy_path = '/home/allen/data/wasserstein/pixel2xy'+str(self.img_size)+'.npz'
		self.pixel2xy_mat = np.load(pixel_xy_path)['pixel2xy']  # HxWx2


	def depth_callback(self, msg):
		self.depth_rect = ros_numpy.numpify(msg)  # 576x640, fov 65x75


	def capture_depth(self, req):

		# print("============ Press Enter to record a sample depth image...")
		# input()
		r = rospy.Rate(5)

		table_offset = 0.64	# 756
		processed_height_half = 64  # 576*0.075/0.644
		processed_width_half = 64
		target_dim = 128 	# assume square
		normalizing_height = 0.10

		self.depth_cropped = self.depth_rect \
				[288-processed_height_half:288+processed_height_half, \
				320-processed_width_half:320+processed_width_half]
		self.depth_normalized = ((table_offset-self.depth_cropped)/normalizing_height).clip(min=0.0, max=1.0)
		self.depth_final = np.rot90(self.depth_normalized, k=1, axes=(1,0))

		# Inference
		depth_infer_copy = self.depth_final.copy()
		depth_orig = torch.from_numpy(depth_infer_copy[np.newaxis,np.newaxis]).to('cpu')
		depth_rot_all = torch.empty((0,1,self.img_size,self.img_size))
		for theta in self.thetas:
			depth_rotated = rotate_tensor(depth_orig, theta=theta)
			depth_rot_all = torch.cat((depth_rot_all,depth_rotated))
		pred_infer = self.fcn(depth_rot_all).squeeze(1).detach().numpy()
		(theta_ind, px, py) = np.unravel_index(np.argmax(pred_infer), pred_infer.shape)
		x, y = self.pixel2xy_mat[py, px]  # actual pos, a bug
		theta = self.thetas[theta_ind]

		# Find the target z height
		z = depth_rot_all[theta_ind, 0, px, py]*normalizing_height
		z_target = max(0, z - self.delta_z) # clip
		z_target_ee = z_target + self.min_ee_z
		# print(z, z_target, z_target_ee)

		# Rotate into local frame
		xy_orig = array([[np.cos(theta), -np.sin(theta)],
					[np.sin(theta),np.cos(theta)]]).dot(array([[x-0.5],[y]]))
		xy_orig[0] += 0.5

		# Visualize
		f, axarr = plt.subplots(1,2)
		print(px, py, xy_orig[0], xy_orig[1], z_target_ee, theta)
		axarr[0].imshow(self.depth_rect, cmap='Greys', interpolation='nearest')
		axarr[1].imshow(self.depth_final, cmap='Greys', interpolation='nearest')
		axarr[1].scatter(px, 128-py, s=30)
		plt.show()

		# Respond
		res = GraspInferResponse()
		res.pos = Vector3(x=xy_orig[0], y=xy_orig[1], z=z_target_ee)
		res.yaw = theta

		# Save pose
		x = datetime.datetime.now()
		np.savez('/home/allen/data/drpl_grasp_0607/'+x.strftime("%X"), depth_rect=self.depth_rect, 
			depth_final=self.depth_final,
			target_pos=[xy_orig[0], xy_orig[1], z_target_ee],
			target_yaw=theta)

		return res

def wrap2halfPi(angle):  # assume input in [-pi, pi]
	if angle < -np.pi/2:
		return angle + np.pi 
	elif angle > np.pi/2:
		return angle - np.pi
	return angle

def wrap2pi(angle):
	if angle < -np.pi:
		return angle + 2*np.pi 
	elif angle > np.pi:
		return angle - 2*np.pi
	return angle

if __name__ == '__main__':
	seed = 0
	np.random.seed(seed)
	torch.manual_seed(seed)

	rospy.init_node('camera_env')
	cameraEnv = CameraEnv()
	rospy.spin()

	# all_offset = np.zeros((processed_height_radius*2))
	# for height_ind in range(processed_height_radius*2):  # account for weird differences in height on table, assume 1st pixel is table (not covered)
	# 	all_offset[height_ind] = np.mean(self.depth_cropped[height_ind,0:5])
	# 	depth_ini = self.depth_cropped[height_ind,0]
	# 	self.depth_cropped[height_ind] += (table_offset-depth_ini)
	# np.savez('/home/allen/catkin_ws/src/franka_irom/grasp_depth_offset.npz', all_offset=all_offset)
