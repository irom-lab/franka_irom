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

import rospy
import ros_numpy  # for converting Image to np array
from drpl_policy import FCN

from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image


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
		# rospy.Subscriber('/depth/image_raw', Image, self.rect_depth_callback, queue_size=1)
		rospy.Subscriber('/depth/image_rect', Image, self.rect_depth_callback, queue_size=1)

		# Rectified depth image from camera
		self.depth_rect = None
		self.depth_normalized = None
		self.depth_final = None

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
		model_path = '/home/allen/data/wasserstein/3D/ad_grasp_8c2/policy_model/epoch_13000_step_5_acc_0.727.pt'
		self.fcn.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
		for _, param in self.fcn.named_parameters():
			param.requires_grad = False
		self.fcn.eval()

		# Pixel to xy
		pixel_xy_path = '/home/allen/data/wasserstein/pixel2xy'+str(self.img_size)+'.npz'
		self.pixel2xy_mat = np.load(pixel_xy_path)['pixel2xy']  # HxWx2


	def rect_depth_callback(self, msg):
		self.depth_rect = ros_numpy.numpify(msg)  # 576x640, fov 65x75


	def capture_depth(self):

		print("============ Press Enter to record a sample depth image...")
		input()
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
		# self.depth_final = np.rot90(bin_image(self.depth_normalized, 
		# 								target_height=target_dim, 
		# 								target_width=target_dim, 
		# 								bin_average=False), k=1,  axes=(1,0))
		# self.depth_final[self.depth_final >= 0.99] = 0.0

		# Visualize
		f, axarr = plt.subplots(1,2) 
		axarr[0].imshow(self.depth_rect, cmap='Greys', interpolation='nearest')
		axarr[1].imshow(self.depth_final, cmap='Greys', interpolation='nearest')
		# axarr[1].scatter(processed_height_radius, processed_width_radius, s=10)
		plt.show()

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
		print(f'Target x {xy_orig[0]}, y {xy_orig[1]}, z {z_target_ee}')
		return 1


def bin_image(image_raw, target_height, target_width, bin_average=True):
	"""
	Assume square image out
	"""
	image_out = np.zeros((target_height, target_width))
	raw_height, raw_width = image_raw.shape

	start_time = time.time()
	for height_ind in range(target_height):
		for width_ind in range(target_width):
			# find the top left pixel involving in raw image
			first_height_pixel = np.floor(height_ind/target_height*raw_height).astype('int')
			end_height_pixel = np.ceil(height_ind/target_height*raw_height).astype('int')+1
			first_width_pixel = np.floor(width_ind/target_width*raw_width).astype('int')
			end_width_pixel = np.ceil(width_ind/target_width*raw_width).astype('int')+1

			if bin_average:
				image_out[height_ind, width_ind] = np.mean(image_raw[first_height_pixel:end_height_pixel, \
					first_width_pixel:end_width_pixel])
			else:  # use max
				image_out[height_ind, width_ind] = np.max(image_raw[first_height_pixel:end_height_pixel, \
					first_width_pixel:end_width_pixel])
	print('Time used:', time.time()-start_time)
	return image_out


if __name__ == '__main__':
	rospy.init_node('camera_env')
	cameraEnv = CameraEnv()
	while 1:
		cameraEnv.capture_depth()
