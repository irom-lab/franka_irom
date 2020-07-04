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
from nn_policy import PolicyNetGrasp

from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image


class CameraEnv(object):
	def __init__(self):
		super(CameraEnv, self).__init__()

		# Subscribe to depth topic
		# rospy.Subscriber('/depth/image_raw', Image, self.rect_depth_callback, queue_size=1)
		rospy.Subscriber('/depth/image_rect', Image, self.rect_depth_callback, queue_size=1)

		# Raw and rectified depth image from camera
		# self.depth_raw = None
		self.depth_rect = None

		# Processed depth image ready for inference
		self.depth_normalized = None
		self.depth_binned = None

		# Offset for 6 deg
		# self.all_offset = np.load('/home/allen/catkin_ws/src/franka_irom/grasp_depth_offset.npz')['all_offset']

		# Set up policy and load posterior
		self.actor = PolicyNetGrasp(input_num_chann=1,
				dim_mlp_append=0,
				num_mlp_output=5,
				out_cnn_dim=40,  # 64
				z_conv_dim=2,    # 7
				z_mlp_dim=8,     # 23
				img_size=128).to('cpu')
		# actor_path = '/home/allen/PAC-Imitation/model/grasp_bc_12/550.pt'
		actor_path = '/home/allen/PAC-Imitation/model/grasp_bc_13/800.pt'
		self.actor.load_state_dict(torch.load(
			actor_path, 	
			map_location=torch.device('cpu')))
		# training_details_dic_path = '/home/allen/PAC-Imitation/result/grasp_pac_6/train_details'
		training_details_dic_path = '/home/allen/PAC-Imitation/result/grasp_pac_8/train_details'
		training_details_dic = torch.load(training_details_dic_path)
		# _, _, self.mu_ps, logvar_ps, _, _, _ = training_details_dic['best_data']  # best bound
		best_emp_data = training_details_dic['best_emp_data']  # best emp, for grasp_pac_8
		self.mu_ps = best_emp_data[3]
		logvar_ps = best_emp_data[4]
		self.sigma_ps = torch.exp(0.5*logvar_ps)

	# def raw_depth_callback(self, msg):
	# 	self.depth_raw = ros_numpy.numpify(msg)  # 576x640, fov 65x75
	def rect_depth_callback(self, msg):
		self.depth_rect = ros_numpy.numpify(msg)  # 576x640, fov 65x75


	def capture_depth(self):

		print("============ Press Enter to record a sample depth image...")
		input()
		r = rospy.Rate(5)

		table_offset = 0.755
		normalizing_height = 0.20
		processed_height_radius = 200  # 576*0.2625/0.755
		processed_width_radius = 200

		self.depth_cropped = self.depth_rect \
				[288-processed_height_radius:288+processed_height_radius, \
				320-processed_width_radius:320+processed_width_radius]
		self.depth_normalized = ((table_offset-self.depth_cropped)/normalizing_height).clip(min=0.0, max=1.0)
		self.depth_normalized = np.rot90(table_offset-self.depth_cropped, k=1,  axes=(1,0))
		# self.depth_binned = np.rot90(bin_image(self.depth_normalized, 
		# 							  target_height=128, 
		# 							  target_width=128, 
		# 							  bin_average=False), k=1,  axes=(1,0))

		# plt.imshow(self.depth_raw, cmap='Greys', interpolation='nearest')
		# plt.show()
		np.savez('/home/allen/catkin_ws/src/franka_irom/mug_depth.npz', depth=self.depth_normalized)
		f, axarr = plt.subplots(1,2) 
		axarr[0].imshow(self.depth_rect, cmap='Greys', interpolation='nearest')
		axarr[1].imshow(self.depth_normalized, cmap='Greys', interpolation='nearest')
		axarr[1].scatter(processed_height_radius, processed_width_radius, s=10)
		plt.show()

		# Inference
		depth_nn = torch.from_numpy(self.depth_binned.copy()).float().unsqueeze(0).unsqueeze(0)
		zs = torch.normal(mean=self.mu_ps, std=self.sigma_ps).reshape(1, -1)
		pred = self.actor(depth_nn, zs).squeeze(0).detach().numpy()

		# Extract target pos
		target_pos = pred[:3]
		target_pos[:2] /= 20
		target_pos[0] += 0.5  # add offset

		# Extract target yaw
		target_yawEnc = pred[3:5]
		target_yaw = np.arctan2(target_yawEnc[0], target_yawEnc[1])
		target_yaw = wrap2halfPi(target_yaw)

		return 1


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
	cameraEnv.capture_depth()
