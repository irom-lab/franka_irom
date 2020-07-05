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
from nn_policy import PolicyNet

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

		# Set up policy and load posterior
		self.actor = PolicyNet(input_num_chann=1,
				dim_mlp_append=10,
				num_mlp_output=2,
				out_cnn_dim=40,
				z_conv_dim=1,
				z_mlp_dim=4,
				img_size=150).to('cpu')
		actor_path = '/home/allen/PAC-Imitation/model/push_bc_30/bc_actor_20.pt'
		self.actor.load_state_dict(torch.load(
			actor_path, 	
			map_location=torch.device('cpu')))
		training_details_dic_path = '/home/allen/PAC-Imitation/result/push_pac_26/train_details'
		training_details_dic = torch.load(training_details_dic_path)
		best_emp_data = training_details_dic['best_emp_data']
		self.mu_ps = best_emp_data[3]
		logvar_ps = best_emp_data[4]
		self.sigma_ps = torch.exp(0.5*logvar_ps)

		# 
		self.ee_history = torch.zeros((10))


	def rect_depth_callback(self, msg):
		self.depth_rect = ros_numpy.numpify(msg)  # 576x640, fov 65x75


	def capture_depth(self):

		print("============ Press Enter to record a sample depth image...")
		input()
		r = rospy.Rate(5)

		table_offset = 0.76
		normalizing_height = 0.12
		processed_height_radius = 190  # 576*0.2625/0.755
		processed_width_radius = 190

		self.depth_cropped = self.depth_rect \
				[288-processed_height_radius:288+processed_height_radius, \
				320-processed_width_radius:320+processed_width_radius]
		self.depth_normalized = ((table_offset-self.depth_cropped)/normalizing_height).clip(min=0.0, max=1.0)
		# self.depth_normalized = np.rot90(table_offset-self.depth_cropped, k=1,  axes=(1,0))
		self.depth_binned = np.rot90(bin_image(self.depth_normalized, 
									  target_height=150, 
									  target_width=150, 
									  bin_average=False), k=1,  axes=(1,0))
		self.depth_binned[self.depth_binned >= 0.99] = 0.0

		# Visualize
		f, axarr = plt.subplots(1,2) 
		axarr[0].imshow(self.depth_rect, cmap='Greys', interpolation='nearest')
		axarr[1].imshow(self.depth_binned, cmap='Greys', interpolation='nearest')
		# axarr[1].scatter(processed_height_radius, processed_width_radius, s=10)
		plt.show()

		# Scaling in xy
		action_scale = array([50, 50])
		eePose_scale = array([3, 6])
		eePos, eeOrn = self.pandaEnv.get_ee()
		eePos -= array([0.35,0.0,0.18])
		eePos[:2] *= eePose_scale
		self.ee_history[2:10] = self.ee_history[0:8]
		self.ee_history[0:2] = self.eePos[:2]

		# Inference
		depth_nn = torch.from_numpy(self.depth_binned.copy()).float().unsqueeze(0).unsqueeze(0)
		# TODO: use single z, as req
		zs = torch.normal(mean=self.mu_ps, std=self.sigma_ps).reshape(1, -1)
		action_pred = self.actor(depth_nn, zs).squeeze(0).detach().numpy()

		# Extract
		action_pred /= action_scale[:2]
		posAction = np.hstack((action_pred[:2], 0))
		eulerAction = [0,0,0]

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
	cameraEnv.capture_depth()
