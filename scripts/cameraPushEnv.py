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
from skimage.measure import block_reduce

import rospy
import ros_numpy  # for converting Image to np array
from nn_policy import PolicyNet

from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from franka_msgs.msg import FrankaState
from std_msgs.msg import Empty


class CameraEnv(object):
	def __init__(self):
		super(CameraEnv, self).__init__()

		# Subscribe to depth topic
		rospy.Subscriber('/depth/image_rect', Image, self.__depth_callback, queue_size=1)

		# Service to graspEnv
		self.action_pub = rospy.Publisher('push_infer', Vector3, queue_size=2)

		# Raw depth image from camera
		self.depth_rect = None
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

		# Sample epsilon
		eps = torch.randn_like(self.sigma_ps)
		self.z = (self.mu_ps + eps*self.sigma_ps).reshape(1,-1)
		# self.z = torch.randn_like(self.sigma_ps).reshape(1,-1)

		# Subscribe to robot state for ee pos
		self.robot_state = None
		rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.__robot_state_callback, queue_size=1)
		self.ee_history = np.zeros((10))
		self.eeX = 0.0
		self.eeY = 0.0

		# Set up update with inference
		self.update_rate = 5.0  # for inference
		update_topic_name = '~/update_push'
		self.update_pub = rospy.Publisher(update_topic_name, Empty, queue_size=1)
		rospy.Subscriber(update_topic_name, Empty, self.__update_callback, queue_size=1)

	def __robot_state_callback(self, msg):
		self.robot_state = msg
		self.eeX = 3*(self.robot_state.O_T_EE[-4]-0.35)
		self.eeY = 6*self.robot_state.O_T_EE[-3]

	def __depth_callback(self, msg):
		self.depth_rect = ros_numpy.numpify(msg)  # 576x640, fov 65x75

	def __trigger_update(self):
		# Let ROS handle the threading for me.
		self.update_pub.publish(Empty())
	def __update_callback(self, msg):
		self.ee_history[2:10] = self.ee_history[0:8]
		self.ee_history[0] = self.eeX
		self.ee_history[1] = self.eeY

	def infer(self):

		table_offset = 0.667
		normalizing_height = 0.12
		processed_height_radius = 225
		processed_width_radius = 225

		r = rospy.Rate(5)
		ctr = 0
		start_time = time.time()
		while not rospy.is_shutdown():
			if self.depth_rect is not None:
				# print(self.ee_history)
				self.depth_cropped = self.depth_rect \
						[288-processed_height_radius: \
						288+processed_height_radius, \
						320-processed_width_radius: \
						320+processed_width_radius]
				self.depth_normalized = ((table_offset-self.depth_cropped)/normalizing_height).clip(min=0.0, max=1.0)
				self.depth_binned = block_reduce(self.depth_normalized, 
												block_size=(3,3),
												func=np.mean, 
												cval=0.0)
				self.depth_binned = np.rot90(self.depth_binned, 
											k=1, axes=(1,0))
				self.depth_binned = np.hstack((np.zeros((150,5)), 
											   self.depth_binned[:,:145]))
				self.depth_binned[self.depth_binned >= 0.90] = 0.0

				# f, axarr = plt.subplots(1,2)
				# axarr[0].imshow(self.depth_cropped, cmap='Greys', interpolation='nearest')
				# axarr[1].imshow(self.depth_binned, cmap='Greys', interpolation='nearest')
				# axarr[1].scatter(x=75,y=75,s=10)
				# plt.show()

				# Update wTep in 5Hz
				self.__trigger_update()

				# Inference
				depth_nn = torch.from_numpy(self.depth_binned.copy()).float().unsqueeze(0).unsqueeze(0)
				print(self.ee_history)
				action_pred = self.actor(depth_nn, 
										self.z, 
						torch.from_numpy(self.ee_history).float().unsqueeze(0)).squeeze(0)
				action_pred[0] /= 50  # scaling
				action_pred[1] /= 30  # scaling
				if action_pred[0] > 0.02:
					action_pred[0] = 0.02
				print(action_pred)

				# Publish inferred action
				self.action_pub.publish(Vector3(x=action_pred[0], 
												y=action_pred[1], 
												z=0.0))

				# print(time.time()-start_time)
				# start_time = time.time()
			r.sleep()


if __name__ == '__main__':
	seed = 1000
	np.random.seed(seed)
	torch.manual_seed(seed)

	rospy.init_node('camera_env')
	cameraEnv = CameraEnv()
	cameraEnv.infer()
