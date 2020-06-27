#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
sys.dont_write_bytecode = True

import rospy
import numpy as np
from numpy import array
import matplotlib
import matplotlib.pyplot as plt
import ros_numpy

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image


class CameraEnv(object):
	def __init__(self):
		super(CameraEnv, self).__init__()

		# Initialize rospy node
		rospy.init_node('camera_env', anonymous=True)

		# Subscribe to depth topic
		rospy.Subscriber('/depth/image_raw', Image, self.depth_callback, queue_size=1)

		# Raw depth image from camera
		self.depth_raw = None

		# Processed depth image ready for inference
		self.depth_processed = None


	def depth_callback(self, msg):
		self.depth_raw = ros_numpy.numpify(msg)

	def capture_depth(self):

		# self.pcl_sub.unregister()
		
		print("============ Press Enter to record a sample depth image...")
		raw_input()
		r = rospy.Rate(5)
		while 1:
			print(type(self.depth_raw))
			plt.imshow(self.depth_raw, cmap='Greys', interpolation='nearest')
			plt.show()

			r.sleep()
			while 1:
				continue

		return 1


if __name__ == '__main__':
	cameraEnv = CameraEnv()
	cameraEnv.capture_depth()
