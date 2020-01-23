#!/usr/bin/env python3

import rospy
import PyKDL
import open3d as o3d
import numpy as np

from geometry_msgs.msg import Vector3, Quaternion, TransformStamped
from sensor_msgs.msg import PointCloud2
from pcl_conversion import convertCloudFromRosToOpen3d

import tf2_ros
import tf2_py as tf2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

class PerceptionInterface(object):
    def __init__(self):
        super(PerceptionInterface, self).__init__()

        # Subscribe to point cloud topic
        self.pcl_sub = None

        # Point cloud array
        self.pcl_count = 0
        self.pcl_count_thres = 1
        self.pcl = []

        # Post-processing open3d point cloud
        self.pcl_converted = None

        self.point_min_z = 0.01 # TODO:

        self.target_num_point = 1024

    def pcl_callback(self, msg):
        self.pcl += [msg]
        self.pcl_count += 1

    def capture_pcl(self, trans):
        
        # Save pcl multiple times
        self.pcl_sub = rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.pcl_callback)
        while self.pcl_count < self.pcl_count_thres:
            continue
        self.pcl_sub.unregister()

        # Convert point cloud to base frame
        pcl_global = do_transform_cloud(self.pcl[-1], trans)

        # Convert to open3d format
        self.pcl_converted = convertCloudFromRosToOpen3d(pcl_global, self.point_min_z)

        # Downsample
        self.pcl_converted = self.downsamplePCL(self.pcl_converted)

        # Estimate normals
        self.pcl_converted.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.01, max_nn=30))  # using points in 1cm radius
        self.pcl_converted.orient_normals_towards_camera_location(camera_location=[0,0,0])
        
        # Visualize
        o3d.visualization.draw_geometries([self.pcl_converted])
        num_points = len(self.pcl_converted.points)
        print('Number of points: ', num_points)

        # Choose point
        contact_point_idx_list = []
        for idx, point in enumerate(self.pcl_converted.points):
            if point[2] > 0.05:
                contact_point_idx_list += [idx]

        return self.pcl_converted, contact_point_idx_list

    def downsamplePCL(self, pcd):
        voxel_size = 0.001

        while 1:
            numPointBefore = len(pcd.points)
            down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            numPointAfter = len(down_pcd.points)

            if numPointAfter < self.target_num_point:
                indChosen = list(np.random.choice(np.arange(numPointBefore), self.target_num_point, replace=False))
                points = np.asarray(pcd.points)[indChosen]

                output = o3d.geometry.PointCloud()
                output.points = o3d.utility.Vector3dVector(points)
                return output

            voxel_size *= 1.1  # increment voxel size
