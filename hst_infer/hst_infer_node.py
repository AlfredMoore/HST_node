#! /usr/bin/env python

import os
import threading
import argparse
import cv2
import pickle
import numpy as np
from typing import Optional, TypedDict
from pathlib import Path

# import rosbag
import rclpy
from rclpy.node import Node
# import message_filters
# from rospy.numpy_msg import numpy_msg

# Message type
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Header, ColorRGBA, MultiArrayDimension
from geometry_msgs.msg import PoseArray, Pose, Quaternion, Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from skeleton_interfaces.msg import MultiHumanSkeleton, HumanSkeleton
from cv_bridge import CvBridge


from hst_infer.data_buffer import skeleton_buffer


HST_INFER_NODE = "hst"

MULTI_HUMAN_SKELETON_TOPIC = 'skeleton/data/multi_human'


class HST_infer_node(Node):
    def __init__(self,
                 rotate: int = cv2.ROTATE_90_CLOCKWISE,
                 ):
        
        super().__init__(HST_INFER_NODE)

        # Paramneters and Data buffer
        self.rotate = rotate

        self.skeleton_databuffer = skeleton_buffer()

        # Subscriber #######################################
        
        self._skeleton_sub = self.create_subscription(
            MultiHumanSkeleton, MULTI_HUMAN_SKELETON_TOPIC, self._skeleton_callback, 5
        )




    def _skeleton_callback(self, msg: MultiHumanSkeleton):

        self.skeleton_databuffer.receive_msg(msg)
        keypointATKD, human_pos_ATD, keypoint_mask_ATK = self.skeleton_databuffer.get_data_array()

        # debug ###
        print(keypointATKD,'\n', human_pos_ATD,'\n', keypoint_mask_ATK)
        exit()
        ###


def main(args=None):
    
    rclpy.init(args=args)
    node = HST_infer_node()
    rclpy.spin(node)


if __name__ == "__main__":
    main()