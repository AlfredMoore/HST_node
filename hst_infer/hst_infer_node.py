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
from hst_infer.utils.logger import logger
from hst_infer.utils.run_once import run_once

HST_INFER_NODE = "hst"

MULTI_HUMAN_SKELETON_TOPIC = 'skeleton/data/multi_human'


class HST_infer_node(Node):
    def __init__(self,
                 transform_camera2robot: tuple[str, str],
                 ):
        
        super().__init__(HST_INFER_NODE)

        # Paramneters and Data buffer
        self.rotate = rotate

        self.skeleton_databuffer = skeleton_buffer()

        # Subscriber #######################################
        
        self._skeleton_sub = self.create_subscription(
            MultiHumanSkeleton, MULTI_HUMAN_SKELETON_TOPIC, self._skeleton_callback, 5
        )
        
        logger.info(
            f"Node Name: {HST_INFER_NODE} \n \
            receive message from {MULTI_HUMAN_SKELETON_TOPIC} \n \
        ")


    def _skeleton_callback(self, msg: MultiHumanSkeleton):

        self._get_start_time(header=msg.header)
        # t2 = self.get_clock().now()

        self.skeleton_databuffer.receive_msg(msg)
        keypointATKD, human_pos_ATD, keypoint_mask_ATK = self.skeleton_databuffer.get_data_array()

        # debug ###
        # logger.debug(f"Buffer depth:{len(self.skeleton_databuffer)}\n")
        # logger.debug(f"\nget_image:{msg.header.stamp}\nreceive_skeleton:{t2}\nafter_databuffer:{self.get_clock().now()}")
        # logger.debug(f"keypointATKD nonzero:{np.nonzero(keypointATKD)}\n \
        #       human position nonzero:{np.nonzero(human_pos_ATD)}\n \
        #       mask sparse:{np.nonzero(keypoint_mask_ATK)}\n \
        #       ")
        # exit()
        ##

    @run_once
    def _get_start_time(self, header: Header):
        self._get_start_time = header.stamp



    def tf2_transformation(self, pc: np.ndarray, source_frame: str, target_frame: str) -> np.ndarray:
        assert Exception, "Not implemented function"


    def math_transformation(self, pc: np.ndarray, source_coor: np.ndarray, target_coor: np.ndarray) -> np.ndarray:
        pass


def main(args=None):
    
    rclpy.init(args=args)
    node = HST_infer_node()
    rclpy.spin(node)


if __name__ == "__main__":
    main()