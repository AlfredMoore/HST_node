#! /usr/bin/env python

import os
import threading
import argparse
import cv2
import pickle
import numpy as np
from typing import Optional, TypedDict
from pathlib import Path
import tensorflow as tf

# import rosbag
import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
# import message_filters
# from rospy.numpy_msg import numpy_msg

import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_point

# Message type
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Header, ColorRGBA, MultiArrayDimension
from geometry_msgs.msg import PoseArray, Pose, Quaternion, Point, PointStamped, Transform, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from skeleton_interfaces.msg import MultiHumanSkeleton, HumanSkeleton
from cv_bridge import CvBridge

# inference pipeline
from hst_infer.data_buffer import skeleton_buffer
from hst_infer.utils.logger import logger
from hst_infer.utils.run_once import run_once
from hst_infer.utils.transform import transformstamped_to_tr
from hst_infer.utils.rviz2_marker import add_multihuman_pos_markers, delete_multihuman_pos_markers
from hst_infer.node_config import *
from hst_infer.human_scene_transformer.model import model as hst_model
from hst_infer.human_scene_transformer.config import hst_config
# HST model
from hst_infer.human_scene_transformer import infer
from hst_infer.utils import keypoints
# from human_scene_transformer import run

class HST_infer_node(Node):
    def __init__(self,
                 ):
        
        super().__init__(HST_INFER_NODE)
        self.skeleton_databuffer = skeleton_buffer(maxlen=hst_config.hst_model_param.num_history_steps)

        # Subscriber #######################################
        ### human skeleton msg 
        self._skeleton_sub = self.create_subscription(
            MultiHumanSkeleton, MULTI_HUMAN_SKELETON_TOPIC, self._skeleton_callback, 5
        )
        ### tf2
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        ### hst checkpoint
        self.model: hst_model.HumanTrajectorySceneTransformer = infer.init_model()
        _checkpoint_path = HST_CKPT_PATH.as_posix()
        _param_path = NETWORK_PARAM_PATH.as_posix()
        checkpoint_mngr = tf.train.Checkpoint(model=self.model)
        checkpoint_mngr.restore(_checkpoint_path).assert_existing_objects_matched()

        #  Publisher ######################################
        if RVIZ_HST:
            self._traj_marker_pub = self.create_publisher(
                Marker, RVIZ_HST_TOPIC, 5
            )
        # logger
        logger.info(
            f"\nNode Name: {HST_INFER_NODE} \n \
            receive message from {MULTI_HUMAN_SKELETON_TOPIC} \n \
            HST checkpoint: {_checkpoint_path} \n \
            GPU: { len(tf.config.list_physical_devices('GPU')) } \
        ")
        logger.debug(f"tensorflow eager mode: {tf.executing_eagerly()}")
        
    
    def _skeleton_callback(self, msg: MultiHumanSkeleton):

        self._get_start_time(header=msg.header)
        # t2 = self.get_clock().now()

        ### human position, human skeleton
        self.skeleton_databuffer.receive_msg(msg)
        keypointATKD, human_pos_ATD, keypoint_mask_ATK = self.skeleton_databuffer.get_data_array()
        A,T,K,D = keypointATKD.shape

        current_human_id_set = self.skeleton_databuffer.get_current_multihumanID_set()
        if len(current_human_id_set) == 0:
            # TODO: delete all rviz
            if RVIZ_HST:
                marker = delete_multihuman_pos_markers(frame_id=STRETCH_BASE_FRAME,)

        else:
            t, r = self.tf2_array_transformation(source_frame=CAMERA_FRAME, target_frame=STRETCH_BASE_FRAME)
            if msg.header.frame_id != STRETCH_BASE_FRAME:
                logger.warning(f"The frame {msg.header.frame_id} is not {STRETCH_BASE_FRAME}, please check the skeleton extractor")
                try:
                    # np dot ##########
                    keypoint_stretch = r @ keypointATKD[...,np.newaxis]
                    human_pos_stretch = r @ human_pos_ATD[...,np.newaxis]
                    keypointATKD_stretch = np.squeeze(keypoint_stretch, axis=-1) + t
                    human_pos_ATD_stretch = np.squeeze(human_pos_stretch, axis=-1) + t
                except:
                    # einsum ##########
                    # [3,3] @ [3,1] = [3,1] but einsum is slower
                    keypointATKD_stretch = np.einsum("ji,...i->...j", r, keypointATKD) + t
                    human_pos_ATD_stretch = np.einsum("ji,...i->...j", r, human_pos_ATD) + t
            else:
                human_pos_ATD_stretch = human_pos_ATD
                keypointATKD_stretch = keypointATKD

            ### robot position
            robot_pos_TD = np.zeros((HISTORY_LENGTH, DIM_XYZ))

            # To HST tensor input
            agent_position = tf.convert_to_tensor(human_pos_ATD_stretch[np.newaxis,...,:2])     # 2D position
            agent_keypoint = tf.convert_to_tensor(
                keypoints.human_keypoints.map_yolo_to_hst(keypoints_ATKD= keypointATKD_stretch,
                                                        keypoint_mask_ATK= keypoint_mask_ATK,
                                                        keypoint_center_ATD= human_pos_ATD_stretch,)
            )

            # no agent orientation data
            agent_orientation = tf.convert_to_tensor(np.full((1,1,hst_config.hst_dataset_param.num_steps,1),
                                                            np.nan, dtype=float))
            # TODO: robot remains static
            robot_position = tf.convert_to_tensor(np.zeros((1,hst_config.hst_dataset_param.num_steps,3)))

            # input dict
            # agents/keypoints: [B,A,T,33*3]
            # agents/position: [B,A,T,2]
            # agents/orientation: [1,1,T,1] useless
            # robot/position: [B,T,3]
            input_batch = {'agents/keypoints': agent_keypoint,
                        'agents/position': agent_position,
                        'agents/orientation': agent_orientation,
                        'robot/position': robot_position,}
            # logger.debug(f'shape:\n keypoints {agent_keypoint.shape}, position {agent_position.shape}, {robot_position.shape}')

            full_pred, output_batch = self.model(input_batch, training=False)
            agent_position_pred = full_pred['agents/position']
            # logger.debug(f"{type(agent_position_pred)}, {agent_position_pred.shape}")

            try:
                agent_position_pred = agent_position_pred.numpy()
            except:
                logger.error(f"cannot convert agent position into numpy")

            
            # current_human_pred = agent_position_pred
            # window_humanID_id2idx = self.skeleton_databuffer.id2idx_in_window
            # window_humanID_list = self.skeleton_databuffer.humanID_in_window

            if RVIZ_HST:
                multi_human_pos_ATD = np.squeeze(agent_position_pred, axis=0)       # remove batch 1
                multi_human_mask_AT = np.any(keypoint_mask_ATK, axis=-1)
                assert multi_human_mask_AT.shape == (A,T)

                marker = add_multihuman_pos_markers(
                    multi_human_pos_ATD=multi_human_pos_ATD,
                    multi_human_mask_AT=multi_human_mask_AT,
                    present_idx=HISTORY_LENGTH,
                    frame_id=STRETCH_BASE_FRAME,
                    ns=HST_INFER_NODE,
                )
                
        if RVIZ_HST:
            self._traj_marker_pub.publish(marker)



            # debug ###
            # logger.debug(f"Buffer depth:{len(self.skeleton_databuffer)}\n")
            # logger.debug(f"\nget_image:{msg.header.stamp}\nreceive_skeleton:{t2}\nafter_databuffer:{self.get_clock().now()}")
            # logger.debug(f"keypointATKD nonzero:{np.nonzero(keypointATKD)}\n \
            #       human position nonzero:{np.nonzero(human_pos_ATD)}\n \
            #       mask sparse:{np.nonzero(keypoint_mask_ATK)}\n \
            #       ")
            # exit()
            ##



    def tf2_array_transformation(self, source_frame: str, target_frame: str):
        try:
            # P^b = T_a^b @ P^a, T_a^b means b wrt a transformation
            transformation = self._tf_buffer.lookup_transform(target_frame=target_frame, 
                                                              source_frame=source_frame,
                                                              time=rclpy.time.Time(seconds=0, nanoseconds=0),
                                                              timeout=rclpy.duration.Duration(seconds=0, nanoseconds=int(0.5e9)),
                                                              )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            logger.exception(f'Unable to find the transformation from {source_frame} to {target_frame}')

        t,r = transformstamped_to_tr(transformation)
        return t,r


    @run_once
    def _get_start_time(self, header: Header):
        self._start_time = header.stamp
        logger.info(f"hst started at {self._start_time}")




        


def main(args=None):
    
    rclpy.init(args=args)
    node = HST_infer_node()
    rclpy.spin(node)


if __name__ == "__main__":
    main()