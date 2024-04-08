"""
transform
"""
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import tf2_ros
from tf2_geometry_msgs import do_transform_point
# from geometry_msgs.msg import PointStamped

def tf2_point_transformation(self, points: list[], source_frame: str, target_frame: str) -> list[]:
    


def tf2_array_transformation(self, pc: np.ndarray, source_frame: str, target_frame: str) -> np.ndarray:


def math_transformation(self, pc: np.ndarray, source_coor: np.ndarray, target_coor: np.ndarray) -> np.ndarray:
    assert Exception, f"function math_transformation Not implemented function"