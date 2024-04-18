
import numpy as np

# Message type
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Header, ColorRGBA, MultiArrayDimension
from geometry_msgs.msg import PoseArray, Pose, Quaternion, Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge

from hst_infer.node_config import *
from hst_infer.utils.logger import logger
from hst_infer.utils.humanid_to_markerid import get_marker_id, marker_id_offset

def add_multihuman_pos_markers(multi_human_pos_ATD: np.ndarray,
                               multi_human_mask_AT: np.ndarray,
                               present_idx: int = HISTORY_LENGTH,
                               frame_id: str = STRETCH_BASE_FRAME,
                               ns: str = marker_id_offset.HST_NS.value,
                               ) -> Marker:
    """
    @multi_human_pos_ATD: np.ndarray (A,T,2)
    @multi_human_mask_AT: np.ndarray (A,T)
    @frame_id: str, marker's tf2 name
    return (points_list, colors_list)
    """
    marker_id = get_marker_id(offset=marker_id_offset.HST_HUMAN_TRAJ.value,
                              ns=ns,
                              )
    
    # Marker building
    marker = Marker()
    marker.header.frame_id = frame_id

    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.POINTS
    marker.action = Marker.ADD

    # marker.pose = Pose()
    marker.scale.x = 0.03
    marker.scale.y = 0.03

    A,T,D = multi_human_pos_ATD.shape       # [A,T,2]
    # marker_history_pos_ATD = multi_human_pos_ATD[:,:present_idx,:]
    # marker_history_mask_AT = multi_human_mask_AT[:,:present_idx]

    # marker_current_pos_AD = multi_human_pos_ATD[:,present_idx,:]
    # marker_current_mask_A = multi_human_mask_AT[:,present_idx]
    # assert marker_current_pos_AD.shape == (A,D)
    # assert marker_current_mask_A.shape == (A,)

    # marker_future_pos_ATD = multi_human_pos_ATD[:,present_idx+1:,:]
    # marker_future_mask_A = marker_current_mask_A        # we should not know the new human in the future

    points_list: list[Point] = list()
    colors_list: list[ColorRGBA] = list()
    current_color = ColorRGBA(r=0.5, g=0.0, b=0.0, a=0.7)
    future_color = ColorRGBA(r=0.0, g=0.0, b=0.5, a=0.7)

    for agent_idx in range(A):
        if multi_human_mask_AT[agent_idx,present_idx]:
            # current
            current_point = Point(
                x= multi_human_pos_ATD[agent_idx,present_idx,0].item(),
                y= multi_human_pos_ATD[agent_idx,present_idx,1].item(),
                z= float(0),
                )
            points_list.append(current_point)
            colors_list.append(current_color)

            # future
            for t_idx in range(present_idx+1,T):
                future_point = Point(
                    x = multi_human_pos_ATD[agent_idx,t_idx,0].item(),
                    y = multi_human_pos_ATD[agent_idx,t_idx,1].item(),
                    z = float(),
                )
                points_list.append(future_point)
                colors_list.append(future_color)

    marker.points = points_list
    marker.colors = colors_list
    
    return marker


def delete_multihuman_pos_markers(frame_id: str = STRETCH_BASE_FRAME,
                                      ns: str = marker_id_offset.HST_NS.value,
                                      ) -> Marker:
    marker_id = get_marker_id(offset=marker_id_offset.HST_HUMAN_TRAJ.value,
                              ns=ns,
                              )
    
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.DELETE

    return marker

