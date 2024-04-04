
import numpy as np
from collections import deque

from geometry_msgs.msg import PoseArray, Pose, Quaternion, Point, Point32, PointStamped

from skeleton_interfaces.msg import MultiHumanSkeleton, HumanSkeleton
from hst_infer.utils import skeleton2msg

KEYPOINT_NUM = 17
DIM_XYZ = 3
MAX_AGENT_NUM = 19

class skeleton_buffer():
    def __init__(self, maxlen:int = 12):
        
        self.maxlen = maxlen
        self.buffer: deque[tuple[list[HumanSkeleton], set[int]]] = deque(maxlen=maxlen)
        self.existing_id = dict()       # dict [ id: index ]

        # self.empty: bool = True
        # self.full: bool = False

    def receive_msg(self, data: MultiHumanSkeleton):
        """
        append (list[HumanSkeleton], id_set) to the deque

        if non-exsiting skeleton, put (list(),set()) into buffer
        """
        header = data.header
        multihuman_data: list[HumanSkeleton] = data.multi_human_skeleton
        multihumanID_set = set()

        if multihuman_data == list():
            self.buffer.append((multihuman_data, set()))
            return

        else:
            for human_data in multihuman_data:
                multihumanID_set.add(human_data.human_id)

            self.buffer.append((multihuman_data, multihumanID_set))
            return


    def get_data_array(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        read the buffer and return keypoints_ATKD, human_center_ATD, keypoint_mask_ATK
        """

        msg_seq: list[tuple[list[HumanSkeleton], set[int]]] = list(self.buffer)
        
        keypoints_ATKD: np.ndarray = np.zeros(MAX_AGENT_NUM, self.maxlen, KEYPOINT_NUM, DIM_XYZ)
        center_ATD: np.ndarray = np.zeros(MAX_AGENT_NUM, self.maxlen, DIM_XYZ)
        mask_ATK: np.ndarray = np.zeros(MAX_AGENT_NUM, self.maxlen, KEYPOINT_NUM)

        # TODO: convert msg into array, put the array in the large array
        # x x x x m m m 
        # x x x x m m m
        # x x x x x x x 

        # all the exsisting human in the buffer
        allhumanID_set = set()
        for _, multihumanID_set in msg_seq:
            allhumanID_set = allhumanID_set | multihumanID_set      # O(m+n)
        allhuamnID_list = list(allhumanID_set)

        # allocate the agent positions array
        id2idx = dict()
        for idx, id in enumerate(allhuamnID_list, start=0):
            id2idx[id] = idx

        t_start = self.maxlen - len(msg_seq)      # T axis
        # external loop for time
        for t_idx, (multihuman_data, _) in enumerate(msg_seq, start=t_start):
            # internal loop for agents, A axis
            for human_data in multihuman_data:
                id = human_data.human_id
                a_idx = id2idx[id]
                geo_center = skeleton2msg.point32_to_np_vector(human_data.human_center) 
                keypoint_list: list[Point32] = human_data.keypoint_data

                for k_idx in range(KEYPOINT_NUM):
                    keypoint_vector = skeleton2msg.point32_to_np_vector(keypoint_list[k_idx])
                    
                    keypoints_ATKD[a_idx, t_idx, k_idx, :] = keypoint_vector
                    mask_ATK[a_idx, t_idx, k_idx] = human_data.keypoint_mask[k_idx]

                center_ATD[a_idx, t_idx, :] = geo_center
        
        return keypoints_ATKD, center_ATD, mask_ATK
