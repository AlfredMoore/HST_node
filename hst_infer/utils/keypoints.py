import numpy as np
from hst_infer.node_config import KEYPOINT_INTERPOLATION

YOLO_POSE_DICT = {
    0: "Nose",
    1: "Left-eye",
    2: "Right-eye",
    3: "Left-ear",
    4: "Right-ear",
    5: "Left-shoulder",
    6: "Right-shoulder",
    7: "Left-elbow",
    8: "Right-elbow",
    9: "Left-wrist",
    10: "Right-wrist",
    11: "Left-hip",
    12: "Right-hip",
    13: "Left-knee",
    14: "Right-knee",
    15: "Left-ankle",
    16: "Right-ankle",
}

BLAZE_POSE_DICT = {
    0: "Nose",
    1: "Left-eye-inner",
    2: "Left-eye",
    3: "Left-eye-outer",
    4: "Right-eye-inner",
    5: 'Right-eye',
    6: "Right-eye outer",
    7: "Left-ear",
    8: "Right-ear",
    9:" Mouth-left",
    10: "Mouth-right",
    11: "Left-shoulder",
    12: "Right-shoulder",
    13: "Left-elbow",
    14: "Right-elbow",
    15: "Left-wrist",
    16: "Right-wrist",
    17: "Left-pinky", #1 knuckle
    18: "Right-pinky", #1 knuckle
    19: "Left-index", #1 knuckle
    20: "Right-index",#1 knuckle
    21: "Left-thumb", #2 knuckle
    22: "Right-thumb", #2 knuckle
    23: "Left-hip",
    24: "Right-hip",
    25: "Left-knee",
    26: "Right-knee",
    27: "Left-ankle",
    28: "Right-ankle",
    29: "Left-heel",
    30: "Right-heel",
    31: "Left-foot-index",
    32: "Right-foot-index",
    }

def get_keypoint_mapping(old_dict: dict = YOLO_POSE_DICT, 
                         new_dict: dict = BLAZE_POSE_DICT,
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    return tuple(old_idx, new_idx), where they share the same value in each's dict respectively
    """
    old_idx = np.arange(len(old_dict),dtype=int)
    new_idx = np.zeros_like(old_idx, dtype=int)

    for key1, val1 in old_dict.items():
        for key2, val2 in new_dict.items():
            if val1 == val2:
                new_idx[key1] = key2
                break
        assert Exception, "Could find a mapping from old_dict to new_dict"
    
    return old_idx, new_idx


class human_keypoints:

    old_idx, new_idx = get_keypoint_mapping(YOLO_POSE_DICT, BLAZE_POSE_DICT)
    interpolation: bool = KEYPOINT_INTERPOLATION

    @classmethod
    def map_yolo_to_hst(cls, keypoints_ATKD: np.ndarray,
                        keypoint_mask_ATK: np.ndarray, 
                        keypoint_center_ATD: np.ndarray,
                        hst_K: int = len(BLAZE_POSE_DICT),
                        ) -> np.ndarray:
        """
        input: keypoints [A,T,17,3], mask [A,T,K]
        return: keypoinst [1,A,T,99]
        """
        A,T,K,D = keypoints_ATKD.shape 
        # Normalize keypoints
        keypoints_ATKD = keypoints_ATKD - keypoint_center_ATD[:,:,np.newaxis,:]

        keypoint_mask_ATKD = np.repeat(keypoint_mask_ATK[...,np.newaxis], 3, axis=3)
        assert keypoint_mask_ATKD.shape == (A,T,K,3)

        keypoints_ATKD = np.where(keypoint_mask_ATKD, keypoints_ATKD, np.nan)
        # init 33 keypoints with nan
        hst_keypoints_ATKD = np.full((A,T,hst_K,D), np.nan, dtype=float)
        hst_keypoints_ATKD[:,:,cls.new_idx,:] = keypoints_ATKD[:,:,cls.old_idx,:]
        
        if cls.interpolation:
            raise Exception("Interpolation Model to be implemented")
        
        hst_keypoints_batch = hst_keypoints_ATKD.reshape((1,A,T,hst_K*D))
        assert hst_keypoints_batch.shape == (1,A,T,99)

        return hst_keypoints_batch
        


if __name__ == "__main__":
    print(human_keypoints.old_idx, human_keypoints.new_idx)     # good
    # print(len(BLAZE_POSE_DICT))
    A,T,K,D = 1,5,len(YOLO_POSE_DICT),3
    keypoints_ATKD = np.arange(A*T*K*D).reshape((A,T,K,D))
    keypoint_mask_ATK = np.random.randint(0,2, size=(A,T,K),dtype=bool)
    keypoint_center_ATD = np.random.random(size=(A,T,D))

    res = human_keypoints.map_yolo_to_hst(keypoints_ATKD, keypoint_mask_ATK, keypoint_center_ATD, len(BLAZE_POSE_DICT))
    print(keypoint_mask_ATK)
    print(res)