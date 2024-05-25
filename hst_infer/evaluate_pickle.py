import pickle

from hst_infer.node_config import *

pickle_file_path = PICKLE_DIR_PATH / "evaluation_data.pkl"

def main():

    counter = 0

    with open(pickle_file_path.as_posix(), 'rb') as pickle_hd:
        while True:
            try:
                dict_to_read: dict = pickle.load(pickle_hd)
                counter += 1
                # print(dict_to_read.keys())
                human_pos_ATD_stretch = dict_to_read["human_pos_ground_true_ATD"]
                multi_human_mask_AT = dict_to_read["human_pos_mask_AT"]
                multi_human_pos_ATMD = dict_to_read["human_pos_HST_ATMD"]
                agent_position_prob = dict_to_read["HST_mode_weights"]
                human_t = dict_to_read["human_T"]

                




            except EOFError:
                break
    
    print(counter)


if __name__ == "__main__":
    main()