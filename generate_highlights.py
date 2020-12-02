import argparse
import os
from preprocessing.generate_text import GetLabels
from preprocessing.detect_class import make_video_func
import numpy as np

if __name__ == "__main__":
    arg = argparse.Argumentarg()
    arguments = arg.parse_args()

    pth_root = arguments[0]
    ret_ans_path = arguments[1]
    model_path = arguments[2]
    frequency = arguments[3]
    observation = arguments[4]

    if not arguments.observation:
        answer = make_video_func(pth_root, model_path, frequency)
    else:
        answer = np.load(observation)

    all_classes = [0, 1, 2]
    accuracy_ll = [0.8, 0.9, 0.95]

    obs = GetLabels(answer, all_classes, accuracy_ll)

    obs.find_freq_per_sec(pth_root)

    obs.find_stamp()

    ret_ans_folder = os.path.dirname(ret_ans_path)

    if not os.path.exists(ret_ans_folder) and ret_ans_folder:
        os.mkdir(ret_ans_folder)
        print("Succesfully created directory: " + ret_ans_folder)

    obs.join_videos(pth_root, ret_ans_path)

    print("Output saved at: " + ret_ans_path)