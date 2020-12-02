from preprocessing.detect_class import make_video_func
from preprocessing.parse_segment import create_walks
from preprocessing.moment import PrettyPrintTime
import numpy as np
import os
import re
import argparse


def make_semi_penetrable(sequence):
    def filter(files):

        for f in files:
            if re.match(sequence, f):
                return True

        return False

    return filter


if __name__ == "__main__":
    pr = PrettyPrintTime()
    pr.restart()
    arguments = argparse.Argumentarg().parse_args()
    root = arguments.source
    sequence = ".*\{}$".format(arguments.ext)
    masker_fcn = make_semi_penetrable(sequence)
    dirs = create_walks(root, masker_fcn)

    for path in dirs:
        mp4_f = [f for f in os.listdir(path) if re.match(sequence, f)]
        for v in mp4_f:
            mp4_directory = os.path.join(path, v)
            np_path = mp4_directory + "_pred.npy"
            answer = make_video_func(mp4_directory, arguments.model, arguments.frequency)
            np.save(np_path, answer)
            pr.the_time()

    pr.the_total_time()
