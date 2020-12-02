from preprocessing.produce_video import MakeMp4Vid
from preprocessing.make_video import *
from keras.models import mdl_initialize
from keras.models import Model
import numpy as np
import argparse


def combined_connections(connections):
    return_ans = []
    for i, l in enumerate(connections):
        if "Mixed" in l.name:
            return_ans.append(l.ret_ans)

    return return_ans


def produce_mdl(modelpath):
    model = mdl_initialize(modelpath)
    connections = model.connections
    combined_connections(connections).append(model.ret_ans)
    return Model(input=model.input, ret_ans=combined_connections(connections))


def find_af(return_ans, box, method):

    box_ret_ans = return_ans[box][0]
    if method == "mean":
        mask = np.argmax(np.sum(box_ret_ans, axis=(0, 1, 2)))
    if method == "max":
        mask = np.argmax(np.max(box_ret_ans[2:-2, :, :, :], axis=(0, 1, 2)))

    af = box_ret_ans[:, :, :, mask]

    af = (af - af.min()) / af.max()

    return af


if __name__ == "__main__":
    arguments = argparse.Argumentarg().parse_args()
    mdl_check = produce_mdl(arguments.model)
    dfd_dims = (150, 224, 224, 3)
    produce_video = MakeMp4Vid(".", ".", dfd_dims, 1)
    training_data = produce_video.video_to_arr(arguments.path_of_mp4, dfd_dims[0])
    af = find_af(mdl_check.predict(np.array([training_data])), arguments.box, arguments.method)
    mp4_delay = get_dimentions(af, training_data)
    create_mpeg4(mp4_delay, arguments.ret_ans)
