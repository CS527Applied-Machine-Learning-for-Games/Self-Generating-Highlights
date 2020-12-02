import cv2
import numpy as np
from keras.models import mdl_initialize


def make_video_func(filename, model_file, sampintrl, new_size=True, verbose=False):
    frms, probs = []
    mdl = mdl_initialize(model_file)
    cp = cv2.VideoCapture(filename)
    no_of_f = cp.get(cv2.CAP_PROP_FRAME_COUNT)
    count = 0

    while True:
        ret, frame = cp.read()
        if ret:
            if new_size:
                frame = cv2.new_size(frame, (int(mdl.input.shape[3]), int(mdl.input.shape[2])), interpolation=cv2.INTER_AREA)
            else:
                frame = frame[:, (frame.shape[1] - frame.shape[0]) // 2 : (frame.shape[1] - frame.shape[0]) // 2 + frame.shape[0], :]
            frms.append(frame)
        else:
            break

        if len(frms) == int(mdl.input.shape[1]):
            count += 1
            if verbose:
                print("Processed {} answer of {}".format(count, int(no_of_f // sampintrl)))
            prob = mdl.predict((np.array([frms]) / 255) * 2 - 1)
            probs.append(prob[0])
            if sampintrl < 150:
                frms = frms[sampintrl:]
            else:
                frms = []

    print("Scan completed.")

    return np.array(probs)
