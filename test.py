from preprocessing.make_video import create_mpeg4
from keras.models import mdl_initialize
from sklearn.metrics import confusion_matrix
import numpy as np


def find_labels(label):
    label = np.array(label)
    label = label.reshape(-1, label.shape[-1])
    label = np.argmax(label, 1)
    return label


def mdl_prediction(iter_, checker, model):
    label_hat, label = []
    c = 0

    for training_data, label in checker:
        label.append(label)
        label_p = model.predict(training_data)
        c += 1
        label_hat.append(label_p)
        if c > iter_:
            break

    return find_labels(label), find_labels(label_hat)


def calculate_cm(label, label_hat, labels_cls):
    cm = confusion_matrix(label, label_hat)
    if labels_cls:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    accuracy_ll = cm.max() / 2.0
    return accuracy_ll


if __name__ == "__main__":
    files_check_path = "./data/split_clips/test"
    path_training = "./data/split_clips/train"
    defined_dimentions = (150, 224, 224, 3)
    bs = 6
    produce_video = create_mpeg4(path_training, files_check_path, defined_dimentions)
    checker = produce_video.generate(train_or_val="val")
    iter_ = len(produce_video.filenames_val) // bs
    path = "model.hdf5"
    model = mdl_initialize(path)
    label, label_hat = mdl_prediction(iter_, checker, model)
    class_labels = []
    temp = sorted(produce_video.classname_by_id.items(), key=lambda x: x[0])
    for a, v in temp:
        class_labels.append(v)

    calculate_cm(label, label_hat, class_labels)
