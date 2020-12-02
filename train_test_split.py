import shutil
import os


def train_test_split(root, final, labels_cls=None, distribute_data=[0.8, 0.1, 0.1]):
    if not os.path.exists(os.path.join(final, "train")):
        os.mkdir(os.path.join(final, "train"))

    if not os.path.exists(os.path.join(final, "val")):
        os.mkdir(os.path.join(final, "val"))

    if not os.path.exists(os.path.join(final, "test")):
        os.mkdir(os.path.join(final, "test"))

    if not labels_cls:
        labels_cls = os.listdir(root)

    for all_classes in labels_cls:
        path_class = os.path.join(root, all_classes)
        classes = os.listdir(path_class)
        classes_len = len(classes)
        distribute_training = max(0, round(classes_len * distribute_data[0]))

        distribute_test = len(classes) - round(classes_len * distribute_data[2])
        distribute_test = min(distribute_test, classes_len - 1)

        if not os.path.exists(os.path.join(os.path.join(final, "train"), all_classes)):
            os.mkdir(os.path.join(os.path.join(final, "train"), all_classes))

        if not os.path.exists(os.path.join(os.path.join(final, "test"), all_classes)):
            os.mkdir(os.path.join(os.path.join(final, "test"), all_classes))

        if not os.path.exists(os.path.join(os.path.join(final, "val"), all_classes)):
            os.mkdir(os.path.join(os.path.join(final, "val"), all_classes))

        for path in classes[:distribute_training]:
            shutil.copy2(os.path.join(path_class, path), os.path.join(os.path.join(final, "train"), all_classes))

        for path in classes[distribute_training:distribute_test]:
            shutil.copy2(os.path.join(path_class, path), os.path.join(os.path.join(final, "val"), all_classes))

        for path in classes[distribute_test:]:
            shutil.copy2(os.path.join(path_class, path), os.path.join(os.path.join(final, "test"), all_classes))
