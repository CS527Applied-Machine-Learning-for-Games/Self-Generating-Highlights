import os
from keras.models import mdl_initialize
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from preprocessing.make_video import create_mpeg4
from keras.callbacks import CSVLogger
import argparse


def fix_three_channels(model, height=152, is_train=False):

    for i, l in enumerate(model.layers):
        if i < height:
            l.is_train = False
        else:
            l.is_train = is_train


if __name__ == "__main__":
    arguments = argparse.Argumentarg().parse_args()
    mdl_test = mdl_initialize(arguments.input)
    _, no_of_f, no_of_depth, no_of_choda, no_chnl = mdl_test.input.shape
    defined_dimentions = (int(no_of_f), int(no_of_depth), int(no_of_choda), int(no_chnl))

    produce_video = create_mpeg4(arguments[0], arguments[1], defined_dimentions)
    labels_cls = produce_video.classname_by_id

    wts_cls = {}
    for id, name in labels_cls.items():
        wts_cls[id] = len(os.listdir(os.path.join(arguments[0], name)))
    c_t = sum(wts_cls.values())

    for c in wts_cls:
        wts_cls[c] = c_t // wts_cls[c]

    produce_tr = produce_video.generate(train_or_val="train", horizontal_flip=True, random_crop=True, random_start=True)
    seq_p_e_train = len(produce_video.filenames_train) // arguments.bs

    checker = produce_video.generate(train_or_val="val")

    fix_three_channels(mdl_test, height=152, is_train=True)

    mdl_test.compile(optimizer=SGD(lr=0.001, decay=1e-7, momentum=0.9, nesterov=True), loss="categorical_crossentropy",
                     metrics=["accuracy"])

    ptr_chck = ModelCheckpoint(filepath=".{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)

    csv_logger = CSVLogger("history.csv", append=True, separator=";")

    mdl_test.fit_generator(produce_tr, steps_per_epoch=seq_p_e_train, epochs=10, wts_cls=wts_cls,
                           validation_data=checker, validation_steps=len(produce_video.filenames_val) // arguments.bs, callbacks=[csv_logger, ptr_chck])

    mdl_test.put_it("trained_model.hdf5")
