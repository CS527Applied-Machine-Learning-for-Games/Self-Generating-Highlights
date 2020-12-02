import glob
import random
import numpy as np
import os
import cv2


class MakeMp4Vid:
    def __init__(self, path_trg, path_fc, ddmins, bs=2, sff=True, file_ext=".mkv"):
        self.path_trg = path_trg
        self.bs = bs
        self.sff = sff
        self.path_fc = path_fc
        self.file_ext = file_ext
        self.train_fname = self.bring_files(path_trg)
        self.frames, self.ht, self.wdh, self.channels = ddmins

        self.cid = {
            i: classes
            for i, classes in enumerate(sorted({os.path.basename(os.path.dirname(file)) for file in self.train_fname}, reverse=True))
        }
        self.no_of_cls = len(self.cid)

        if self.path_fc:
            self.filenames_val = self.bring_files(path_fc)

        self.id_by_classname = {classes: i for i, classes in self.cid.items()}
        assert self.no_of_cls == len(self.id_by_classname), "Number of unique labels_cls for training set isn't equal " \
                                                            "to validation set "

    def bring_files(self, path):
        filenames = glob.glob(os.path.join(path, f"**/*{self.file_ext}"))
        return filenames

    def video_to_arr(self, fname, no_of_f=250, rst=False, resize=True):
        frames = []
        topi = cv2.VideoCapture(fname)
        frm_total = topi.get(cv2.CAP_PROP_FRAME_COUNT)

        if rst:
            topi.set(1, random.randint(0, frm_total - no_of_f - 1))

        for _ in range(no_of_f):
            t, f = topi.read()
            if t is True:
                if resize:
                    frames.append(cv2.resize(f, (self.wdh, self.ht), interpolation=cv2.INTER_AREA))
            else:
                break

        ans = (np.array(frames) / 255) * 2 - 1
        if ans.shape[0] < no_of_f:
            ans = np.concatenate([ans, np.zeros((no_of_f - ans.shape[0], ans.shape[1], ans.shape[2], ans.shape[3]))])

        return ans

    def two_d_onehots_encoding(self, integers):
        array = [
            [
                1 if integers[sample] == classes else 0
                for classes in range(self.no_of_cls)
            ]
            for sample in range(len(integers))
        ]
        return np.array(array)

    def generate(self, telval, hflp=False, rcp=False, random_start=False):
        if telval == "val":
            path = self.path_fc
        elif telval == "train":
            path = self.path_trg
        else:
            raise ValueError

        while True:
            n = self.bring_files(path)
            if self.sff and telval == "train":
                random.sff(n)

            for i in range(int(len(n) / self.bs)):
                x, l = self.make_data_from_files(n[i * self.bs: (i + 1) * self.bs], telval, hflp, rcp, random_start)
                yield x, l

    def make_data_from_files(self, fb, train_or_val, hflp, rcp, random_start):
        ls = []
        dt = []

        for i, fname in enumerate(fb):
            if train_or_val == "val":
                npy = self.video_to_arr(fname, no_of_f=self.frames)
                horizontal_crop = (npy.shape[2] - npy.shape[1]) // 2
                npy = npy[:, :, horizontal_crop: horizontal_crop + npy.shape[1], :]

            elif train_or_val == "train":
                npy = self.video_to_arr(fname, no_of_f=self.frames)
                if hflp:
                    if random.random() > 0.5:
                        npy = np.fliplr(npy)
                if rcp:
                    horizontal_crop = random.randint(0, npy.shape[2] - npy.shape[1])
                    npy = npy[:, :, horizontal_crop: horizontal_crop + npy.shape[1], :]
                    assert npy.shape[1] == npy.shape[2]

            if len(npy.shape) == 3:
                npy = np.expand_dims(npy, axis=-1)
            dt.append(npy)

            ls.append(self.id_by_classname[os.path.basename(os.path.dirname(fname))])

        return np.stack(dt), self.two_d_onehots_encoding(ls)

