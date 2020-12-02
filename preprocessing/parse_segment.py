from moviepy.editor import *
import os
import json
import argparse


def timestamp_parse(gt):
    hf, t = gt.split(" - ")
    m, s = t.split(":")
    return hf, (float(s) + float(m)*60) // 1


def parseLabels(l):
    return [(timestamp_parse(e["gameTime"]), e["label"]) for e in l["annotations"]]


def create_walks(data_dir, file_masker):
    ans = []
    rt = os.walk(data_dir)

    for diry in rt:
        fl = diry[2]
        if file_masker(fl):
            ans.append(diry[0])

    return ans


def vidcutter(vid_name, st_pos=0, durat=20, i=0, ret_ans_dir="./clips"):
    vid = VideoFileClip(vid_name).subclip(st_pos, st_pos + durat)
    vid.write_videofile(os.path.join(ret_ans_dir, str(i) + ".mkv"), codec="libx264", verbose=None)


def onthetop_eve(lls, half, timestamp, win):
    for l, _ in lls:
        if l[0] == half:
            ceil = l[1] + win // 2
            floor = l[1] - win // 2
            if ceil >= timestamp >= floor:
                return True
    return False


def masker(fl):
    if "1.mkv" in fl and "2.mkv" in fl and "Labels.json" in fl:
        return True

    return False


def create_clips(dir_in, ret_ans_dir, siz=20, ext=".mkv"):
    rad = [os.path.join(ret_ans_dir, "goals"), os.path.join(ret_ans_dir, "bg"), os.path.join(ret_ans_dir, "cards"), os.path.join(ret_ans_dir, "subs")]
    arr = [0, 0, 0, 0]

    label_list = ['soccer-ball', 'substitution', 'card']

    for path in rad:
        if not os.path.exists(path):
            os.mkdir(path)

    for path in dir_in:
        with open(os.path.join(path, "Labels.json")) as f:
            l_ls = parseLabels(json.load(f))

            for tstamp, l_l in l_ls:
                vid_name = os.path.join(path, tstamp[0] + ext)
                t = tstamp[1]
                tmt = t-4
                tmf = t-5
                tmn = t-45

                tmnb = tmn > 0

                if tmf > 0:
                    if l_l == label_list[0]:
                        vidcutter(vid_name, tmf, siz, arr[0], rad[0])
                        arr[0] += 1
                        if onthetop_eve(l_ls, tstamp[0], tmn, siz) is False and tmnb:
                            vidcutter(vid_name, tmn, siz, arr[1], rad[1])
                            arr[1] += 1

                    elif label_list[1] in l_l:
                        vidcutter(vid_name, tmt, siz, arr[3], rad[3])
                        arr[3] += 1
                        if onthetop_eve(l_ls, tstamp[0], tmn, siz) is False and tmnb:
                            vidcutter(vid_name, tmn, siz, arr[1], rad[1])
                            arr[1] += 1

                    elif label_list[2] in l_l:
                        vidcutter(vid_name, tmt, siz, arr[2], rad[2])
                        arr[2] += 1
                        if onthetop_eve(l_ls, tstamp[0], tmn, siz) is False and tmnb:
                            vidcutter(vid_name, tmn, siz, arr[1], rad[1])
                            arr[1] += 1


if __name__ == "__main__":
    ga = argparse.Argumentarg()
    ga.add_argument("source_dir", type=str, help="source directory for videos and labels")
    ga.add_argument("ret_ans_dir", type=str, help="ret_ans directory for labeled video clips")
    argts = ga.parse_args()
    if not os.path.exists(argts.ret_ans_dir):
        os.mkdir(argts.ret_ans_dir)

    create_clips(create_walks(argts.source_dir, masker), argts.ret_ans_dir, 10)
