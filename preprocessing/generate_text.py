import os


class GetLabels:
    def __init__(self, none_py, list_cls, list_accuracy, freq=75):
        self.none_py = none_py
        self.input_offset = 3
        self.frames_per_sec = 25.0
        self.esec = []
        self.list_accuracy = list_accuracy
        self.list_cls = list_cls
        self.freq = freq

    def main_extractor(self, vector, all_classes, none_py):
        tp = []
        for i in range(vector.shape[0]):
            if vector[i]:
                tp.append([i / self.frames_per_sec * self.freq + self.input_offset, all_classes, float(none_py[i, all_classes])])

        return len(tp), tp

    def join_videos(self, input_dir, ret_ans_path=None):
        if not ret_ans_path:
            filename = os.path.basename(input_dir) + "_summary.mp4"
            ret_ans_path = os.path.join(os.path.dirname(input_dir), filename)

        from moviepy.editor import VideoFileClip, concatenate_videoclips

        source = VideoFileClip(input_dir)
        s_list = []
        s, e = 0

        for i, _, _ in self.esec:
            if i >= e:
                s = max(i - 5, 0)
            else:
                s_list.pop()

            e = min(i + 5, source.end + 5)
            if s != e:
                s_list.append(source.subclip(s, e))

        summary = concatenate_videoclips(s_list)
        summary.write_videofile(ret_ans_path, codec="libx264", verbose=None)

    def find_stamp(self, none_py=None, list_cls=None, accuracies=None):
        if not list_cls:
            list_cls = self.list_cls

        if not none_py:
            none_py = self.none_py

        if not accuracies:
            accuracies = self.accuracies

        for all_classes, t in zip(list_cls, accuracies):
            events, timestamp = self.main_extractor(none_py[:, all_classes] > t, all_classes, none_py)
            self.esec = self.esec + timestamp

        self.esec = sorted(self.esec, key=lambda x: x[0])

    def put_it(self, ans_dir=None):
        if not ans_dir:
            ans_dir = "."

        ans_path = os.path.join(ans_dir, "Annotations.json")
        print("Annotations saved to " + ans_path)
        data = {"annotations": self.esec}

        import json
        with open(ans_path, "w") as file:
            json.dump(data, file, indent=4)

    def get_it(self, ip):
        print("Annotations loaded from " + ip)
        import json

        with open(ip) as file:
            data = json.load(file)

        self.esec = data["annotations"]

    def find_freq_per_sec(self, source):
        import cv2
        cap = cv2.VideoCapture(source)
        self.frames_per_sec = cap.get(cv2.CAP_PROP_FPS)
        self.input_offset = 75 / self.frames_per_sec

        print("Source FPS : {}".format(self.frames_per_sec))
