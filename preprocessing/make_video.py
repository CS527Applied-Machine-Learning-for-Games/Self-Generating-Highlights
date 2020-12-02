import cv2
import numpy as np


def overlay_images(source, overlay, alpha):
    ret_ans = source.copy()
    cv2.addWeighted(overlay, alpha, ret_ans, 1 - alpha, 0, ret_ans)

    return ret_ans


def create_mpeg4(root, outdir, fps=25.0):
    if root.dtype == "float32":
        root = (root * 255).astype("uint8")

    ans = cv2.VideoWriter(outdir, cv2.VideoWriter_fourcc(*"MP4V"), fps, (root.shape[2], root.shape[1]))
    frames = root.shape[0]

    for f in range(frames):
        ans.write(root[f])


def get_dimentions(ip, temp, oy=False, alp=0.5):
    no_of_f, h, w, no_chnl = temp.shape
    print(temp.shape)
    ip = np.power(ip, 0.8)
    input_frames = ip.shape[0]

    if temp.dtype == "uint8":
        temp = (temp / 255).astype("float32")
    else:
        temp = ((temp - temp.min()) / (temp.max() - temp.min())).astype("float32")

    ans = np.repeat(ip, round(no_of_f / input_frames), axis=0)
    ip_shape_len = len(ip.shape)
    if ip_shape_len == 3:
        ans = np.stack([ans, ans, ans], axis=-1)

    ans = [ans[n] for n in range(no_of_f)]
    for i in range(no_of_f):
        ans[i] = cv2.resize(ans[i], (w, h))
        if oy:
            ans[i] = overlay_images(temp[i], ans[i], alp)

    return np.array(ans)


if __name__ == "__main__":
    overlay_video = get_dimentions(o_stack, img_stack)
    create_mpeg4(overlay_video, "overlay_video.mp4")
