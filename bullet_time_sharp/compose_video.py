import cv2
import os

def compose_video():
    frames_dir = "frames"
    files = sorted(os.listdir(frames_dir))

    img0 = cv2.imread(os.path.join(frames_dir, files[0]))
    H, W, _ = img0.shape

    out = cv2.VideoWriter(
        "bullet_time_sharp.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        24,
        (W, H)
    )

    for f in files:
        img = cv2.imread(os.path.join(frames_dir, f))
        out.write(img)

    out.release()
