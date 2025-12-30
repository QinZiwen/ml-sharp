import cv2
import os

def compose_video(video_name="bullet_time_sharp.mp4"):
    frames_dir = "frames"
    files = sorted(os.listdir(frames_dir))

    img0 = cv2.imread(os.path.join(frames_dir, files[0]))
    H, W, _ = img0.shape

    out = cv2.VideoWriter(
        video_name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        24,
        (W, H)
    )

    for f in files:
        img = cv2.imread(os.path.join(frames_dir, f))
        out.write(img)

    out.release()
    print(f"save video to {video_name}")
