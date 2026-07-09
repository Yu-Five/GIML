import os
import pandas as pd

ROOT = "/root/autodl-tmp/data/MOSI"
IMG_ROOT = os.path.join(ROOT, "Process", "img")
LABEL_PATH = os.path.join(ROOT, "label.csv")

NUM_FRAME = 10  

def check_frames():

    df = pd.read_csv(LABEL_PATH)

    total = 0
    less_than_required = 0
    zero_frame = 0

    min_frame = 999999
    min_sample = None

    for _, row in df.iterrows():

        mode = row["mode"]
        video_id = row["video_id"]
        clip_id = str(row["clip_id"])

        frame_dir = os.path.join(
            IMG_ROOT,
            mode,
            video_id,
            clip_id
        )

        if not os.path.exists(frame_dir):
            continue

        images = [f for f in os.listdir(frame_dir) if f.endswith(".jpg")]

        frame_count = len(images)
        total += 1

        if frame_count == 0:
            zero_frame += 1

        if frame_count < NUM_FRAME:
            less_than_required += 1
            print(f"[不足] {mode}/{video_id}/{clip_id} 只有 {frame_count} 帧")

        if frame_count < min_frame:
            min_frame = frame_count
            min_sample = f"{mode}/{video_id}/{clip_id}"

    print("\n========== 统计结果 ==========")
    print(f"总样本数: {total}")
    print(f"帧数为0的样本数: {zero_frame}")
    print(f"帧数少于 {NUM_FRAME} 的样本数: {less_than_required}")
    print(f"最少帧样本: {min_sample}")
    print(f"最少帧数: {min_frame}")

if __name__ == "__main__":
    check_frames()
