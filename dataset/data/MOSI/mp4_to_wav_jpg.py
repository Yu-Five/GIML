import os
import pandas as pd
import subprocess
import cv2


LABEL_PATH = "/root/autodl-tmp/data/CMU-MOSI/label.csv"
RAW_PATH = "/root/autodl-tmp/data/CMU-MOSI/Raw"

WAV_ROOT = "/root/autodl-tmp/data/CMU-MOSI/Process/wav"
IMG_ROOT = "/root/autodl-tmp/data/CMU-MOSI/Process/img"

# LABEL_PATH = "/root/autodl-tmp/data/MOSI/label.csv"
# RAW_PATH = "/root/autodl-tmp/data/MOSI/Raw"

# WAV_ROOT = "/root/autodl-tmp/data/MOSI/Process/wav"
# IMG_ROOT = "/root/autodl-tmp/data/MOSI/Process/img"


# def extract_audio(video_path, save_path):
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     if os.path.exists(save_path):
#         return

#     cmd = [
#         "ffmpeg",
#         "-i", video_path,
#         "-acodec", "pcm_s16le",
#         "-ar", "16000",
#         "-ac", "1",
#         save_path,
#         "-y"
#     ]

#     subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_audio(video_path, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        return

    cmd = [
        "ffmpeg",
        "-y",                    # 覆盖输出
        "-i", video_path,
        "-vn",                   # 不要视频流
        "-ac", "1",              # 单声道
        "-ar", "16000",          # 16kHz
        "-sample_fmt", "s16",    # 16bit
        "-acodec", "pcm_s16le",  # PCM signed 16bit little endian
        save_path
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)



# def extract_frames(video_path, save_dir, fps_keep=10):
#     os.makedirs(save_dir, exist_ok=True)

#     if len(os.listdir(save_dir)) > 0:
#         return

#     cap = cv2.VideoCapture(video_path)
#     video_fps = int(cap.get(cv2.CAP_PROP_FPS))

#     if video_fps == 0:
#         cap.release()
#         return

#     frame_interval = max(1, video_fps // fps_keep)

#     frame_count = 0
#     saved_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_count % frame_interval == 0:
#             img_name = os.path.join(save_dir, f"{saved_count:05d}.jpg")
#             cv2.imwrite(img_name, frame)
#             saved_count += 1

#         frame_count += 1

#     cap.release()


# 每个clip太短了，保留所有帧
def extract_frames(video_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    if len(os.listdir(save_dir)) > 0:
        return

    cap = cv2.VideoCapture(video_path)

    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_name = os.path.join(save_dir, f"{saved_count:05d}.jpg")
        cv2.imwrite(img_name, frame)
        saved_count += 1

    cap.release()




def process_mosi():

    df = pd.read_csv(LABEL_PATH)

    for _, row in df.iterrows():

        video_id = row["video_id"]
        clip_id = str(row["clip_id"])
        mode = row["mode"]  # train / valid / test

        video_path = os.path.join(RAW_PATH, video_id, f"{clip_id}.mp4")

        if not os.path.exists(video_path):
            continue

        # ===== 音频路径 =====
        wav_save_path = os.path.join(
            WAV_ROOT,
            mode,
            video_id,
            f"{clip_id}.wav"
        )

        # ===== 图片路径 =====
        img_save_dir = os.path.join(
            IMG_ROOT,
            mode,
            video_id,
            clip_id
        )

        extract_audio(video_path, wav_save_path)
        extract_frames(video_path, img_save_dir)


if __name__ == "__main__":
    process_mosi()

