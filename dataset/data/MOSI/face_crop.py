import os
import cv2
from tqdm import tqdm

ROOT = "/root/autodl-tmp/data/CMU-MOSI/Process/img"
SAVE_ROOT = "/root/autodl-tmp/data/CMU-MOSI/Process/img_face"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def crop_face(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=3
    )

    h, w, _ = img.shape

    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, fw, fh = faces[0]

        pad = int(0.2 * max(fw, fh))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + fw + pad)
        y2 = min(h, y + fh + pad)

        face = img[y1:y2, x1:x2]
    else:
        # fallback 中心裁剪
        face = img[h//4:3*h//4, w//4:3*w//4]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, face)


for split in ["train", "valid", "test"]:
    split_root = os.path.join(ROOT, split)

    for video_id in tqdm(os.listdir(split_root)):
        video_dir = os.path.join(split_root, video_id)

        for clip_id in os.listdir(video_dir):
            clip_dir = os.path.join(video_dir, clip_id)

            for frame in os.listdir(clip_dir):
                img_path = os.path.join(clip_dir, frame)

                save_path = img_path.replace("img", "img_face")

                crop_face(img_path, save_path)
                