import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision.models.video import r3d_18
from torchvision.transforms import Compose, Lambda, Resize
from torchvision.transforms._transforms_video import NormalizeVideo

# ------------ Config ------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_LEN = 16
INPUT_DIR = "tracked_videos"
OUTPUT_CSV = "activity_predictions.csv"
# --------------------------------

# ------------ Load Model ------------
print("🔄 Loading r3d_18 model...")
model = r3d_18(pretrained=True, progress=True)
model = model.eval().to(DEVICE)
print("✅ Model loaded.")

# ------------ Transform ------------
transform = Compose([
    Lambda(lambda x: x / 255.0),
    NormalizeVideo(mean=[0.43216, 0.394666, 0.37645],
                   std=[0.22803, 0.22145, 0.216989]),
    Resize((112, 112))
])

# ------------ Load and Process Video ------------
def load_video_tensor(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < CLIP_LEN:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.tensor(frame).permute(2, 0, 1))  # C, H, W

    cap.release()

    if len(frames) < CLIP_LEN:
        return None

    video = torch.stack(frames)  # T, C, H, W
    video = video.permute(1, 0, 2, 3)  # C, T, H, W
    video = transform(video)
    return video.unsqueeze(0).to(DEVICE)  # B, C, T, H, W

# ------------ Predict ------------
def predict(video_path):
    video_tensor = load_video_tensor(video_path)
    if video_tensor is None:
        return None
    with torch.no_grad():
        preds = model(video_tensor)
    class_id = torch.argmax(preds, dim=1).item()
    return class_id

# ------------ Process All Videos ------------
results = []

for group in ["normal", "anomaly"]:
    group_path = os.path.join(INPUT_DIR, group)
    if not os.path.isdir(group_path):
        continue
    for label in os.listdir(group_path):
        label_path = os.path.join(group_path, label)
        if not os.path.isdir(label_path):
            continue
        for vid in tqdm(os.listdir(label_path), desc=f"{group}/{label}"):
            if not vid.endswith(".avi"):
                continue
            video_path = os.path.join(label_path, vid)
            class_id = predict(video_path)
            results.append({
                "video": video_path,
                "group": group,
                "folder_label": label,
                "predicted_class_id": class_id
            })

# ------------ Save Results ------------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Done. Predictions saved to: {OUTPUT_CSV}")