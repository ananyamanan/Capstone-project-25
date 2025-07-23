import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.models.video import r3d_18
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Lambda, Resize
from torchvision.transforms._transforms_video import NormalizeVideo
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn

# -------- Config --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_LEN = 16
TEST_DIR = os.path.join("tracked_videos", "test")
MODEL_PATH = os.path.join("activity_predictions", "r3d18_sphar_best.pth")
OUTPUT_DIR = os.path.join("activity_predictions", "test_results")
BATCH_SIZE = 4

LABEL_MAP = {
    'carcrash': 0,
    'kicking': 1,
    'stealing': 2,
    'vandalizing': 3,
    'neutral': 4,
    'running': 5,
    'sitting': 6
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- Transform --------
transform = Compose([
    Lambda(lambda x: x / 255.0),
    NormalizeVideo(mean=[0.43216, 0.394666, 0.37645],
                   std=[0.22803, 0.22145, 0.216989]),
    Resize((112, 112))
])

# -------- Dataset Class --------
class ActionDataset(Dataset):
    def __init__(self, root_dir, label_map, clip_len=16, transform=None):
        self.samples = []
        self.label_map = label_map
        self.clip_len = clip_len
        self.transform = transform

        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if not os.path.isdir(category_path):
                continue
            for class_name in os.listdir(category_path):
                class_path = os.path.join(category_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                for vid in os.listdir(class_path):
                    if vid.endswith(".mp4") and not vid.startswith('.'):
                        video_path = os.path.join(class_path, vid)
                        label = self.label_map.get(class_name)
                        if label is not None:
                            self.samples.append((video_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []

        while len(frames) < self.clip_len:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.tensor(frame).permute(2, 0, 1))
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"Empty video: {video_path}")
        while len(frames) < self.clip_len:
            frames.append(frames[-1])

        video = torch.stack(frames).permute(1, 0, 2, 3).float()
        if self.transform:
            video = self.transform(video)
        return video, label, video_path

# -------- Load Dataset & Model --------
test_dataset = ActionDataset(TEST_DIR, LABEL_MAP, clip_len=CLIP_LEN, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = r3d_18(pretrained=False)
model.fc = nn.Linear(in_features=512, out_features=7)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------- Run Inference --------
y_true, y_pred, video_paths = [], [], []

with torch.no_grad():
    for videos, labels, paths in tqdm(test_loader, desc="🔍 Evaluating"):
        videos = videos.to(DEVICE)
        outputs = model(videos)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())
        video_paths.extend(paths)

# -------- Save Results --------
df = pd.DataFrame({
    "video": video_paths,
    "true_label": [INV_LABEL_MAP[i] for i in y_true],
    "predicted_label": [INV_LABEL_MAP[i] for i in y_pred],
    "correct": [yt == yp for yt, yp in zip(y_true, y_pred)]
})
df.to_csv(os.path.join(OUTPUT_DIR, "test_results.csv"), index=False)
print(f"✅ Saved test results to {OUTPUT_DIR}/test_results.csv")

# -------- Save Confusion Matrix --------
target_names = [INV_LABEL_MAP[i] for i in range(7)]
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix (Test)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_test.png"))
print(f"📊 Saved test confusion matrix to {OUTPUT_DIR}/confusion_matrix_test.png")

# -------- Log Misclassifications --------
misclassified = df[df["correct"] == False]
misclassified.to_csv(os.path.join(OUTPUT_DIR, "misclassified.csv"), index=False)
print(f"⚠️ Logged {len(misclassified)} misclassified videos to misclassified.csv")