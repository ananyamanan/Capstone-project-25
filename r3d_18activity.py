import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18
from torchvision.transforms import Compose, Lambda, Resize
from torchvision.transforms._transforms_video import NormalizeVideo
import torch.nn as nn
import torch.optim as optim
import random
from torchvision.transforms._transforms_video import NormalizeVideo
from torchvision.transforms import RandomResizedCrop
import torchvision.transforms as T



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# ------------ Config ------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_LEN = 16
INPUT_DIR = os.path.join("tracked_videos", "train")
VAL_DIR = os.path.join("tracked_videos", "val")
OUTPUT_DIR = "activity_predictions"
BATCH_SIZE = 4
NUM_EPOCHS = 20

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

# ------------ Transform ------------
transform = Compose([
    Lambda(lambda x: x / 255.0),
    NormalizeVideo(mean=[0.43216, 0.394666, 0.37645],
                   std=[0.22803, 0.22145, 0.216989]),
    Resize((112, 112))
])

# ------------ Dataset Class ------------
class ActionDataset(Dataset):
    def __init__(self, root_dir, label_map, clip_len=16, transform=None):
        self.samples = []
        self.label_map = label_map
        self.clip_len = clip_len
        self.transform = transform

        for category in os.listdir(root_dir):  # anomaly, normal
            category_path = os.path.join(root_dir, category)
            if not os.path.isdir(category_path):
                continue
            for class_name in os.listdir(category_path):  # kicking, running, etc.
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
            frames.append(torch.tensor(frame).permute(2, 0, 1))  # C, H, W
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"Empty video: {video_path}")
        while len(frames) < self.clip_len:
            frames.append(frames[-1])

        video = torch.stack(frames).permute(1, 0, 2, 3).float()  # C, T, H, W
        if self.transform:
            video = self.transform(video)
        return video, label, video_path

# ------------ Load Dataset ------------
train_dataset = ActionDataset(INPUT_DIR, LABEL_MAP, clip_len=CLIP_LEN, transform=transform)
val_dataset = ActionDataset(VAL_DIR, LABEL_MAP, clip_len=CLIP_LEN, transform=transform)

# Safety check
print(f"🧪 Found {len(train_dataset)} training videos")
print(f"🧪 Found {len(val_dataset)} validation videos")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ------------ Model Setup ------------
model = r3d_18(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=7)  # 7 custom classes
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ------------ Training Loop ------------
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for videos, labels, _ in train_loader:
        videos, labels = videos.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {total_loss / len(train_loader):.4f}")

# ------------ Evaluation on Validation Set ------------
model.eval()
y_true, y_pred, video_paths = [], [], []
with torch.no_grad():
    for videos, labels, paths in val_loader:
        videos = videos.to(DEVICE)
        outputs = model(videos)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())
        video_paths.extend(paths)

# ------------ Report & Save Results ------------
target_names = [INV_LABEL_MAP[i] for i in range(7)]

report = classification_report(y_true, y_pred, target_names=target_names)
print("\nClassification Report:\n", report)

df = pd.DataFrame({
    "video": video_paths,
    "true_label": [INV_LABEL_MAP[i] for i in y_true],
    "predicted_label": [INV_LABEL_MAP[i] for i in y_pred]
})
df.to_csv(os.path.join(OUTPUT_DIR, "fine_tuned_results.csv"), index=False)
print(f"✅ Saved prediction results to {OUTPUT_DIR}/fine_tuned_results.csv")

# ------------ Confusion Matrix Plot ------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_finetuned.png"))
print(f"📈 Saved confusion matrix to {OUTPUT_DIR}/confusion_matrix_finetuned.png")

# ------------ Save Trained Model ------------
model_path = os.path.join(OUTPUT_DIR, "r3d18_sphar_best.pth")
torch.save(model.state_dict(), model_path)
print(f"💾 Model saved to {model_path}")