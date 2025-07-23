import os
import shutil
import random
from pathlib import Path

# === CONFIGURATION ===
# Path to your extracted frame folders
input_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "frames")
output_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "split_frames")

splits = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}

random.seed(42)  # For reproducibility

def split_class(event_path, category, class_name):
    video_dirs = [d for d in os.listdir(event_path) if os.path.isdir(os.path.join(event_path, d))]
    total = len(video_dirs)
    random.shuffle(video_dirs)

    train_cutoff = int(splits['train'] * total)
    val_cutoff = train_cutoff + int(splits['val'] * total)

    split_map = {
        'train': video_dirs[:train_cutoff],
        'val': video_dirs[train_cutoff:val_cutoff],
        'test': video_dirs[val_cutoff:]
    }

    for split, videos in split_map.items():
        for vid in videos:
            src = os.path.join(event_path, vid)
            dest = os.path.join(output_root, split, category, class_name, vid)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copytree(src, dest, dirs_exist_ok=True)

    print(f"✅ {class_name}: {total} videos ➝ "
          f"{len(split_map['train'])} train, {len(split_map['val'])} val, {len(split_map['test'])} test")

# === MAIN ===
for category in ["anomaly", "normal"]:
    category_path = os.path.join(input_root, category)
    if not os.path.exists(category_path):
        continue

    for class_name in os.listdir(category_path):
        class_path = os.path.join(category_path, class_name)
        if os.path.isdir(class_path):
            split_class(class_path, category, class_name)

print("\n🎉 Splitting complete. Files stored in 'split_frames' folder.")