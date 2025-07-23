import os
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# === Paths ===
frames_root = os.path.join(os.path.dirname(__file__), 'split_frames')  # root folder with train/val/test splits
output_root = os.path.join(os.path.dirname(__file__), 'tracked_videos')
os.makedirs(output_root, exist_ok=True)

# === Load YOLOv8 model ===
model = YOLO('yolov8n.pt')

# === Initialize DeepSORT ===
tracker = DeepSort(max_age=30)

# === Process each split ===
for split in ['train', 'val', 'test']:
    split_path = os.path.join(frames_root, split)
    if not os.path.exists(split_path):
        print(f"⚠️ Split folder not found: {split}")
        continue

    for event_category in ['anomaly', 'normal']:
        category_path = os.path.join(split_path, event_category)
        if not os.path.exists(category_path) or not os.path.isdir(category_path):
            print(f"⚠️ Skipping {event_category} in {split}: not found or not a directory")
            continue

        for event_type in os.listdir(category_path):
            event_path = os.path.join(category_path, event_type)
            if not os.path.isdir(event_path):
                continue

            print(f"🎥 [{split}] Processing category: {event_category}, event type: {event_type}")
            for video_name in os.listdir(event_path):
                video_folder = os.path.join(event_path, video_name)
                if not os.path.isdir(video_folder):
                    continue

                print(f"🎥 [{split}] Processing: {video_name}")
                frame_files = sorted([
                    f for f in os.listdir(video_folder) if f.endswith('.jpg')
                ])

                if not frame_files:
                    print(f"⚠️ No frames found in {video_folder}")
                    continue
                sample_frame = cv2.imread(os.path.join(video_folder, frame_files[0]))
                if sample_frame is None:
                    print(f"❌ Failed to load sample frame in {video_folder}")
                    continue
                h, w, _ = sample_frame.shape

                # Output directory and video path
                out_dir = os.path.join(output_root, split, event_category, event_type)
                os.makedirs(out_dir, exist_ok=True)
                mp4_path = os.path.join(out_dir, f"{video_name}_tracked.mp4")

                # Write video
                mp4_writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (w, h))

                for frame_file in frame_files:
                    frame_path = os.path.join(video_folder, frame_file)
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        print(f"❌ Failed to load {frame_path}")
                        continue

                    # YOLOv8 detection
                    results = model(frame, verbose=False)[0]

                    # Extract detections
                    detections = []
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf)
                        cls_id = int(box.cls[0])
                        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, str(cls_id)))

                    # DeepSORT tracking
                    tracks = tracker.update_tracks(detections, frame=frame)

                    # Draw tracks
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        track_id = track.track_id
                        l, t, r, b = track.to_ltrb()
                        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID {track_id}", (int(l), int(t) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    mp4_writer.write(frame)

                mp4_writer.release()
                print(f"✅ [{split}] Saved: {mp4_path}")

print("🎉 Tracking completed for train, val, and test splits.")