import os
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Paths
frames_root = os.path.join(os.path.dirname(__file__), 'training_videos')  # or 'testing_videos'
output_root = os.path.join(os.path.dirname(__file__), 'tracked_videos')
os.makedirs(output_root, exist_ok=True)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt', etc. for more accuracy

# Initialize DeepSORT
tracker = DeepSort(max_age=30)

for video_name in os.listdir(frames_root):
    video_folder = os.path.join(frames_root, video_name)
    if not os.path.isdir(video_folder):
        continue

    print(f"🎥 Processing: {video_name}")
    frame_files = sorted([
        f for f in os.listdir(video_folder) if f.endswith('.jpg')
    ])

    # Video writer setup
    sample_frame = cv2.imread(os.path.join(video_folder, frame_files[0]))
    h, w, _ = sample_frame.shape
    out_path = os.path.join(output_root, f"{video_name}_tracked.avi")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (w, h))

    for frame_file in frame_files:
        frame_path = os.path.join(video_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # Run YOLOv8 detection
        results = model(frame, verbose=False)[0]

        # Extract detection info
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls_id = int(box.cls[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, str(cls_id)))

        # Update DeepSORT
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw boxes
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w_box, h_box = track.to_ltrb()
            cv2.rectangle(frame, (int(l), int(t)), (int(l + w_box), int(t + h_box)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (int(l), int(t) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        writer.write(frame)

    writer.release()
    print(f"✅ Saved: {out_path}")

print("🎉 All videos processed with YOLOv8 + DeepSORT.")
