import os
import cv2
import numpy as np

# Path to the extracted frames (e.g., from training_videos or testing_videos)
frames_root = os.path.join(os.path.dirname(__file__), 'training_videos')  # or 'testing_videos'

# Resize dimensions (optional)
target_size = (224, 224)  # You can set this to match your model's input

# Iterate over each video folder
for video_name in os.listdir(frames_root):
    video_folder = os.path.join(frames_root, video_name)

    if not os.path.isdir(video_folder):
        continue

    print(f"📁 Reading frames from: {video_name}")
    
    # List and sort frame files
    frame_files = sorted([
        f for f in os.listdir(video_folder) if f.endswith('.jpg')
    ])

    for frame_file in frame_files:
        frame_path = os.path.join(video_folder, frame_file)

        # Read the frame using OpenCV
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Skipping unreadable frame: {frame_path}")
            continue

        # Resize the frame
        frame_resized = cv2.resize(frame, target_size)

        # Convert BGR to RGB (if needed)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Normalize pixel values (0 to 1)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0

        # Now frame_normalized is ready for model input
        # You can store it in a list, feed it to a model, etc.
        # Example:
        # model_input.append(frame_normalized)

    print(f"Done processing: {video_name}\n")

print("All frames preprocessed.")
