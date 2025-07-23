import os
import os
import subprocess
import logging

# Setup logging
logging.basicConfig(filename='frame_extraction.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set SPHAR dataset directory
base_dataset_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "SPHAR-Dataset", "videos")
if not os.path.exists(base_dataset_dir):
    logging.error(f"Video dataset not found at {base_dataset_dir}. Please ensure the SPHAR-Dataset/videos folder exists.")
    print(f"❌ Video dataset not found at {base_dataset_dir}. Please verify the SPHAR-Dataset/videos folder.")
    exit(1)

output_base_dir = os.path.abspath(os.path.dirname(__file__))

# Configure video limits and event classification
selected_event_types = ['sitting', 'stealing', 'vandalizing', 'running', 'carcrash', 'kicking', 'neutral']  # Event types to process
max_videos_per_event = 30  # Maximum number of videos per event type
event_classification = {
    'stealing': 'anomaly', 'vandalizing': 'anomaly', 'carcrash': 'anomaly', 'kicking': 'anomaly',  # Anomalous events
    'sitting': 'normal', 'running': 'normal', 'neutral': 'normal'  # Normal events
}  # Map event types to anomaly or normal

# Define valid video extensions
video_extensions = ['.avi', '.mp4', '.mov']

# Dynamically detect and filter video subfolders based on event types
video_subfolders = []
for event_type in selected_event_types:
    event_path = os.path.join(base_dataset_dir, event_type)
    if os.path.exists(event_path):
        video_files = [f for f in os.listdir(event_path) if f.endswith(tuple(video_extensions))]
        if video_files:
            video_subfolders.append((event_path, event_type))
        else:
            logging.warning(f"No video files found in '{event_path}'. Skipping...")
            print(f"⚠️ No video files found in '{event_path}'. Skipping...")
    else:
        logging.warning(f"Event type folder '{event_type}' not found in {base_dataset_dir}. Skipping...")
        print(f"⚠️ Event type folder '{event_type}' not found in {base_dataset_dir}. Skipping...")

if not video_subfolders:
    logging.error(f"No valid event type subfolders found in {base_dataset_dir}. Please check folder names.")
    print(f"❌ No valid event type subfolders found in {base_dataset_dir}. Please check folder names.")
    exit(1)

for subfolder, event_type in video_subfolders:
    # Determine the event category (anomaly or normal)
    event_category = event_classification.get(event_type, 'unknown')
    if event_category == 'unknown':
        logging.warning(f"Event type '{event_type}' not classified. Skipping...")
        print(f"⚠️ Event type '{event_type}' not classified. Skipping...")
        continue

    output_dir = os.path.join(output_base_dir, "frames", event_category, event_type)
    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(subfolder) if f.endswith(tuple(video_extensions))][:max_videos_per_event]

    if not video_files:
        logging.warning(f"No video files found in '{subfolder}' after limiting. Skipping...")
        print(f"⚠️ No video files found in '{subfolder}' after limiting. Skipping...")
        continue

    print(f"📂 Processing {len(video_files)} video(s) in '{event_type}' ({event_category})...")

    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        logging.error("FFmpeg is not found in the system PATH. Please verify the PATH setting or reinstall FFmpeg.")
        print("❌ FFmpeg is not found in the system PATH. Please verify the PATH setting (e.g., C:\\ffmpeg\\bin) or reinstall FFmpeg.")
        exit(1)
    except PermissionError:
        logging.error("Permission denied accessing FFmpeg. Please run this script as administrator.")
        print("❌ Permission denied accessing FFmpeg. Please run this script as administrator.")
        exit(1)

    for video_file in video_files:
        video_path = os.path.join(subfolder, video_file)
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        # FFmpeg command with frame rate control
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", "fps=1",
            os.path.join(video_output_dir, "frame_%04d.jpg")
        ]

        print(f"⏳ Extracting from: {video_file}")
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            logging.info(f"Successfully extracted frames from {video_file} to {video_output_dir}")
            print(f"✅ Done: {video_file} ➝ {video_output_dir}\n")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error processing {video_file}: {e}")
            print(f"❌ Error processing {video_file}\n")

print("🎉 Frame extraction completed for all videos.")