import os
import subprocess

# Path to Avenue Dataset in Downloads
base_dataset_dir = os.path.expanduser("~/Downloads/Avenue Dataset")
output_base_dir = os.path.abspath(os.path.dirname(__file__))

# Define subfolders to process
video_subfolders = ['training_videos', 'testing_videos']

for subfolder in video_subfolders:
    video_dir = os.path.join(base_dataset_dir, subfolder)
    output_dir = os.path.join(output_base_dir, subfolder)
    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir) if f.endswith('.avi')]

    if not video_files:
        print(f"⚠️ No .avi files found in '{subfolder}'. Skipping...")
        continue

    print(f"📂 Processing {len(video_files)} video(s) in '{subfolder}'...")

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        ffmpeg_cmd = [
            "ffmpeg",
            "-i", video_path,
            os.path.join(video_output_dir, "frame_%04d.jpg")
        ]

        print(f"⏳ Extracting from: {video_file}")
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print(f" Done: {video_file} ➝ {video_output_dir}\n")
        except subprocess.CalledProcessError:
            print(f"Error processing {video_file}\n")

print("Frame extraction completed for all videos.")
