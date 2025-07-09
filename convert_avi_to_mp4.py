import os
import subprocess

def convert_all_avi_to_mp4(base_dir="tracked_videos"):
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".avi"):
                input_path = os.path.join(root, file)
                output_path = input_path.replace(".avi", ".mp4")
                if not os.path.exists(output_path):
                    print(f"🔁 Converting: {input_path}")
                    subprocess.call([
                        "ffmpeg", "-i", input_path,
                        "-vcodec", "libx264", "-acodec", "aac",
                        output_path
                    ])
    print("✅ All .avi files converted to .mp4")

convert_all_avi_to_mp4()