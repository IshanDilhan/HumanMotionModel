import os
import subprocess
import sys

# ==========================================================
# MANUALLY ADD YOUR VIDEOS HERE
# Just put the filename (e.g., "5.mp4") if it's in testVideo
# ==========================================================
CUSTOM_VIDEO_LIST = [
    "1.mp4",
    "2.mp4",
    "3.mp4"
]

def run_custom():
    video_dir = "testVideo"
    script_name = "action_recognizer.py"
    
    if not os.path.exists(video_dir):
        print(f"Error: Directory '{video_dir}' not found.")
        return

    # Filter out empty or missing videos
    valid_videos = []
    for item in CUSTOM_VIDEO_LIST:
        path = os.path.join(video_dir, item)
        if os.path.isfile(path):
            valid_videos.append(path)
        else:
            print(f"Warning: Video file '{item}' not found in '{video_dir}'. Skipping.")

    if not valid_videos:
        print("No valid videos to process in CUSTOM_VIDEO_LIST.")
        return

    print(f"Starting custom run of {len(valid_videos)} videos...")

    for video_path in valid_videos:
        print(f"\n========================================")
        print(f"Running LSTM model on: {os.path.basename(video_path)}")
        print(f"========================================")
        
        command = [sys.executable, script_name, "--video", video_path] + sys.argv[1:]
        
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Process exited with error on {os.path.basename(video_path)}: {e}")
        except KeyboardInterrupt:
            print("\nExecution stopped by user.")
            break

    print("\nCustom run completed.")

if __name__ == "__main__":
    run_custom()
