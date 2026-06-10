import os
import subprocess
import sys

def select_and_run():
    video_dir = "testVideo"
    script_name = "action_recognizer.py"
    
    print("--- Single Video Action Recognizer (LSTM Model) ---")
    print(f"Using Python: {sys.executable}")
    print(f"Default video folder: {video_dir}\n")
    
    # 1. Get video list
    if not os.path.exists(video_dir):
        print(f"Error: Directory '{video_dir}' not found.")
        return

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    videos = [f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)]
    
    if not videos:
        print(f"No video files found in '{video_dir}'.")
        return

    # Sort videos numerically if possible, otherwise alphabetically
    try:
        videos.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        videos.sort()

    print("Available videos:")
    for i, v in enumerate(videos):
        print(f"[{i+1}] {v}")
    
    # 2. Select Video
    try:
        val = input("\nEnter video number or filename (e.g. 1 or 1.mp4): ").strip()
        if not val:
            print("No input provided.")
            return
            
        target_video = ""
        if val.isdigit():
            idx = int(val) - 1
            if 0 <= idx < len(videos):
                target_video = os.path.join(video_dir, videos[idx])
            else:
                print("Invalid number.")
                return
        else:
            # Check if it's a filename in testVideo
            potential_path = os.path.join(video_dir, val)
            if os.path.isfile(potential_path):
                target_video = potential_path
            elif os.path.isfile(val):
                target_video = val
            else:
                print(f"Could not find video: {val}")
                return
    except ValueError:
        print("Invalid input.")
        return

    # 3. Execute
    print(f"\n>>> Running LSTM recognizer on: {target_video}")
    command = [sys.executable, script_name, "--video", target_video] + sys.argv[1:]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    select_and_run()
