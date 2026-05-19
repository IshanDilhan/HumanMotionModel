"""
Real-time Human Motion Recognizer using PyTorch LSTM
=====================================================
Uses:
  - MediaPipe Pose  → 33 body keypoints
  - PyTorch LSTM    → Trained on 9 HRI motion classes (100% accuracy)

Usage:
    # Webcam
    python action_recognizer.py --webcam

    # Video file
    python action_recognizer.py --video path/to/video.mp4

    # Video file + save output
    python action_recognizer.py --video path/to/video.mp4 --save output.avi
"""

import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# MOTION LABELS
# ─────────────────────────────────────────────────────────────────────────────
MOTION_LABELS = [
    "Stationary",
    "Approaching",
    "Backing Away",
    "Fast Across",
    "Slow Approach",
    "Fast Toward",
    "Approach + Stop",
    "Minimal",
    "Across",
]

# Colour per label (BGR)
LABEL_COLORS = {
    "Stationary":     (160, 160, 160),  # Grey
    "Approaching":    (0,   210, 255),  # Yellow-orange
    "Backing Away":   (255, 120,   0),  # Blue
    "Fast Across":    (0,   255, 100),  # Green
    "Slow Approach":  (50,  220, 255),  # Light yellow
    "Fast Toward":    (0,    80, 255),  # Orange-red
    "Approach + Stop":(220, 255,   0),  # Lime
    "Minimal":        (130, 130, 130),  # Dark Grey
    "Across":         (0,   255, 200),  # Turquoise
}

# ─────────────────────────────────────────────────────────────────────────────
# LSTM MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
class MotionLSTM(nn.Module):
    def __init__(self, input_size=99, hidden_size=64, num_layers=2, num_classes=9, dropout=0.3):
        super(MotionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        x = x.reshape(batch_size, seq_len, -1)  # (batch, seq_len, 99)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        out = self.fc(last_hidden)
        return out

# ─────────────────────────────────────────────────────────────────────────────
# POSE CLASSIFIER (Rule-based)
# ─────────────────────────────────────────────────────────────────────────────
class PoseClassifier:
    def classify(self, landmarks) -> str:
        if landmarks is None:
            return "Unknown"
        lm = landmarks.landmark

        # Vertical span: nose (0) vs mid-hip (23/24)
        nose_y     = lm[0].y
        hip_y      = (lm[23].y + lm[24].y) / 2
        knee_y     = (lm[25].y + lm[26].y) / 2
        ankle_y    = (lm[27].y + lm[28].y) / 2
        shoulder_y = (lm[11].y + lm[12].y) / 2

        body_height = abs(ankle_y - nose_y) + 1e-6
        torso_ratio = abs(hip_y - shoulder_y) / body_height
        leg_ratio   = abs(ankle_y - knee_y)  / body_height

        if body_height < 0.25:
            return "Lying"
        if torso_ratio > 0.35 and leg_ratio < 0.15:
            return "Sitting"
        if knee_y < hip_y + 0.05:
            return "Crouching"
        return "Standing"

# ─────────────────────────────────────────────────────────────────────────────
# PREMIUM SIDEBAR DASHBOARD DRAWING
# ─────────────────────────────────────────────────────────────────────────────
def draw_dashboard(frame, pose_label, motion_label, confidence, probabilities, fps, landmarks, mp_drawing, mp_pose):
    h, w = frame.shape[:2]
    sidebar_w = 320
    
    # Draw MediaPipe skeleton on frame
    if landmarks:
        mp_drawing.draw_landmarks(
            frame, landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 100), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2),
        )
        
    # Create final canvas
    canvas = np.zeros((h, w + sidebar_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    
    # Draw dark dashboard panel
    cv2.rectangle(canvas, (w, 0), (w + sidebar_w, h), (20, 24, 33), -1)
    cv2.line(canvas, (w, 0), (w, h), (43, 52, 69), 2)
    
    # Title
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "HRI MOTION ANALYZER", (w + 20, 35), font, 0.65, (230, 235, 245), 2, cv2.LINE_AA)
    cv2.line(canvas, (w + 20, 48), (w + sidebar_w - 20, 48), (60, 75, 100), 1)
    
    # Stats
    cv2.putText(canvas, f"FPS: {fps:.1f}", (w + 20, 75), font, 0.5, (150, 160, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Pose: {pose_label}", (w + 140, 75), font, 0.5, (150, 160, 180), 1, cv2.LINE_AA)
    
    y_offset = 110
        
    # Active Motion Badge
    cv2.putText(canvas, "ACTIVE MOTION:", (w + 20, y_offset), font, 0.55, (100, 120, 150), 1, cv2.LINE_AA)
    y_offset += 25
    
    color = LABEL_COLORS.get(motion_label, (255, 255, 255))
    cv2.rectangle(canvas, (w + 20, y_offset - 20), (w + sidebar_w - 20, y_offset + 10), color, -1)
    text_color = (255, 255, 255) if np.mean(color) < 120 else (20, 20, 20)
    cv2.putText(canvas, f"{motion_label} ({confidence*100:.1f}%)", (w + 30, y_offset - 1), font, 0.55, text_color, 2, cv2.LINE_AA)
    
    y_offset += 40
    cv2.line(canvas, (w + 20, y_offset), (w + sidebar_w - 20, y_offset), (60, 75, 100), 1)
    y_offset += 25
    
    # Class Probabilities Section
    cv2.putText(canvas, "CLASS CONFIDENCE:", (w + 20, y_offset), font, 0.55, (100, 120, 150), 1, cv2.LINE_AA)
    y_offset += 25
    
    bar_max_w = 150
    for i, label in enumerate(MOTION_LABELS):
        prob = probabilities[i] if probabilities is not None else 0.0
        cv2.putText(canvas, label[:14], (w + 20, y_offset), font, 0.45, (170, 180, 200), 1, cv2.LINE_AA)
        
        # Draw bar
        bar_w = int(prob * bar_max_w)
        bar_color = LABEL_COLORS.get(label, (100, 100, 100))
        cv2.rectangle(canvas, (w + 130, y_offset - 10), (w + 130 + bar_max_w, y_offset + 2), (40, 48, 64), -1)
        if bar_w > 0:
            cv2.rectangle(canvas, (w + 130, y_offset - 10), (w + 130 + bar_w, y_offset + 2), bar_color, -1)
            
        cv2.putText(canvas, f"{prob*100:.0f}%", (w + 135 + bar_max_w, y_offset), font, 0.4, (120, 130, 150), 1, cv2.LINE_AA)
        y_offset += 22
        
    return canvas

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION ROUTINE
# ─────────────────────────────────────────────────────────────────────────────
def run(source, model_path, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load LSTM Model
    print(f"[INFO] Loading LSTM model from: {model_path}")
    model = MotionLSTM(input_size=99, hidden_size=64, num_layers=2, num_classes=9, dropout=0.3)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model weights: {e}")
        sys.exit(1)
        
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_classifier = PoseClassifier()
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)
        
    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    
    if save_path:
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sidebar_w = 320
        # Canvas size is frame width + sidebar width
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(save_path, fourcc, fps_cam, (fw + sidebar_w, fh))
        print(f"[INFO] Saving output to: {save_path}")
        
    print("[INFO] Press 'Q' to quit.")
    
    # Keep sliding window queue of last 30 frames keypoints
    keypoints_queue = deque(maxlen=30)
    
    frame_idx = 0
    t_prev = time.time()
    fps_display = 0.0
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as pose:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True
            
            landmarks = results.pose_landmarks
            
            # Predict pose
            pose_label = pose_classifier.classify(landmarks)
            
            # Process sliding window keypoints for LSTM
            if landmarks:
                # Extract 33 landmark points
                pts = np.zeros((33, 3))
                for i, lm in enumerate(landmarks.landmark):
                    pts[i] = [lm.x, lm.y, lm.z]
                keypoints_queue.append(pts)
            else:
                keypoints_queue.clear()
                
            # Run model inference on sliding window
            motion_label = "Stationary"
            confidence = 1.0
            probabilities = np.zeros(9)
            probabilities[0] = 1.0
            
            if len(keypoints_queue) >= 2:
                keypoints_arr = np.array(keypoints_queue)  # (seq_len, 33, 3)
                # Compute frame-to-frame velocity features
                velocities = np.diff(keypoints_arr, axis=0)  # (seq_len - 1, 33, 3)
                
                # IMPORTANT: Scale velocities by 100.0, matching the scale of the clean dataset
                scaled_vel = velocities * 100.0
                
                # Pad/truncate to exactly 29 frames
                max_len = 29
                seq = scaled_vel
                if seq.shape[0] < max_len:
                    pad_len = max_len - seq.shape[0]
                    seq = np.pad(seq, ((0, pad_len), (0, 0), (0, 0)), mode='constant')
                else:
                    seq = seq[:max_len]
                    
                # Convert to tensor and run model
                x = torch.FloatTensor(seq).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(x)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    
                pred_idx = np.argmax(probs)
                motion_label = MOTION_LABELS[pred_idx]
                confidence = probs[pred_idx]
                probabilities = probs
            elif not landmarks:
                # If no landmarks, default values
                probabilities = np.zeros(9)
                probabilities[0] = 1.0
                motion_label = "Stationary"
                confidence = 1.0
                
            # Measure FPS
            now = time.time()
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            
            # Draw premium dashboard
            canvas = draw_dashboard(
                frame, pose_label, motion_label,
                confidence, probabilities, fps_display,
                landmarks, mp_drawing, mp_pose
            )
            
            if writer:
                writer.write(canvas)
                
            cv2.imshow("Human Motion Recognizer LSTM", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Execution finished successfully.")

# ─────────────────────────────────────────────────────────────────────────────
# CLI ARGUMENTS PARSING
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Human Motion Recognizer using PyTorch LSTM & MediaPipe Pose"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--webcam", action="store_true", help="Use default webcam")
    src.add_argument("--video", type=str, metavar="PATH", help="Path to video file")
    src.add_argument("--camera", type=int, metavar="INDEX", help="External camera device index")
    
    parser.add_argument("--model-path", type=str, metavar="PATH",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "model_train", "models", "motion_lstm_best.pth"),
                        help="Path to trained LSTM model weights")
    parser.add_argument("--save", type=str, metavar="OUT", help="Save output video path (.avi)")
    
    args = parser.parse_args()
    
    if args.webcam:
        source = 0
    elif args.camera is not None:
        source = args.camera
    else:
        source = args.video
        
    run(source, args.model_path, save_path=args.save)
