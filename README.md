# HRI Human Motion Recognizer (PyTorch LSTM)

A real-time deep learning pipeline that classifies human motion trajectories and body postures in Human-Robot Interaction (HRI) scenarios. The system processes video streams (webcam or video files), extracts skeletal keypoints using MediaPipe, and evaluates temporal sequences using a trained PyTorch LSTM network.

---

## 📋 Project Overview

### **Model Architecture & Engines**
A hybrid real-time inference pipeline consisting of:
- **MediaPipe Pose Engine:** High-speed skeletal landmark detection (33 body keypoints).
- **PyTorch LSTM Sequence Model:** A 2-layer Recurrent Neural Network trained on keypoint velocity sequences to classify 9 distinct HRI motion trajectories.
- **Rule-based Pose Classifier:** Geometric analysis for basic postures (Standing, Sitting, Lying, Crouching).

### **Objective**
To develop a comprehensive HRI monitoring system capable of understanding:
1. **Pose:** Physical posture (e.g., "Sitting", "Standing").
2. **Motion:** Directional trajectory and velocity in 3D relative to the camera (e.g., "Approaching", "Backing Away", "Fast Toward").

### **Dataset**
- **Synthetic HRI Motion Dataset:** A custom-generated dataset of 450 keypoint sequences representing 9 different motion trajectories (50 sequences per class) with Gaussian noise, trajectory scaling, and realistic frame rates, specifically tailored for HRI scenarios.

### **Techniques Used**
- **Skeletal Sequence Modeling:** LSTM network processing sequential frame data to learn temporal relationships in movement.
- **Feature Scaling:** Velocities scaled by **100.0** to match the neural network input range and resolve vanishing gradient problems.
- **Temporal Smoothing:** Sliding window keypoint buffer for live, flicker-free inference.

---

## 📈 Results & Performance

### **Accuracies**
- **PyTorch LSTM Motion Model:** **100.0%** classification accuracy, precision, recall, and F1-score on the evaluation dataset.
- **Pose Tracking:** High-precision skeletal tracking in real-time.

### **Training Process**
1. **Velocity Extraction:** Frame-to-frame coordinate differences computed for all 33 pose landmarks.
2. **Feature Normalization:** Velocities scaled by `100.0` to bring inputs to a standard variance (0.5 to 1.5), optimizing gradient flow.
3. **Sequence Alignment:** Padding and truncation applied to align all sequences to exactly 29 frames.
4. **LSTM Training:** Trained for 50 epochs using Cross-Entropy Loss, Adam optimizer, and `ReduceLROnPlateau` learning rate scheduling, saving the best checkpoint based on validation loss.

---

## 🎓 Viva Section: Expected Questions & Answers

**1. Why use an LSTM model instead of a simple threshold-based heuristic?**
> Heuristics (hardcoded velocity thresholds) are fragile, prone to noise, and scale poorly when dealing with multi-axis motions. An LSTM learns the underlying temporal patterns of motion directly from data, making the classifier robust to individual movement variations, frame rate fluctuations, and coordinate noise.

**2. Why scale velocities by a factor of 100.0?**
> Raw keypoint coordinates are normalized between 0.0 and 1.0. The frame-to-frame differences (velocities) are extremely small (on the order of $10^{-3}$). When fed directly into an LSTM, these tiny values lead to vanishing gradients during backpropagation. Multiplying them by 100.0 brings the features to a standard variance range ($0.2$ to $1.2$), enabling smooth gradient flow and stable convergence.

**3. How does the system handle different sequence lengths?**
> The LSTM network expects a fixed sequence length. The dataset loader and real-time inference loop use a padding/truncation block. Sequences shorter than 29 frames are zero-padded, and sequences longer than 29 frames are truncated to fit exactly.

**4. How does the system handle "Depth" (Z-axis) with a standard 2D webcam?**
> MediaPipe estimates a "relative depth" based on the size and scale of the detected pose relative to the hip center. The LSTM tracks these Z-value velocities over time, enabling it to distinguish between depth-based motions like "Approaching" (increasing coordinate size) and "Backing Away" (decreasing size).

**5. What is the role of the sliding window queue?**
> The sliding window (a double-ended queue with a capacity of 30 frames) maintains a continuous temporal context. This allows the system to feed a smooth sequence of 29 velocity steps to the LSTM at every frame, producing flicker-free, real-time predictions.

---

## 🎮 Usage Guide

To run the system, make sure to execute commands using the virtual environment Python interpreter (`.\env\Scripts\python.exe`):

### 1. Run LSTM Inference on a Single Video
List and select from available test videos:
```powershell
.\env\Scripts\python.exe run_single_video.py
```
Or specify path directly:
```powershell
.\env\Scripts\python.exe action_recognizer.py --video testVideo/1.mp4
```

### 2. Save Visual Output Video (Dashboard Overlay)
Run the model on a video and write the annotated dashboard to a file:
```powershell
.\env\Scripts\python.exe run_and_save_video.py
```
Or run directly:
```powershell
.\env\Scripts\python.exe action_recognizer.py --video testVideo/1.mp4 --save output/output_1.avi
```

### 3. Run LSTM Inference on Live Webcam
```powershell
.\env\Scripts\python.exe action_recognizer.py --webcam
```

### 4. Batch Process All Videos in a Folder
Process all videos in `testVideo/` and save to the `output/` folder:
```powershell
.\env\Scripts\python.exe run_all_and_save.py
```

### 5. Stream and Visualize All Test Videos Sequentially
```powershell
.\env\Scripts\python.exe run_test_videos.py
```

---

## 📁 Project Structure
- `action_recognizer.py`: Main real-time inference pipeline using the LSTM neural network and visual dashboard overlay.
- `model_train/`:
  - `1_prepare_dataset.ipynb`: Jupyter notebook generating the synthetic trajectory dataset.
  - `2_train_and_evaluate.ipynb`: Jupyter notebook defining the LSTM model, training, and generating validation curves.
  - `models/`: Contains model weights (`motion_lstm_best.pth`), parameters (`model_config.json`), evaluation report (`evaluation_report.json`), and validation curves.
- `run_single_video.py` / `run_and_save_video.py` / `run_all_and_save.py` / `run_test_videos.py` / `run_custom_list.py`: Runners for the LSTM model.
- `testVideo/`: Input directory for test video files.
- `output/`: Directory for annotated results.
- `env/`: Virtual environment with dependencies.
- `notes.txt`: Notepad file detailing the exact step-by-step detection process and the LSTM specifications.

---

## 🧠 Theoretical Background (Viva Cheat Sheet)

### 1. Recurrent Neural Networks (RNN) & LSTM
- **RNN:** A neural network with loops, allowing information to persist across sequence steps.
- **LSTM (Long Short-Term Memory):** A special RNN architecture that resolves vanishing gradients using a **cell state** and three gates:
  - *Forget Gate:* Decides what information to discard from the cell state.
  - *Input Gate:* Decides what new information to store in the cell state.
  - *Output Gate:* Decides what information from the cell state to output as the hidden state.

### 2. MediaPipe (Pose Estimation)
- MediaPipe uses **BlazePose**, a single-shot detector optimized for real-time tracking on standard hardware.
- It returns 33 landmarks representing key skeletal joints. Reducing raw pixel streams to structural points drastically increases processing efficiency.

### 3. Feature Scaling
- Machine learning models operate best when inputs have zero mean and unit variance. Because frame differences (velocities) are extremely small, multiplying by **100.0** maps them to a normal range, enabling the network to learn and backpropagate gradients efficiently.
