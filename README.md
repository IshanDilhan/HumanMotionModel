# Human Action & Motion Recognizer

A robust computer vision system that combines skeletal tracking, deep learning, and geometric analysis to classify human actions, poses, and movement patterns in real-time.

---

## 📋 Project Overview

### **Model Architecture & Engines**
A Hybrid Multi-Stage Recognition System:
- **MediaPipe Pose Engine:** High-speed skeletal landmark detection (33 body keypoints).
- **PyTorch LSTM Sequence Model:** 2-layer Recurrent Neural Network trained on keypoint velocities to classify 9 Human-Robot Interaction (HRI) motions with high temporal consistency and accuracy.
- **ResNet50 (Residual Network):** Fine-tuned on the **Stanford40 Actions** dataset for static action context classification (optional).
- **Rule-based Pose Classifier:** Geometric analysis for basic postures (Standing, Sitting, Lying, Crouching).

### **Objective**
To develop a comprehensive human-robot interaction (HRI) monitoring system capable of understanding:
1. **Pose:** Physical posture (e.g., "Sitting", "Standing").
2. **Motion:** Directional trajectory and velocity in 3D relative to the camera (e.g., "Approaching", "Backing Away", "Fast Toward").
3. **Action:** Static context classification (e.g., "Reading", "Phoning").

### **Dataset**
- **Stanford 40 Actions:** 9,532 images across 40 diverse categories.
- **Synthetic HRI Motion Dataset:** A custom-generated dataset of 450 keypoint sequences representing 9 different motion trajectories (50 sequences per class) with Gaussian noise, trajectory scaling, and realistic frame rates, specifically tailored for HRI scenarios.

### **Techniques Used**
- **Skeletal Sequence Modeling:** LSTM network processing sequential frame data to learn temporal relationships in movement.
- **Feature Scaling:** Velocities scaled by 100.0 to match the neural network input range and resolve vanishing gradient problems.
- **Data Augmentation:** Gaussian noise injections, translation scaling, and dynamic speed variations applied to synthetic coordinates.
- **Transfer Learning:** ImageNet-pretrained ResNet50 fine-tuned on Stanford40.
- **Temporal Smoothing:** Sliding window keypoint buffer for live, flicker-free inference.

---

## 📈 Results & Performance

### **Accuracies**
- **PyTorch LSTM Motion Model:** **100.0%** classification accuracy, precision, recall, and F1-score on the evaluation dataset.
- **ResNet50 Action Model:** ~**88.5%** accuracy on the Stanford40 test set.
- **Pose Tracking:** High-precision skeletal tracking in real-time.

### **Training Process**
1. **Velocity Extraction:** Frame-to-frame coordinate differences computed for all 33 pose landmarks.
2. **Feature Normalization:** Velocities scaled by `100.0` to bring inputs to a standard variance (0.5 to 1.5), optimizing gradient flow.
3. **Sequence Alignment:** Padding and truncation applied to align all sequences to exactly 29 frames.
4. **LSTM Training:** Trained for 50 epochs using Cross-Entropy Loss, Adam optimizer, and `ReduceLROnPlateau` learning rate scheduling, saving the best checkpoint based on validation loss.

---

## 🎓 Viva Section: Expected Questions & Answers

**1. Why use a hybrid approach instead of just one Deep Learning model?**
> MediaPipe provides extremely fast skeletal tracking on the CPU, which is perfect for movement direction. ResNet50 is much better at "understanding" the visual context of an action (like holding a phone) which skeletal points alone might miss. Combining them gives us the best of both worlds: speed and context.

**2. How does the system handle "Depth" (Z-axis) with a standard 2D webcam?**
> MediaPipe estimates a "relative depth" based on the pose size and perspective. The Motion Analyser tracks this Z-value over time. If Z is decreasing significantly, the system classifies the motion as "Approaching."

**3. What is Transfer Learning and why did you use it here?**
> Transfer Learning is taking a model trained on a massive dataset (ImageNet) and repurposing it for a smaller, specific task (Stanford40). It saves days of training time and achieves much higher accuracy than training from scratch.

**4. How is the 'Center of Mass' (CoM) calculated?**
> We average the coordinates of the shoulders and hips (Landmarks 11, 12, 23, 24). This provides a stable point that represents the main body trunk, reducing "noise" from waving hands or moving feet.

**5. Why ResNet50 specifically?**
> ResNet (Residual Network) solves the "vanishing gradient" problem using skip connections. ResNet50 is deep enough to capture complex actions but light enough to run near real-time on modern hardware.

---

## 🎮 Usage Guide

To run the system, make sure to execute commands using the virtual environment Python interpreter (`.\env\Scripts\python.exe`):

### 1. Run LSTM Inference on a Single Video
List and select from available test videos:
```powershell
.\env\Scripts\python.exe run_single_video_lstm.py
```
Or specify path directly:
```powershell
.\env\Scripts\python.exe action_recognizer_lstm.py --video testVideo/1.mp4
```

### 2. Save Visual Output Video (Dashboard Overlay)
Run the model on a video and write the annotated dashboard to a file:
```powershell
.\env\Scripts\python.exe run_and_save_video_lstm.py
```
Or run directly:
```powershell
.\env\Scripts\python.exe action_recognizer_lstm.py --video testVideo/1.mp4 --save output/output_lstm_1.avi
```

### 3. Run LSTM Inference on Live Webcam
```powershell
.\env\Scripts\python.exe action_recognizer_lstm.py --webcam
```

### 4. Batch Process All Videos in a Folder
Process all videos in `testVideo/` and save to the `output/` folder:
```powershell
.\env\Scripts\python.exe run_all_and_save_lstm.py
```

### 5. Run Original Heuristic Recognizer
To run the original velocity-heuristic version:
```powershell
.\env\Scripts\python.exe run_single_video.py
```

---

## 📁 Project Structure
- `action_recognizer_lstm.py`: Main real-time inference pipeline using the LSTM neural network and visual dashboard overlay.
- `action_recognizer.py`: Original heuristic-based motion recognizer.
- `model_train/`:
  - `1_prepare_dataset.ipynb`: Jupyter notebook generating the synthetic trajectory dataset.
  - `2_train_and_evaluate.ipynb`: Jupyter notebook defining the LSTM model, training, and generating validation curves.
  - `models/`: Contains model weights (`motion_lstm_best.pth`), parameters (`model_config.json`), evaluation report (`evaluation_report.json`), and validation curves.
- `run_single_video_lstm.py` / `run_and_save_video_lstm.py` / `run_all_and_save_lstm.py`: Convenient runners for the LSTM model.
- `testVideo/`: Input directory for test video files.
- `output/`: Directory for annotated results.
- `env/`: Virtual environment with dependencies.

---

## 🧠 Theoretical Background (Viva Cheat Sheet)

If you are asked about the "Why" and "How" of the technology, use these simplified explanations:

### 1. Machine Learning (ML) vs. Deep Learning (DL)
- **Machine Learning:** Think of this as the "Geometric Motion Analyser" in our code. We (the humans) write specific rules (e.g., "if velocity > 0.05, then it is Fast"). The computer follows these hard-coded mathematical rules.
- **Deep Learning:** This is the **ResNet50** part. We don't give the computer rules. Instead, we show it 9,000+ images from the **Stanford40** dataset. The computer "learns" to recognize a "Guitar" or a "Phone" by identifying patterns in pixels across many layers.

### 2. Convolutional Neural Networks (CNN)
The ResNet50 model is a **CNN**. CNNs are special neural networks designed to "see" like humans:
- **Lower Layers:** Detect simple things like edges, lines, and corners.
- **Middle Layers:** Detect shapes like circles, rectangles, or textures.
- **Higher Layers:** Detect complex objects like a human face, a hand, or a bicycle.

### 3. Residual Networks (ResNet) & Skip Connections
Standard deep networks often "forget" information as it passes through too many layers (called the *Vanishing Gradient* problem).
- **ResNet** introduced **Skip Connections** (or Identity Shortcuts). These allow information to "skip" some layers, ensuring the model stays accurate even when it is very deep (50 layers in our case).

### 4. Transfer Learning
We didn't train ResNet50 from scratch (which would take weeks and millions of images).
- We used **Transfer Learning**: We took a model already trained on **ImageNet** (1.2 million images) so it already knew how to see "edges" and "shapes." 
- We only "fine-tuned" the very last part of the model to recognize our specific **40 actions**.

### 5. MediaPipe (Pose Estimation)
MediaPipe uses a model called **BlazePose**.
- It is optimized for mobile/real-time use.
- It doesn't just look at pixels; it predicts a **Skeletal Topology** (33 landmarks).
- This is much more efficient than traditional image processing because it reduces a whole person into just 33 (x, y, z) points.

### 6. Velocity & Temporal Smoothing
To prevent the labels from flickering (e.g., jumping between "Stationary" and "Moving"), we use a **Sliding Window (deque)** of 12 frames.
- We calculate the average movement over these 12 frames. This is called **Temporal Smoothing**. It makes the detection look "stable" and "smooth" to the user.
