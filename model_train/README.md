# Motion Classification Model Training

Complete pipeline to train LSTM model on MediaPipe keypoints for 9 motion types.

## Workflow

### Step 1: Download & Organize JHMDB Dataset
```bash
cd model_train
python 1_download_jhmdb.py
```

**What it does:**
- Checks if JHMDB is downloaded
- Maps action classes to your 9 motion types
- Creates organized symlinks

**Manual step:**
- Visit: http://jhmdb.org/
- Download "Renamed Folders" (~348MB)
- Extract to: `data/JHMDB/`

### Step 2: Extract MediaPipe Keypoints
```bash
python 2_extract_keypoints.py
```

**What it does:**
- Processes all videos in `data/selected_motions/`
- Extracts 33 keypoints per frame using MediaPipe
- Saves sequences as `.npy` files in `extracted_keypoints/`
- Output: `(n_frames, 33, 3)` arrays (x, y, z per keypoint)

**Time:** ~15-30 min (depends on dataset size)

### Step 3: Train LSTM Model (Jupyter Notebook)
```bash
jupyter notebook 3_train_motion_lstm.ipynb
```

**What it does:**
- Loads extracted keypoint sequences
- Computes velocity features (frame-to-frame differences)
- Maps JHMDB actions to your 9 motion labels
- Builds lightweight LSTM (50K-100K params)
- Trains for 50 epochs
- Saves best model to `models/motion_lstm_best.pth`

---

## 9 Motion Classes

| ID | Motion | Expected Accuracy |
|----|--------|-------------------|
| 0 | Stationary | High |
| 1 | Approaching | Medium-High |
| 2 | Backing Away | Medium-High |
| 3 | Across | Medium |
| 4 | Slow Approach | Medium |
| 5 | Fast Toward | Medium |
| 6 | Fast Across | Medium |
| 7 | Approach + Stop | Low-Medium |
| 8 | Minimal | Low |

---

## Project Structure

```
model_train/
├── data/                          # Raw videos
│   ├── JHMDB/                     # Downloaded dataset
│   └── selected_motions/          # Organized by motion type
├── extracted_keypoints/           # Extracted .npy sequences
│   ├── approaching/
│   ├── backing_away/
│   ├── across/
│   └── ...
├── models/                        # Saved models
│   ├── motion_lstm_best.pth       # Best model
│   └── motion_model_config.json   # Model config
├── 1_download_jhmdb.py
├── 2_extract_keypoints.py
└── 3_train_motion_lstm.ipynb
```

---

## Expected Results

- **Keypoints extracted:** 500-1000+ sequences
- **Training time:** 5-10 min (GPU) / 20-30 min (CPU)
- **Model size:** ~500KB
- **Accuracy:** 70-85% (depends on dataset quality)

---

## Next Steps

After training:
1. **Integrate** trained model into `action_recognizer.py`
2. **Test** on real videos
3. **Deploy** on Jetson Orin

---

## Troubleshooting

**KeyError: JHMDB data not found**
- Make sure `data/JHMDB/` exists
- Run `1_download_jhmdb.py` first

**Low accuracy**
- Increase `epochs` in notebook
- Add more training data
- Tune learning rate: `lr=0.0005` or `lr=0.002`

**Out of memory**
- Reduce batch size: `batch_size=8`
- Reduce `max_len=40` in dataset

---

## Hardware Requirements

- **GPU:** Recommended (faster training)
- **CPU:** Works but ~4x slower
- **RAM:** 8GB minimum
- **Storage:** ~3GB for dataset + models

