# BPM Heart Rate Classification System

A machine learning solution for classifying heart rate (BPM) values into **Normal** and **Emergency** categories using Support Vector Machine (SVM).

## 📋 Overview

This system addresses the critical issue where standard Logistic Regression models incorrectly classify low BPM values (< 50) as Normal. The solution uses an SVM classifier with a hard margin to enforce strict decision boundaries.

### Classification Rules

- **BPM < 50** → Emergency (Bradycardia)
- **50 ≤ BPM ≤ 120** → Normal
- **BPM > 120** → Emergency (Tachycardia)

## 🎯 Key Features

✓ **SVM with Hard Margin** - Uses C=1000 for strict classification boundaries  
✓ **Balanced Dataset** - Equal samples across all BPM ranges  
✓ **Feature Normalization** - StandardScaler for optimal SVM performance  
✓ **Edge Case Handling** - Correctly classifies BPM < 50 and BPM > 120  
✓ **DataFrame Format** - Prevents feature name mismatch warnings  
✓ **100% Accuracy** - Perfect classification on test set

## 📁 Project Structure

```
Minor model _Last Try/
├── generate_bpm_dataset.py    # Dataset generator (balanced samples)
├── train_bpm_model.py          # SVM model training script
├── test_bpm_model.py           # Model testing and inference
├── Dataset.csv                 # Generated dataset (450 samples)
├── bpm_classifier.pkl          # Trained SVM model
├── bpm_scaler.pkl              # StandardScaler for normalization
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Generate Dataset

```bash
python generate_bpm_dataset.py
```

Creates a balanced dataset with:
- 150 Emergency samples (BPM < 50)
- 150 Normal samples (50-120)
- 150 Emergency samples (BPM > 120)

### 2. Train Model

```bash
python train_bpm_model.py
```

Trains SVM classifier and displays:
- Training/testing split information
- Model accuracy (100%)
- Classification report
- Confusion matrix
- Edge case validation results

### 3. Test Model

```bash
python test_bpm_model.py
```

Runs comprehensive tests and provides interactive mode for custom BPM values.

### 4. Generate Graphs

```bash
python plot_bpm_model_graphs.py
```

Creates visual outputs in the `graphs/` folder:
- `01_bpm_distribution.png`
- `02_prediction_regions.png`
- `03_decision_function.png`
- `04_confusion_matrix.png`

## 🔬 Technical Details

### Model Architecture

- **Algorithm**: Support Vector Machine (SVC)
- **Kernel**: Radial Basis Function (RBF)
- **Penalty**: C = 1000 (hard margin)
- **Preprocessing**: StandardScaler normalization
- **Framework**: scikit-learn

### Why SVM Over Logistic Regression?

**Problem with Logistic Regression:**
- Uses a linear decision boundary
- Assumes sigmoid probability distribution
- Can misclassify edge cases (especially BPM < 50)

**SVM Advantages:**
- Non-linear decision boundaries with RBF kernel
- Hard margin (high C value) enforces strict separation
- Better handling of multi-region classification
- Superior performance on edge cases

### Performance Metrics

```
Accuracy: 100%
Precision: 100% (both classes)
Recall: 100% (both classes)
F1-Score: 100% (both classes)
```

## 📊 Example Usage

### Python Code

```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("bpm_classifier.pkl")
scaler = joblib.load("bpm_scaler.pkl")

# Predict single value
bpm_value = 45
bpm_df = pd.DataFrame([[bpm_value]], columns=["BPM"])
bpm_scaled = scaler.transform(bpm_df)
prediction = model.predict(bpm_scaled)[0]

status = "Normal" if prediction == 0 else "Emergency"
print(f"BPM {bpm_value}: {status}")  # Output: BPM 45: Emergency
```

### Batch Prediction

```python
# Predict multiple values
bpm_values = [45, 80, 150]
bpm_df = pd.DataFrame(bpm_values, columns=["BPM"])
bpm_scaled = scaler.transform(bpm_df)
predictions = model.predict(bpm_scaled)

for bpm, pred in zip(bpm_values, predictions):
    status = "Normal" if pred == 0 else "Emergency"
    print(f"BPM {bpm}: {status}")
```

## ✅ Validation Results

All edge cases correctly classified:

| BPM | Expected | Predicted | Status |
|-----|----------|-----------|--------|
| 25  | Emergency | Emergency | ✓ |
| 45  | Emergency | Emergency | ✓ |
| 49  | Emergency | Emergency | ✓ |
| 50  | Normal | Normal | ✓ |
| 85  | Normal | Normal | ✓ |
| 120 | Normal | Normal | ✓ |
| 121 | Emergency | Emergency | ✓ |
| 150 | Emergency | Emergency | ✓ |
| 200 | Emergency | Emergency | ✓ |

## 🔧 Requirements

```
pandas
numpy
scikit-learn
joblib
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn joblib
```

## 📝 Notes

- **Dataset Balance**: Critical for unbiased learning
- **Feature Scaling**: Essential for SVM performance
- **DataFrame Format**: Prevents sklearn warnings during inference
- **High C Value**: Enforces hard margin for strict classification
- **Academic Use**: Code is well-commented for educational purposes

## 🏥 Healthcare Application

This system can be integrated into:
- Patient monitoring systems
- Wearable health devices
- Emergency alert systems
- Telemedicine platforms

**Warning**: This is a demonstration system. For actual medical use, consult healthcare professionals and follow medical device regulations.

## 📄 License

Educational and research use. For commercial applications, ensure compliance with healthcare regulations.

## 👤 Author

Created for academic and healthcare monitoring projects.

---

**Last Updated**: February 2026
