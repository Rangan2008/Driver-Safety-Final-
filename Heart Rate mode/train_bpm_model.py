"""
BPM Classification Model Training

Train a Support Vector Machine (SVM) with hard margin to classify
heart rate values into Normal and Emergency categories.

Key Features:
- Uses SVM with high C value (hard margin) for strict decision boundaries
- StandardScaler normalization for better SVM performance
- Properly handles edge cases (BPM < 50 and BPM > 120)
- Balanced dataset ensures unbiased learning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

print("="*60)
print("BPM CLASSIFICATION MODEL TRAINING")
print("Using Support Vector Machine (SVM)")
print("="*60)

# Check if dataset exists
if not os.path.exists("Dataset.csv"):
    print("\n❌ Error: Dataset.csv not found!")
    print("Please run generate_bpm_dataset.py first.")
    exit(1)

# Load dataset
print("\n📊 Loading dataset...")
data = pd.read_csv("Dataset.csv")
print(f"✓ Loaded {len(data)} samples")

# Display dataset statistics
print("\n📈 Dataset Statistics:")
print(data.groupby('Status').size())
print(f"\nBPM Range: {data['BPM'].min()} - {data['BPM'].max()}")

# Features and labels
X = data[["BPM"]]          # Input feature (as DataFrame)
y = data["Status"]         # Target label

# Convert labels to numeric: Normal -> 0, Emergency -> 1
y = y.map({"Normal": 0, "Emergency": 1})

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📂 Data Split:")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Feature scaling using StandardScaler
print("\n🔧 Normalizing features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model with high C value (hard margin)
# High C value means less tolerance for misclassification
# This enforces strict decision boundaries
print("\n🤖 Training SVM model...")
print("   Settings: C=1000 (hard margin), kernel='rbf'")
model = SVC(
    C=1000,              # High penalty for misclassification (hard margin)
    kernel='rbf',        # Radial Basis Function kernel
    gamma='scale',       # Auto-adjust gamma
    random_state=42
)

model.fit(X_train_scaled, y_train)
print("✓ Model training complete!")

# Make predictions
print("\n🔍 Evaluating model performance...")
y_pred = model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)
print(f"\n✓ Accuracy: {accuracy*100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal", "Emergency"],
    digits=4
))

print("\n📉 Confusion Matrix:")
print("                Predicted")
print("              Normal  Emergency")
print(f"Actual Normal     {conf_matrix[0][0]:4d}     {conf_matrix[0][1]:4d}")
print(f"       Emergency  {conf_matrix[1][0]:4d}     {conf_matrix[1][1]:4d}")

# Test critical edge cases
print("\n" + "="*60)
print("EDGE CASE VALIDATION")
print("="*60)

edge_cases = [
    (25, "Emergency"),
    (45, "Emergency"),
    (49, "Emergency"),
    (50, "Normal"),
    (85, "Normal"),
    (120, "Normal"),
    (121, "Emergency"),
    (150, "Emergency"),
    (200, "Emergency")
]

print("\nBPM   | Expected   | Predicted  | Status")
print("-" * 50)

all_correct = True
for bpm, expected in edge_cases:
    # Create DataFrame with proper column name
    bpm_df = pd.DataFrame([[bpm]], columns=["BPM"])
    bpm_scaled = scaler.transform(bpm_df)
    prediction = model.predict(bpm_scaled)[0]
    predicted_label = "Normal" if prediction == 0 else "Emergency"
    
    status = "✓" if predicted_label == expected else "✗ FAIL"
    if predicted_label != expected:
        all_correct = False
    
    print(f"{bpm:3d}   | {expected:10s} | {predicted_label:10s} | {status}")

if all_correct:
    print("\n✓ All edge cases passed!")
else:
    print("\n⚠ Warning: Some edge cases failed. Model may need adjustment.")

# Save model and scaler
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

joblib.dump(model, "bpm_classifier.pkl")
joblib.dump(scaler, "bpm_scaler.pkl")

print("\n✓ Model saved as: bpm_classifier.pkl")
print("✓ Scaler saved as: bpm_scaler.pkl")
print("\n💡 Use test_bpm_model.py to test the model with new data.")
print("="*60)
