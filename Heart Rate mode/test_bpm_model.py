"""
BPM Classification Model Testing

Test the trained SVM model for heart rate classification.
Uses proper DataFrame format to avoid feature name mismatch warnings.

Usage:
    python test_bpm_model.py
"""

import joblib
import pandas as pd
import os
import sys

print("="*60)
print("BPM CLASSIFICATION MODEL TESTING")
print("="*60)

# Check if model files exist
if not os.path.exists("bpm_classifier.pkl"):
    print("\n❌ Error: bpm_classifier.pkl not found!")
    print("Please run train_bpm_model.py first to train the model.")
    sys.exit(1)

if not os.path.exists("bpm_scaler.pkl"):
    print("\n❌ Error: bpm_scaler.pkl not found!")
    print("Please run train_bpm_model.py first to train the model.")
    sys.exit(1)

# Load trained model and scaler
print("\n📥 Loading model and scaler...")
model = joblib.load("bpm_classifier.pkl")
scaler = joblib.load("bpm_scaler.pkl")
print("✓ Model and scaler loaded successfully!")

def predict_bpm_status(bpm_value):
    """
    Predict BPM status using the trained model.
    
    Args:
        bpm_value (int/float): Heart rate in beats per minute
    
    Returns:
        str: "Normal" or "Emergency"
    """
    # Create DataFrame with proper column name to avoid warnings
    bpm_df = pd.DataFrame([[bpm_value]], columns=["BPM"])
    
    # Scale the feature
    bpm_scaled = scaler.transform(bpm_df)
    
    # Make prediction
    prediction = model.predict(bpm_scaled)[0]
    
    # Return label
    return "Normal" if prediction == 0 else "Emergency"

def get_status_emoji(status):
    """Return emoji based on status"""
    return "✓ " if status == "Normal" else "⚠ "

# Comprehensive test cases
print("\n" + "="*60)
print("COMPREHENSIVE TEST CASES")
print("="*60)

test_cases = [# Extreme edge & pathological values
(0, "Emergency", "Invalid: zero BPM"),
(1, "Emergency", "Invalid: near zero"),
(10, "Emergency", "Severe bradycardia"),
(20, "Emergency", "Extreme bradycardia"),
(30, "Emergency", "Life-threatening low"),
(48, "Emergency", "Near threshold low"),
(51, "Normal", "Just above lower bound"),

# Boundary jitter tests (±1, ±2 around thresholds)
(47, "Emergency", "Boundary jitter low"),
(48, "Emergency", "Boundary jitter low"),
(49, "Emergency", "Boundary jitter low"),
(50, "Normal", "Exact lower threshold"),
(51, "Normal", "Boundary jitter high"),
(52, "Normal", "Boundary jitter high"),

(118, "Normal", "Near upper bound"),
(119, "Normal", "Boundary jitter low high"),
(120, "Normal", "Exact upper threshold"),
(121, "Emergency", "Boundary jitter high"),
(122, "Emergency", "Boundary jitter high"),
(123, "Emergency", "Boundary jitter high"),

# Physiological states
(55, "Normal", "Sleep heart rate"),
(65, "Normal", "Calm resting"),
(75, "Normal", "Walking"),
(90, "Normal", "Brisk walking"),
(110, "Normal", "Running"),
(130, "Emergency", "Post-exercise spike"),
(150, "Emergency", "Sustained tachycardia"),

# Random stress tests
(33, "Emergency", "Random low"),
(44, "Emergency", "Random low"),
(67, "Normal", "Random normal"),
(93, "Normal", "Random normal"),
(108, "Normal", "Random normal"),
(127, "Emergency", "Random high"),
(169, "Emergency", "Random high"),
(190, "Emergency", "Random extreme")

]

print("\nBPM   | Expected   | Predicted  | Status | Description")
print("-" * 75)

correct = 0
total = len(test_cases)

for bpm, expected, description in test_cases:
    predicted = predict_bpm_status(bpm)
    status = "✓" if predicted == expected else "✗ FAIL"
    emoji = get_status_emoji(predicted)
    
    if predicted == expected:
        correct += 1
    
    print(f"{bpm:3d}   | {expected:10s} | {emoji}{predicted:10s} | {status:6s} | {description}")

accuracy = (correct / total) * 100
print("\n" + "="*60)
print(f"Test Accuracy: {correct}/{total} ({accuracy:.1f}%)")

if accuracy == 100:
    print("✓ All test cases passed! Model is working correctly.")
else:
    print("⚠ Some test cases failed. Model may need retraining.")

# Interactive testing mode
print("\n" + "="*60)
print("INTERACTIVE TESTING MODE")
print("="*60)
print("\nEnter BPM values to test the model.")
print("Type -1 or 'quit' to exit.\n")

while True:
    try:
        user_input = input("Enter BPM value: ").strip()
        
        # Check for exit command
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Exiting test mode...")
            break
        
        # Try to convert to number
        user_bpm = float(user_input)
        
        # Check for exit value
        if user_bpm == -1:
            print("\n👋 Exiting test mode...")
            break
        
        # Validate BPM range
        if user_bpm < 0 or user_bpm > 300:
            print("⚠ Warning: BPM value is outside typical range (0-300)")
        
        # Make prediction
        status = predict_bpm_status(user_bpm)
        emoji = get_status_emoji(status)
        
        # Display result
        print(f"\n{emoji}BPM: {user_bpm:.1f} → Status: {status}")
        
        # Provide context
        if user_bpm < 50:
            print("   ℹ Reason: BPM < 50 (Bradycardia)")
        elif user_bpm <= 120:
            print("   ℹ Reason: 50 ≤ BPM ≤ 120 (Normal range)")
        else:
            print("   ℹ Reason: BPM > 120 (Tachycardia)")
        
        print()
        
    except ValueError:
        print("❌ Invalid input. Please enter a numeric BPM value.\n")
    except KeyboardInterrupt:
        print("\n\n👋 Exiting test mode...")
        break
    except Exception as e:
        print(f"❌ Error: {e}\n")

print("\n" + "="*60)
print("Thank you for using the BPM Classifier!")
print("="*60)
