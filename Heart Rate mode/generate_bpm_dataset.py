"""
BPM Dataset Generator

Generates a balanced dataset for BPM classification with sufficient samples
in all critical ranges to ensure proper model training.

Labeling Rules:
- BPM < 50 → Emergency
- 50 ≤ BPM ≤ 120 → Normal
- BPM > 120 → Emergency
"""

import csv
import random

# Output file name
filename = "Dataset.csv"

# Number of samples per category (balanced dataset)
samples_per_category = 150

# Open CSV file
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    
    # Write header
    writer.writerow(["BPM", "Status"])
    
    # Generate Emergency samples: BPM < 50
    for _ in range(samples_per_category):
        bpm = random.randint(25, 49)  # Below 50
        writer.writerow([bpm, "Emergency"])
    
    # Generate Normal samples: 50 ≤ BPM ≤ 120
    for _ in range(samples_per_category):
        bpm = random.randint(50, 120)  # Normal range
        writer.writerow([bpm, "Normal"])
    
    # Generate Emergency samples: BPM > 120
    for _ in range(samples_per_category):
        bpm = random.randint(121, 200)  # Above 120
        writer.writerow([bpm, "Emergency"])

total_samples = samples_per_category * 3
print(f"✓ Dataset.csv created successfully!")
print(f"✓ Total samples: {total_samples}")
print(f"✓ Emergency (BPM < 50): {samples_per_category}")
print(f"✓ Normal (50-120): {samples_per_category}")
print(f"✓ Emergency (BPM > 120): {samples_per_category}")
print(f"✓ Dataset is balanced for optimal SVM training.")
