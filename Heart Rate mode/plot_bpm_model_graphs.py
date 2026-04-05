"""
Generate graphs for the trained BPM classification model.

Creates the following plots in the graphs/ folder:
1. Dataset distribution by class
2. Model prediction regions across BPM values
3. Decision function curve
4. Confusion matrix on a reproducible test split

Usage:
    python plot_bpm_model_graphs.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split


DATASET_FILE = "Dataset.csv"
MODEL_FILE = "bpm_classifier.pkl"
SCALER_FILE = "bpm_scaler.pkl"
OUTPUT_DIR = "graphs"


def ensure_files_exist() -> None:
    required_files = [DATASET_FILE, MODEL_FILE, SCALER_FILE]
    missing = [file for file in required_files if not os.path.exists(file)]
    if missing:
        print("Missing required files:")
        for file in missing:
            print(f"- {file}")
        print("Run dataset generation and training scripts first.")
        raise SystemExit(1)


def prepare_test_split(data: pd.DataFrame):
    x = data[["BPM"]]
    y = data["Status"].map({"Normal": 0, "Emergency": 1})

    return train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def save_distribution_plot(data: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))

    normal = data[data["Status"] == "Normal"]["BPM"]
    emergency = data[data["Status"] == "Emergency"]["BPM"]

    bins = np.arange(0, 201, 5)
    plt.hist(normal, bins=bins, alpha=0.7, label="Normal", color="#2e8b57")
    plt.hist(emergency, bins=bins, alpha=0.7, label="Emergency", color="#b22222")

    plt.axvline(50, color="black", linestyle="--", linewidth=1, label="Lower threshold")
    plt.axvline(120, color="black", linestyle="--", linewidth=1, label="Upper threshold")

    plt.title("BPM Distribution by Class")
    plt.xlabel("BPM")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_bpm_distribution.png"), dpi=160)
    plt.close()


def save_prediction_regions_plot(model, scaler) -> None:
    bpm_range = np.arange(0, 201).reshape(-1, 1)
    bpm_df = pd.DataFrame(bpm_range, columns=["BPM"])
    bpm_scaled = scaler.transform(bpm_df)
    preds = model.predict(bpm_scaled)

    plt.figure(figsize=(10, 4.5))
    plt.plot(bpm_range, preds, color="#1f77b4", linewidth=2)
    plt.yticks([0, 1], ["Normal (0)", "Emergency (1)"])
    plt.ylim(-0.2, 1.2)
    plt.xlim(0, 200)
    plt.axvline(50, color="gray", linestyle="--", linewidth=1)
    plt.axvline(120, color="gray", linestyle="--", linewidth=1)

    plt.title("Predicted Class Across BPM Values")
    plt.xlabel("BPM")
    plt.ylabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_prediction_regions.png"), dpi=160)
    plt.close()


def save_decision_function_plot(model, scaler) -> None:
    if not hasattr(model, "decision_function"):
        return

    bpm_range = np.arange(0, 201).reshape(-1, 1)
    bpm_df = pd.DataFrame(bpm_range, columns=["BPM"])
    bpm_scaled = scaler.transform(bpm_df)
    decision_values = model.decision_function(bpm_scaled)

    plt.figure(figsize=(10, 4.5))
    plt.plot(bpm_range, decision_values, color="#8a2be2", linewidth=2)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.axvline(50, color="gray", linestyle="--", linewidth=1)
    plt.axvline(120, color="gray", linestyle="--", linewidth=1)

    plt.title("SVM Decision Function vs BPM")
    plt.xlabel("BPM")
    plt.ylabel("Decision Function")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_decision_function.png"), dpi=160)
    plt.close()


def save_confusion_matrix_plot(model, scaler, x_test, y_test) -> None:
    x_test_scaled = scaler.transform(x_test)
    y_pred = model.predict(x_test_scaled)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Emergency"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (Test Split)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_confusion_matrix.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_files_exist()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = pd.read_csv(DATASET_FILE)
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    _, x_test, _, y_test = prepare_test_split(data)

    save_distribution_plot(data)
    save_prediction_regions_plot(model, scaler)
    save_decision_function_plot(model, scaler)
    save_confusion_matrix_plot(model, scaler, x_test, y_test)

    print("Graphs generated successfully in the 'graphs' folder:")
    print("- 01_bpm_distribution.png")
    print("- 02_prediction_regions.png")
    print("- 03_decision_function.png")
    print("- 04_confusion_matrix.png")


if __name__ == "__main__":
    main()
