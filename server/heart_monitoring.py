import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import streamlit as st

# Path configuration
SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent
MODELS_DIR = SERVER_DIR / "models"
HEART_MODE_DIR = PROJECT_ROOT / "Heart Rate mode"

BPM_CLASSIFIER_CANDIDATES = [
    MODELS_DIR / "bpm_classifier.pkl",
    HEART_MODE_DIR / "bpm_classifier.pkl",
]
BPM_SCALER_CANDIDATES = [
    MODELS_DIR / "bpm_scaler.pkl",
    HEART_MODE_DIR / "bpm_scaler.pkl",
]


def _first_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None

@st.cache_resource
def load_heart_model():
    """Load BPM classifier + scaler used by test_bpm_model.py logic."""
    classifier_path = _first_existing(BPM_CLASSIFIER_CANDIDATES)
    scaler_path = _first_existing(BPM_SCALER_CANDIDATES)

    if not classifier_path or not scaler_path:
        return None

    return {
        "classifier": joblib.load(str(classifier_path)),
        "scaler": joblib.load(str(scaler_path)),
    }

def predict_heart_condition(model, rr_sequence, confidence_threshold=0.8):
    """
    Predict heart status using the BPM SVM logic from test_bpm_model.py.

    Mapping:
      classifier output 0 -> NORMAL
      classifier output 1 -> EMERGENCY
    """
    rr_sequence = np.asarray(rr_sequence, dtype=np.float32)
    bpm = calculate_bpm(rr_sequence)

    if model is None:
        # Threshold fallback aligned with train/test scripts: 50..120 is Normal.
        fallback_class = "NORMAL" if 50 <= bpm <= 120 else "EMERGENCY"
        if fallback_class == "NORMAL":
            fallback_probs = {"Normal": 0.9, "Warning": 0.0, "Emergency": 0.1}
        else:
            fallback_probs = {"Normal": 0.1, "Warning": 0.0, "Emergency": 0.9}
        return {
            "class": fallback_class,
            "probabilities": fallback_probs,
            "trigger_sos": bool(fallback_class == "EMERGENCY"),
            "confidence": 0.0,
        }

    classifier = model["classifier"]
    scaler = model["scaler"]

    # Use DataFrame with "BPM" column to match test_bpm_model.py exactly.
    bpm_df = pd.DataFrame([[bpm]], columns=["BPM"])
    bpm_scaled = scaler.transform(bpm_df)

    pred = int(classifier.predict(bpm_scaled)[0])
    model_class = "NORMAL" if pred == 0 else "EMERGENCY"

    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(bpm_scaled)[0]
        # Binary model: index 0=Normal, index 1=Emergency
        normal_p = float(proba[0])
        emergency_p = float(proba[1])
        confidence = float(max(normal_p, emergency_p))
    else:
        normal_p = 1.0 if model_class == "NORMAL" else 0.0
        emergency_p = 1.0 - normal_p
        confidence = 1.0

    prob_dict = {
        "Normal": normal_p,
        # Keep Warning key to preserve UI/engine compatibility.
        "Warning": 0.0,
        "Emergency": emergency_p,
    }

    return {
        "class": model_class,
        "probabilities": prob_dict,
        "trigger_sos": bool(model_class == "EMERGENCY" and confidence >= confidence_threshold),
        "confidence": confidence,
    }

def generate_demo_heart_data(window_size=20):
    """Generate mock RR intervals for demo purposes."""
    # Regime probabilities tuned so demo is mostly normal.
    regime_roll = np.random.rand()

    if regime_roll < 0.96:
        # NORMAL: ~67-82 BPM, very low variability to avoid false warnings.
        data = np.random.normal(loc=0.82, scale=0.02, size=window_size)
    elif regime_roll < 0.995:
        # WARNING: mild tachy/brady or modest irregularity
        base = np.random.choice([0.56, 1.08])  # around ~107 BPM or ~56 BPM
        data = np.random.normal(loc=base, scale=0.07, size=window_size)
    else:
        # EMERGENCY: highly irregular rhythm, including outliers
        data = np.random.normal(loc=0.95, scale=0.22, size=window_size)
        spike_count = max(1, window_size // 8)
        spike_idx = np.random.choice(window_size, size=spike_count, replace=False)
        data[spike_idx] += np.random.choice([-0.35, 0.35], size=spike_count)

    # Physiological clamp: 0.35s to 1.5s RR interval (~40 to 171 BPM)
    return np.clip(data, 0.35, 1.5)

def calculate_bpm(rr_intervals):
    """Calculate average BPM from RR intervals (in seconds)."""
    if len(rr_intervals) == 0:
        return 0
    mean_rr = float(np.mean(rr_intervals))
    if mean_rr <= 0:
        return 0
    bpm = 60.0 / mean_rr
    # Clamp to realistic display range for demo UX.
    return float(np.clip(bpm, 40, 160))

def generate_ecg_point(heart_rate_bpm, time_step):
    """
    Generate a single synthetic ECG signal point using a simplified P-QRS-T model.
    heart_rate_bpm: Current BPM
    time_step: Current time index in the heart cycle (0.0 to 1.0)
    """
    # Define timing of ECG components (fractions of heart cycle)
    p_peak = 0.15; q_peak = 0.25; r_peak = 0.35; s_peak = 0.45; t_peak = 0.65
    
    # Amplitudes
    p_amp = 0.15; q_amp = -0.25; r_amp = 1.0; s_amp = -0.3; t_amp = 0.35
    
    # Pulse widths (Gaussian-like)
    p_width = 0.05; q_width = 0.02; r_width = 0.02; s_width = 0.02; t_width = 0.1
    
    # Summing the components
    p_wave = p_amp * np.exp(-((time_step - p_peak)**2) / (2 * p_width**2))
    q_wave = q_amp * np.exp(-((time_step - q_peak)**2) / (2 * q_width**2))
    r_wave = r_amp * np.exp(-((time_step - r_peak)**2) / (2 * r_width**2))
    s_wave = s_amp * np.exp(-((time_step - s_peak)**2) / (2 * s_width**2))
    t_wave = t_amp * np.exp(-((time_step - t_peak)**2) / (2 * t_width**2))
    
    # Add some noise for realism
    noise = np.random.normal(0, 0.02)
    
    return p_wave + q_wave + r_wave + s_wave + t_wave + noise
