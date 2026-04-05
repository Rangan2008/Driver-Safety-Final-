import re
import threading
import time
from collections import deque
from typing import Any, Dict

import cv2
import mediapipe as mp

from detection import DrowsinessDetector
from distraction_model import load_distraction_models, predict_driver_behavior
from heart_monitoring import (
    calculate_bpm,
    generate_demo_heart_data,
    generate_ecg_point,
    load_heart_model,
    predict_heart_condition,
)


class SafetyEngine:
    """Unified orchestrator that runs vision + heart modules in one pipeline."""

    def __init__(self, frame_skip: int = 2, smoothing_window: int = 12) -> None:
        self.frame_skip = max(1, int(frame_skip))
        self.frame_counter = 0

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
        )
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,  # Upgraded from 0 to 1 for better accuracy
            min_detection_confidence=0.6,  # Stricter detection threshold
            min_tracking_confidence=0.6,  # Stricter tracking threshold
        )
        self.drowsiness_detector = DrowsinessDetector()
        self.distraction_models = load_distraction_models()
        self.heart_model = load_heart_model()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        self.drowsiness_history = deque([False] * smoothing_window, maxlen=smoothing_window)
        self.distraction_history = deque([False] * smoothing_window, maxlen=smoothing_window)
        self.hands_history = deque([False] * smoothing_window, maxlen=smoothing_window)
        self.heart_history = deque([False] * smoothing_window, maxlen=smoothing_window)

        self.heart_lock = threading.Lock()
        self._stop_event = threading.Event()
        self.latest_heart: Dict[str, Any] = {
            "class": "NORMAL",
            "probabilities": {"Normal": 1.0, "Warning": 0.0, "Emergency": 0.0},
            "trigger_sos": False,
            "confidence": 1.0,
            "rr": [0.8] * 20,
            "bpm": 75.0,
        }

        self.ecg_buffer = deque([0.0] * 140, maxlen=140)
        self.ecg_step = 0.0

        # Hands-on-wheel tuning: require at least one tracked hand in wheel zone.
        self.required_hands_on_wheel = 1
        self.no_hands_grace_seconds = 1.2
        self.last_hands_seen_ts = 0.0
        self.last_hands_on_wheel_count = 0
        self.has_ever_detected_hands = False  # Flag to distinguish "first frame" from "hands off wheel"
        
        # Hand detection tuning parameters
        self.hand_landmark_threshold = 0.25  # Relaxed from 0.30 for better detection (25% of landmarks in zone)
        self.hand_confidence_threshold = 0.6  # Stricter threshold to reduce false hand detections

        self.last_visual_state: Dict[str, Any] = {
            "frame": None,
            "drowsiness_raw": False,
            "drowsiness_confidence": 0.0,
            "drowsiness_metrics": {
                "ear": 0.0,
                "mar": 0.0,
                "nod_ratio": 0.0,
                "yawn_count": 0,
                "face_detected": False,
            },
            "distraction_raw": False,
            "hands_off_wheel_raw": False,
            "hands_confidence": 0.0,
            "hands_metrics": {
                "hands_detected": 0,
                "hands_on_wheel": 0,
                "zone": [0, 0, 0, 0],
            },
            "distraction_confidence": 0.0,
            "distraction_label": "No Model",
            "distraction_metrics_text": "Waiting for first frame...",
        }

        self._heart_thread = threading.Thread(target=self._heart_loop, daemon=True)
        self._heart_thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._heart_thread.is_alive():
            self._heart_thread.join(timeout=1.0)

    def _heart_loop(self) -> None:
        """Run heart model asynchronously so vision stays responsive."""
        while not self._stop_event.is_set():
            rr_data = generate_demo_heart_data(window_size=20)
            prediction = predict_heart_condition(self.heart_model, rr_data)
            bpm = calculate_bpm(rr_data)

            with self.heart_lock:
                self.latest_heart = {
                    "class": prediction.get("class", "NORMAL"),
                    "probabilities": prediction.get(
                        "probabilities",
                        {"Normal": 1.0, "Warning": 0.0, "Emergency": 0.0},
                    ),
                    "trigger_sos": bool(prediction.get("trigger_sos", False)),
                    "confidence": float(prediction.get("confidence", 0.0)),
                    "rr": rr_data.tolist(),
                    "bpm": float(bpm),
                }

            time.sleep(0.8)

    def _parse_distraction_confidence(self, metrics_output: str) -> float:
        if not metrics_output:
            return 0.0
        match = re.search(r"Confidence Score:\s*([0-9.]+)%", metrics_output)
        if not match:
            return 0.0
        return max(0.0, min(1.0, float(match.group(1)) / 100.0))

    def _smooth_bool(self, history: deque, value: bool, threshold_ratio: float) -> bool:
        history.append(bool(value))
        return (sum(history) / len(history)) >= threshold_ratio

    def _compute_priority(
        self,
        sos_triggered: bool,
        risk_count: int,
        heart_class: str,
        weak_signals_present: bool,
    ) -> Dict[str, str]:
        if sos_triggered and (heart_class == "EMERGENCY" or risk_count >= 2):
            return {"priority": "EMERGENCY", "overall_status": "DANGER"}
        if sos_triggered:
            return {"priority": "CRITICAL", "overall_status": "DANGER"}
        if weak_signals_present:
            return {"priority": "WARNING", "overall_status": "WARNING"}
        return {"priority": "WARNING", "overall_status": "SAFE"}

    def _run_visual_pipeline(self, frame_bgr):
        processed_frame = frame_bgr

        drowsiness_raw = False
        drowsiness_confidence = 0.0
        drowsiness_metrics = {
            "ear": 0.0,
            "mar": 0.0,
            "nod_ratio": 0.0,
            "yawn_count": 0,
            "face_detected": False,
        }

        h, w, _ = frame_bgr.shape
        # Approximate steering wheel region in driver-facing camera coordinates.
        zone_x1, zone_x2 = int(0.22 * w), int(0.82 * w)
        zone_y1, zone_y2 = int(0.50 * h), int(0.95 * h)
        hands_detected = 0
        hands_on_wheel = 0
        hands_confidence = 0.0
        hands_off_wheel_raw = False

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb)
        if face_results.multi_face_landmarks:
            face_lm = face_results.multi_face_landmarks[0]
            result = self.drowsiness_detector.detect(face_lm, frame_bgr.shape)
            drowsiness_raw = bool(
                result.get("drowsiness_alert")
                or result.get("yawn_alert")
                or result.get("nod_alert")
            )
            ear_risk = max(0.0, min(1.0, (0.22 - result.get("ear", 0.22)) / 0.14))
            mar_risk = max(0.0, min(1.0, (result.get("mar", 0.0) - 0.45) / 0.35))
            nod_risk = max(0.0, min(1.0, (result.get("nod_ratio", 0.0) - 0.35) / 0.35))
            drowsiness_confidence = max(ear_risk, mar_risk, nod_risk)
            drowsiness_metrics = {
                "ear": float(result.get("ear", 0.0)),
                "mar": float(result.get("mar", 0.0)),
                "nod_ratio": float(result.get("nod_ratio", 0.0)),
                "yawn_count": int(result.get("yawn_count", 0)),
                "face_detected": True,
            }

        hand_results = self.hands.process(rgb)
        if hand_results.multi_hand_landmarks and hand_results.multi_hand_world_landmarks:
            hands_detected = len(hand_results.multi_hand_landmarks)
            self.last_hands_seen_ts = time.time()
            self.has_ever_detected_hands = True  # Mark that we've detected hands at least once

            for hand_idx, (hand_lm, hand_conf) in enumerate(zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)):
                # Filter out low-confidence detections
                hand_score = hand_conf.classification[0].score if hand_conf.classification else 0.0
                if hand_score < self.hand_confidence_threshold:
                    continue  # Skip this hand detection if confidence is too low
                
                # Use multiple keypoints instead of only palm center for robust in-zone scoring.
                key_indices = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
                points = []
                in_zone = 0
                for i in key_indices:
                    px = int(hand_lm.landmark[i].x * w)
                    py = int(hand_lm.landmark[i].y * h)
                    points.append((px, py))
                    if zone_x1 <= px <= zone_x2 and zone_y1 <= py <= zone_y2:
                        in_zone += 1

                cx = int(sum(p[0] for p in points) / len(points))
                cy = int(sum(p[1] for p in points) / len(points))

                # Consider hand on wheel if enough landmarks fall inside the wheel zone (relaxed to 25% from 30%).
                landmark_ratio = in_zone / len(points)
                on_wheel = landmark_ratio >= self.hand_landmark_threshold
                if on_wheel:
                    hands_on_wheel += 1

                color = (0, 200, 0) if on_wheel else (0, 0, 255)
                cv2.circle(processed_frame, (cx, cy), 8, color, -1)

        # Interpretation-only inversion requested by deployment policy:
        # - Any detected hand in frame => HANDS OFF WHEEL (danger)
        # - No hands detected => HANDS ON WHEEL (safe)
        effective_hands_on_wheel = 0 if hands_detected > 0 else self.required_hands_on_wheel
        hands_off_wheel_raw = hands_detected > 0
        hands_confidence = 0.95 if hands_off_wheel_raw else 0.0

        # Low-frequency debug logging to avoid hurting real-time throughput.
        if self.frame_counter % 20 == 0:
            print(f"[HAND STATUS] Hands detected: {bool(hands_detected)} (count={hands_detected})")
            print(f"[HAND STATUS] Hands on wheel: {not hands_off_wheel_raw}")

        zone_color = (0, 150, 0) if not hands_off_wheel_raw else (0, 0, 200)
        cv2.rectangle(processed_frame, (zone_x1, zone_y1), (zone_x2, zone_y2), zone_color, 2)
        cv2.putText(
            processed_frame,
            "HANDS OFF WHEEL ⚠️" if hands_off_wheel_raw else "HANDS ON WHEEL ✅",
            (zone_x1, max(24, zone_y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            zone_color,
            2,
            cv2.LINE_AA,
        )

        processed_frame, distraction_metrics_text, distraction_label = predict_driver_behavior(
            processed_frame,
            self.distraction_models,
            self.face_cascade,
            self.eye_cascade,
        )
        distraction_confidence = self._parse_distraction_confidence(distraction_metrics_text)

        distraction_raw = "DISTRACTED" in distraction_label

        self.last_visual_state = {
            "frame": processed_frame,
            "drowsiness_raw": drowsiness_raw,
            "drowsiness_confidence": drowsiness_confidence,
            "drowsiness_metrics": drowsiness_metrics,
            "distraction_raw": distraction_raw,
            "hands_off_wheel_raw": hands_off_wheel_raw,
            "hands_confidence": hands_confidence,
            "hands_metrics": {
                "hands_detected": int(hands_detected),
                "hands_on_wheel": int(effective_hands_on_wheel),
                "required_hands_on_wheel": int(self.required_hands_on_wheel),
                "hands_off_wheel_raw": bool(hands_off_wheel_raw),
                "hand_status_text": "HANDS OFF WHEEL ⚠️" if hands_off_wheel_raw else "HANDS ON WHEEL ✅",
                "zone": [zone_x1, zone_y1, zone_x2, zone_y2],
                "has_ever_detected_hands": bool(self.has_ever_detected_hands),
                "grace_period_active": bool(self.has_ever_detected_hands and (time.time() - self.last_hands_seen_ts) <= self.no_hands_grace_seconds),
            },
            "distraction_confidence": distraction_confidence,
            "distraction_label": distraction_label,
            "distraction_metrics_text": distraction_metrics_text,
        }

    def process_step(self, frame_bgr):
        """Process one unified cycle: visual analysis + heart analysis + fusion."""
        self.frame_counter += 1

        if self.frame_counter % self.frame_skip == 0:
            self._run_visual_pipeline(frame_bgr)

        visual = self.last_visual_state
        with self.heart_lock:
            heart_state = dict(self.latest_heart)

        bpm = float(heart_state.get("bpm", 75.0))
        self.ecg_step = (self.ecg_step + (bpm / 1200.0)) % 1.0
        self.ecg_buffer.append(generate_ecg_point(bpm, self.ecg_step))

        heart_class = heart_state.get("class", "NORMAL")
        heart_raw = bool(heart_state.get("trigger_sos", False) or heart_class in ("WARNING", "EMERGENCY"))
        heart_confidence = float(heart_state.get("confidence", 0.0))

        drowsiness = self._smooth_bool(self.drowsiness_history, visual["drowsiness_raw"], 0.35)
        distraction = self._smooth_bool(self.distraction_history, visual["distraction_raw"], 0.35)
        hands_off_wheel = self._smooth_bool(self.hands_history, visual["hands_off_wheel_raw"], 0.45)
        heart_risk_active = self._smooth_bool(self.heart_history, heart_raw, 0.30)

        risk_flags = {
            "hands_off_wheel": hands_off_wheel,
            "drowsiness": drowsiness,
            "distraction": distraction,
            "heart_risk": heart_risk_active,
        }
        risk_count = sum(1 for v in risk_flags.values() if v)

        sos_triggered = any(risk_flags.values())

        weak_signals_present = any(
            [
                visual["drowsiness_raw"],
                visual["distraction_raw"],
                visual["hands_off_wheel_raw"],
                heart_raw,
            ]
        )
        levels = self._compute_priority(
            sos_triggered=sos_triggered,
            risk_count=risk_count,
            heart_class=heart_class,
            weak_signals_present=weak_signals_present,
        )

        overall_confidence = max(
            float(visual["drowsiness_confidence"]),
            float(visual["distraction_confidence"]),
            float(visual.get("hands_confidence", 0.0)),
            float(heart_confidence),
        )

        # Alerts should show RAW detection (what camera sees NOW), not smoothed values (which lag)
        alerts = []
        if visual["hands_off_wheel_raw"]:
            alerts.append("HANDS OFF WHEEL ⚠️")
        if visual["drowsiness_raw"]:
            alerts.append("Drowsiness detected")
        if visual["distraction_raw"]:
            alerts.append("Driver distraction detected")
        if heart_raw:
            alerts.append(f"Heart anomaly risk: {heart_class}")

        return {
            "timestamp": time.time(),
            "frame": visual["frame"] if visual["frame"] is not None else frame_bgr,
            "drowsiness": drowsiness,
            "distraction": distraction,
            "hands_off_wheel": hands_off_wheel,
            "hand_status_text": "HANDS OFF WHEEL ⚠️" if visual["hands_off_wheel_raw"] else "HANDS ON WHEEL ✅",
            "heart_risk": heart_class if heart_risk_active else "NORMAL",
            "overall_status": levels["overall_status"],
            "priority_level": levels["priority"],
            "sos_triggered": sos_triggered,
            "confidence": {
                "drowsiness": round(float(visual["drowsiness_confidence"]), 3),
                "distraction": round(float(visual["distraction_confidence"]), 3),
                "hands_off_wheel": round(float(visual.get("hands_confidence", 0.0)), 3),
                "heart": round(float(heart_confidence), 3),
                "overall": round(float(overall_confidence), 3),
            },
            "metrics": {
                "drowsiness": visual["drowsiness_metrics"],
                "hands": visual.get("hands_metrics", {}),
                "distraction": {
                    "label": visual["distraction_label"],
                    "details": visual["distraction_metrics_text"],
                },
                "heart": {
                    "class": heart_class,
                    "bpm": round(float(bpm), 1),
                    "probabilities": heart_state.get("probabilities", {}),
                },
            },
            "heart_chart": list(self.ecg_buffer),
            "alerts": alerts,
        }
