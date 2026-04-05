from scipy.spatial import distance as dist
import time

# ============ CONFIG ============

EYE_AR_THRESH = 0.15
EYE_AR_CONSEC_FRAMES = 15
MAR_AR_THRESH = 0.5
YAWN_CONSEC_FRAMES = 20

NOD_THRESH = 0.45
NOD_FRAMES = 20

# Eye and mouth landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [
    78, 95, 88, 178, 87, 14, 317, 402,
    318, 324, 308, 415, 310, 311, 312,
    13, 82, 81, 80, 191
]
NOSE = 1
CHIN = 152

# ============ DETECTION FUNCTIONS ============

def euclidean(p1, p2):
    return dist.euclidean(p1, p2)


def eye_aspect_ratio(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    # Use a simple height/width ratio of the mouth region for robust MAR in MediaPipe.
    # This avoids relying on specific landmark indexing that differs from dlib's 68-point model.
    xs = [p[0] for p in mouth]
    ys = [p[1] for p in mouth]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    # Avoid division by zero
    if width <= 0:
        return 0.0
    return height / width


class DrowsinessDetector:
    
    def __init__(self):
        self.eye_counter = 0
        self.yawn_counter = 0
        self.nod_counter = 0
        self.yawn_history = []
    
    def detect(self, face_landmarks, frame_shape):
        h, w, _ = frame_shape
        now = time.time()
        
        # Extract coordinates from landmarks
        coords = []
        for lm in face_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            coords.append((x, y))
        
        # Calculate metrics
        left_eye = [coords[i] for i in LEFT_EYE]
        right_eye = [coords[i] for i in RIGHT_EYE]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
        
        mouth = [coords[i] for i in MOUTH]
        mar = mouth_aspect_ratio(mouth)
        
        nose_y = coords[NOSE][1]
        chin_y = coords[CHIN][1]

        # Use face width (based on detected landmarks) rather than full frame width for nod detection
        face_min_x = min(p[0] for p in coords)
        face_max_x = max(p[0] for p in coords)
        face_width = max(face_max_x - face_min_x, 1)
        n_ratio = abs(nose_y - chin_y) / face_width
        if ear < EYE_AR_THRESH:
            self.eye_counter += 1
        else:
            self.eye_counter = 0
        
        drowsiness_alert = self.eye_counter >= EYE_AR_CONSEC_FRAMES
        
        # Yawn detection
        if mar > MAR_AR_THRESH:
            self.yawn_counter += 1
        else:
            if self.yawn_counter >= YAWN_CONSEC_FRAMES:
                self.yawn_history.append(now)
            self.yawn_counter = 0
        
        # Clean up old yawn history (keep only last 60 seconds)
        self.yawn_history[:] = [t for t in self.yawn_history if now - t < 60]
        
        yawn_alert = len(self.yawn_history) >= 3
        
        # Nod detection (based on drowsiness.py behavior)
        if n_ratio < NOD_THRESH and ear < 0.25:
            # A nod indicates sustained head drop: count toward both nod alerts and drowsiness
            self.eye_counter += 2
            self.nod_counter += 1
            nod_alert = self.nod_counter >= NOD_FRAMES
        else:
            self.nod_counter = 0
            nod_alert = False
        
        return {
            'landmarks': coords,
            'ear': ear,
            'mar': mar,
            'nod_ratio': n_ratio,
            'yawn_count': len(self.yawn_history),
            'drowsiness_alert': drowsiness_alert,
            'yawn_alert': yawn_alert,
            'nod_alert': nod_alert,
            'left_eye': left_eye,
            'right_eye': right_eye,
            'mouth': mouth
        }
