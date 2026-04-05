# Driver-Safety-main

An integrated Driver Health and Safety Monitoring project with:
- A React + Vite frontend (landing page, docs view, profile/login UI, and embedded model UI)
- A Python Streamlit backend for real-time drowsiness, distraction, and heart-health monitoring

## Project Structure

- `client/`: Frontend web application
- `server/`: AI/ML backend and real-time monitoring app

## How To Run

### Prerequisites

- Node.js 20+
- Python 3.10+
- Webcam access (for live detection modes)

### 1) Run the backend (Streamlit)

From project root:

```powershell
# If using venv in project root
.\.venv\Scripts\python.exe -m pip install -r server\requirements.txt
.\.venv\Scripts\python.exe -m streamlit run server\app.py --server.port 8501 --server.address 0.0.0.0
```

Backend URL:
- http://localhost:8501

### 2) Run the frontend (Vite)

From project root:

```powershell
cd client
npm install
npm run dev
```

Frontend URL:
- http://localhost:3000

## How The System Works

- The frontend controls UI views (home, docs, profile, login, and model showcase).
- The model showcase view embeds the Streamlit backend inside an iframe.
- The backend handles all AI/ML inference in three modes:
  - Drowsiness detection (MediaPipe landmarks + geometric rules)
  - Distraction detection (YOLO + ResNet features + SVM classifier)
  - Heart health monitoring (Keras model + physiological heuristics + synthetic ECG stream)

## File-by-File Explanation

## Root

- `.gitignore`
  - Ignores generated files, environment artifacts, and unnecessary local files from version control.

- `README.md`
  - This project documentation file.

## client/

- `client/index.html`
  - Main HTML shell. Contains the root element where React mounts.

- `client/metadata.json`
  - App metadata (name/description), useful for tooling and deployment metadata.

- `client/package.json`
  - Frontend dependencies and scripts:
  - `dev`: starts Vite dev server on port 3000
  - `build`: creates production build
  - `preview`: previews production build
  - `lint`: TypeScript type-check pass

- `client/README.md`
  - Original AI Studio-generated setup notes for the client app.

- `client/tsconfig.json`
  - TypeScript compiler configuration for the React app.

- `client/vite.config.ts`
  - Vite configuration:
  - React + Tailwind Vite plugins
  - optional `GEMINI_API_KEY` exposure to client build
  - alias configuration
  - HMR behavior

### client/src/

- `client/src/main.tsx`
  - React entrypoint. Mounts `App` into `#root` and imports global styles.

- `client/src/App.tsx`
  - Top-level UI controller. Handles page/view switching:
  - `home`
  - `profile`
  - `showcase`
  - `docs`
  - `login`
  - Also toggles light/dark theme class on the root HTML element.

- `client/src/index.css`
  - Global styling and design tokens:
  - Tailwind import
  - theme variables (colors/fonts)
  - dark/light mode variable sets
  - shared global effects and animations

### client/src/components/

- `client/src/components/Navbar.tsx`
  - Top navigation bar:
  - section navigation
  - view switching
  - profile/login actions
  - light mode toggle

- `client/src/components/Hero.tsx`
  - Home hero section with animated marketing and “Model Showcase” action.

- `client/src/components/Problem.tsx`
  - Problem statement section describing failures in isolated safety systems.

- `client/src/components/Solution.tsx`
  - Solution section describing model stack and capabilities.

- `client/src/components/Response.tsx`
  - Three-tier emergency response section (Detect, Alert, Prevent/Rescue).

- `client/src/components/CTA.tsx`
  - Call-to-action section for requesting a demo.

- `client/src/components/Footer.tsx`
  - Footer branding and policy links.

- `client/src/components/Profile.tsx`
  - Driver profile form UI for personal, vehicle, medical, and emergency-contact info.
  - Currently UI-only (no backend persistence logic in this component).

- `client/src/components/Login.tsx`
  - Login form UI. On submit, returns to `home` view.
  - Currently UI-only (no authentication API wiring in this component).

- `client/src/components/Docs.tsx`
  - Technical documentation page in-app (architecture, model pipeline, and logic tables).

- `client/src/components/ModelShowcase.tsx`
  - Embeds backend Streamlit UI via iframe at `http://localhost:8501`.

## server/

- `server/app.py`
  - Main Streamlit application.
  - Creates the live dashboard and camera pipeline.
  - Loads and coordinates all models.
  - Provides 3 runtime modes:
  - Drowsiness
  - Distraction
  - Heart Health
  - Shows alerts, metrics, charts, and live frame output.

- `server/detection.py`
  - Drowsiness logic utilities:
  - EAR (Eye Aspect Ratio)
  - MAR (Mouth Aspect Ratio)
  - head nod ratio checks
  - includes `DrowsinessDetector` class with temporal counters and alerts.

- `server/distraction_model.py`
  - Distraction pipeline:
  - loads YOLO detector
  - loads CNN feature extractor (ResNet18 backbone)
  - loads SVM (+ optional scaler)
  - runs frame skip optimization and confidence-based labeling
  - applies heuristic override using Haar face/eye cascades.

- `server/heart_monitoring.py`
  - Heart-health pipeline:
  - loads Keras model (`medical_heart_anomaly_model.h5`)
  - normalizes RR interval windows
  - predicts class (NORMAL/WARNING/EMERGENCY)
  - applies confidence-aware heuristic fallback
  - generates demo RR data
  - computes BPM
  - generates synthetic ECG points for chart streaming.

- `server/safety_engine.py`
  - Unified orchestration layer for all models.
  - Initializes and caches drowsiness, distraction, and heart models once.
  - Runs one coordinated real-time cycle per frame.
  - Applies temporal smoothing, confidence scoring, fusion logic, and SOS triggering.

- `server/requirements.txt`
  - Python dependencies for Streamlit app, CV/ML stack, and plotting.

### server/models/

- `server/models/medical_heart_anomaly_model.h5`
  - Trained Keras/TensorFlow model for heart anomaly classification.

- `server/models/PretrainCNN_99.75.pth`
  - Trained PyTorch CNN weights used as visual feature extractor.

- `server/models/yolov8m.pt`
  - YOLOv8 model weights for person/object detection in frames.

## Notes

- `ModelShowcase.tsx` assumes backend is running at `localhost:8501`.
- Several frontend components are currently presentation-focused and can be wired to APIs as next steps.
- For camera-based modes, ensure OS/browser camera permission is granted.
