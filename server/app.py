import cv2
import streamlit as st
from datetime import datetime

from safety_engine import SafetyEngine


st.set_page_config(
    page_title="Unified Driver Safety Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --safe: #16a34a;
        --warn: #d97706;
        --danger: #dc2626;
        --ink-soft: #9ca3af;
    }

    .hero {
        border: 1px solid rgba(148, 163, 184, 0.25);
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.92));
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }

    .hero-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .hero-sub {
        color: var(--ink-soft);
        font-size: 0.92rem;
    }

    .card {
        border: 1px solid rgba(148, 163, 184, 0.2);
        background: rgba(15, 23, 42, 0.75);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
        backdrop-filter: blur(8px);
    }

    .status-safe { border-left: 6px solid var(--safe); }
    .status-warning { border-left: 6px solid var(--warn); }
    .status-danger { border-left: 6px solid var(--danger); }

    .label {
        font-size: 0.85rem;
        color: var(--ink-soft);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .value {
        font-size: 1.45rem;
        font-weight: 700;
        margin: 2px 0 4px 0;
    }

    .pill {
        display: inline-block;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        margin-right: 6px;
    }

    .pill-safe { background: rgba(22, 163, 74, 0.2); color: #86efac; }
    .pill-warning { background: rgba(217, 119, 6, 0.2); color: #fdba74; }
    .pill-danger { background: rgba(220, 38, 38, 0.2); color: #fca5a5; }

    .alert-item {
        border-radius: 8px;
        padding: 8px 10px;
        margin-bottom: 6px;
        border: 1px solid rgba(248, 113, 113, 0.28);
        background: rgba(127, 29, 29, 0.25);
    }

    .video-shell {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.7);
        padding: 10px;
    }

    .section-title {
        margin-top: 2px;
        margin-bottom: 8px;
        font-weight: 600;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine() -> SafetyEngine:
    return SafetyEngine(frame_skip=2, smoothing_window=12)


engine = get_engine()


def _status_class(status: str) -> str:
    if status == "SAFE":
        return "status-safe"
    if status == "WARNING":
        return "status-warning"
    return "status-danger"


def _pill(priority: str) -> str:
    if priority == "WARNING":
        cls = "pill-warning"
    elif priority == "CRITICAL" or priority == "EMERGENCY":
        cls = "pill-danger"
    else:
        cls = "pill-safe"
    return f'<span class="pill {cls}">{priority}</span>'


def _draw_camera_overlay(frame, state: dict):
    """Draw status/SOS overlay directly on camera frame for instant visibility."""
    status = state.get("overall_status", "SAFE")
    sos = bool(state.get("sos_triggered", False))

    if status == "SAFE":
        color = (22, 163, 74)
    elif status == "WARNING":
        color = (6, 119, 217)
    else:
        color = (38, 38, 220)

    cv2.rectangle(frame, (16, 16), (380, 110), (10, 10, 10), -1)
    cv2.rectangle(frame, (16, 16), (380, 110), color, 2)

    cv2.putText(
        frame,
        f"STATUS: {status}",
        (28, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"SOS: {'ON' if sos else 'OFF'}",
        (28, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255) if sos else (120, 220, 120),
        2,
        cv2.LINE_AA,
    )

    return frame


def render_camera_feed(container, frame_rgb, state: dict):
    """CameraFeed section (left column)."""
    with container:
        st.markdown('<div class="card"><div class="section-title">Camera Feed</div>', unsafe_allow_html=True)
        st.image(frame_rgb, channels="RGB", use_container_width=True)

        st.markdown(
            f"""
            <div style="margin-top:8px;">
                {_pill(state['priority_level'])}
                <span class="pill {'pill-danger' if state['sos_triggered'] else 'pill-safe'}">SOS: {state['sos_triggered']}</span>
                <span class="pill {'pill-danger' if state['overall_status'] == 'DANGER' else 'pill-safe' if state['overall_status'] == 'SAFE' else 'pill-warning'}">SYSTEM: {state['overall_status']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_driver_panel(container, state: dict):
    """DriverPanel section (middle column / Page 1)."""
    conf = state["confidence"]
    d = state["metrics"]["drowsiness"]
    hands = state["metrics"].get("hands", {})
    distraction_label = state["metrics"]["distraction"].get("label", "Unknown")

    with container:
        st.markdown('<div class="card"><div class="section-title">Driver Monitoring</div>', unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        r1c1.metric("Drowsiness Score", f"{conf.get('drowsiness', 0.0):.2f}")
        r1c2.metric("Distraction", "DETECTED" if state["distraction"] else "CLEAR")

        r2c1, r2c2 = st.columns(2)
        r2c1.metric("Hand Detection", state.get("hand_status_text", "HANDS ON WHEEL ✅"))
        r2c2.metric("Hands Count", str(hands.get("hands_detected", 0)))

        st.caption(f"Distraction Label: {distraction_label}")

        m1, m2 = st.columns(2)
        m1.metric("EAR", f"{d.get('ear', 0.0):.3f}")
        m2.metric("MAR", f"{d.get('mar', 0.0):.3f}")

        m3, m4 = st.columns(2)
        m3.metric("Nod", f"{d.get('nod_ratio', 0.0):.3f}")
        m4.metric("Yawns", str(d.get("yawn_count", 0)))

        if state.get("alerts"):
            st.caption("Active Alerts")
            for alert in state["alerts"]:
                st.markdown(f'<div class="alert-item">{alert}</div>', unsafe_allow_html=True)
        else:
            st.success("No driver-side anomalies.")

        st.markdown("</div>", unsafe_allow_html=True)


def render_heart_panel(container, state: dict):
    """HeartPanel section (right column / Page 2)."""
    hc = state["metrics"]["heart"]

    with container:
        st.markdown('<div class="card"><div class="section-title">Heart Monitoring</div>', unsafe_allow_html=True)

        h1, h2 = st.columns(2)
        h1.metric("BPM", f"{hc.get('bpm', 0.0):.1f}")
        h2.metric("Heart Class", str(hc.get("class", "NORMAL")))

        st.caption("Class probabilities")
        st.bar_chart(hc.get("probabilities", {}), use_container_width=True)

        st.caption("ECG waveform")
        st.line_chart(state.get("heart_chart", []), height=220, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Unified Safety Controls")
    run = st.toggle("Start Unified Monitoring", value=False)
    st.caption("All models run together in one coordinated pipeline.")
    st.divider()
    st.caption("Monitored channels")
    st.markdown("- Drowsiness\n- Distraction\n- Hands Off Wheel\n- Heart Risk")
    st.caption("Output levels")
    st.markdown("- SAFE\n- WARNING\n- DANGER")

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Unified AI Driver Safety Dashboard</div>
        <div class="hero-sub">Camera + Driver Monitoring + Heart Monitoring in a single real-time view.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

camera_col, driver_col, heart_col = st.columns([4, 3, 3], gap="medium")

camera_feed_box = camera_col.empty()
driver_panel_box = driver_col.empty()
heart_panel_box = heart_col.empty()

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera feed unavailable. Check webcam permissions and availability.")
            break

        frame = cv2.resize(frame, (960, 540))
        state = engine.process_step(frame)

        overlay_frame = _draw_camera_overlay(state["frame"].copy(), state)
        frame_rgb = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)

        with camera_feed_box.container():
            render_camera_feed(st.container(), frame_rgb, state)

        with driver_panel_box.container():
            render_driver_panel(st.container(), state)

        with heart_panel_box.container():
            render_heart_panel(st.container(), state)

    cap.release()
else:
    st.info("Enable 'Start Unified Monitoring' from the sidebar to launch the unified dashboard.")
