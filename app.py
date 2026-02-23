import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time

st.set_page_config(page_title="Construction Safety Monitor", layout="wide", page_icon="🚧")

st.title("Construction Safety Helmet Monitor")
st.markdown("Upload a construction video (or use demo) → detects workers/helmets → beeps alarm on violations (shown as red status).")

# Load model (cache it)
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # your hard-hat model

model = load_model()

# Sidebar settings
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.20, 0.90, 0.40, 0.05)
use_demo_video = st.sidebar.checkbox("Use built-in demo video (Pexels construction clip)", value=True)

# Video source
video_bytes = None
if use_demo_video:
    try:
        with open("construction_demo.mp4", "rb") as f:
            video_bytes = f.read()
        st.sidebar.success("Demo video loaded")
    except FileNotFoundError:
        st.sidebar.warning("Demo video not found → please upload one below")
else:
    uploaded_video = st.sidebar.file_uploader("Upload construction video (mp4)", type=["mp4"])
    if uploaded_video is not None:
        video_bytes = uploaded_video.read()

if video_bytes is None:
    st.info("Please enable demo or upload a video to start.")
    st.stop()

# Temp save for OpenCV
with open("temp_input.mp4", "wb") as f:
    f.write(video_bytes)

cap = cv2.VideoCapture("temp_input.mp4")
if not cap.isOpened():
    st.error("Cannot open video file. Try another one.")
    st.stop()

# Placeholders for UI
frame_placeholder = st.empty()
col1, col2, col3 = st.columns(3)
worker_ph = col1.empty()
helmet_ph = col2.empty()
no_helmet_ph = col3.empty()
status_ph = st.empty()

last_update = time.time()
violation = False

st.markdown("Processing video... (it loops like 24/7 monitoring)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
        continue

    # Resize for faster processing/display
    frame = cv2.resize(frame, (960, 540))  # balanced size

    # Inference
    results = model(frame, conf=conf_threshold, verbose=False)[0]

    helmet_count = 0
    no_helmet_count = 0

    for box in results.boxes:
        cls_id = int(box.cls)
        label = results.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == 'Hardhat':
            helmet_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
            cv2.putText(frame, "Helmet", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        elif label == 'NO-Hardhat':
            no_helmet_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, "NO HELMET", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    worker_count = helmet_count + no_helmet_count

    # Alarm simulation (browser can't play winsound → use visual + text)
    now = time.time()
    if no_helmet_count >= 1 and (now - last_update > 1.0):
        violation = True
        last_update = now
    else:
        violation = False

    # Update UI
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    worker_ph.metric("Workers Detected", worker_count)
    helmet_ph.metric("Helmets", helmet_count)
    no_helmet_ph.metric("No Helmet", no_helmet_count)

    if violation:
        status_ph.error("🚨 SAFETY VIOLATION – NO HELMET DETECTED! Alarm triggered.")
    else:
        status_ph.success("✅ All workers compliant")

    # Slow down to ~ real-time feel (adjust as needed)
    time.sleep(0.05)

cap.release()