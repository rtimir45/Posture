import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import threading # Required for thread safety
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- SHARED STATE OBJECT ---
class SessionState:
    def __init__(self):
        self.calibration_requested = False
        self.baseline_score = None
        self.lock = threading.Lock()

# Initialize the shared state in Streamlit's session_state
if 'app_state' not in st.session_state:
    st.session_state.app_state = SessionState()

# --- MODEL SETUP ---
base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)

class PostureProcessor(VideoProcessorBase):
    def __init__(self, shared_state):
        self.shared_state = shared_state

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        result = detector.detect_for_video(mp_image, int(time.time() * 1000))

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            l_ear, r_ear = landmarks[7], landmarks[8]
            l_shldr, r_shldr = landmarks[11], landmarks[12]

            # Calculate current neck extension (front-facing metric)
            current_score = ((l_shldr.y + r_shldr.y) / 2) - ((l_ear.y + r_ear.y) / 2)

            # --- THREAD-SAFE CALIBRATION LOGIC ---
            with self.shared_state.lock:
                # If UI clicked 'Calibrate', capture this frame's score
                if self.shared_state.calibration_requested:
                    self.shared_state.baseline_score = current_score
                    self.shared_state.calibration_requested = False
                
                baseline = self.shared_state.baseline_score

            # --- POSTURE LOGIC ---
            status, color = "Detecting...", (255, 255, 255)
            if baseline is not None:
                # If neck shrinks by more than 15% from baseline
                if current_score < (baseline * 0.85):
                    status, color = "SLOUCHING!", (0, 0, 255)
                else:
                    status, color = "GOOD POSTURE", (0, 255, 0)
            else:
                status = "Press Calibrate"

            cv2.putText(img, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return img

# --- UI ---
st.title("PostureSense Pro")

# Sidebar button sets the flag in the shared state
if st.sidebar.button("Calibrate Now"):
    with st.session_state.app_state.lock:
        st.session_state.app_state.calibration_requested = True
    st.sidebar.success("Calibrating on next frame...")

# Display current baseline in UI
if st.session_state.app_state.baseline_score:
    st.sidebar.metric("Target Score", round(st.session_state.app_state.baseline_score, 3))

webrtc_streamer(
    key="posture",
    # Pass the shared state to the processor constructor
    video_processor_factory=lambda: PostureProcessor(st.session_state.app_state),
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
