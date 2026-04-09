import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --- 1. SESSION STATE ---
class SessionState:
    def __init__(self):
        self.calibration_requested = False
        self.baseline_score = None
        self.lock = threading.Lock()

if "app_state" not in st.session_state:
    st.session_state["app_state"] = SessionState()

app_state = st.session_state["app_state"]


# --- 2. PROCESSOR CLASS ---
class PostureProcessor(VideoProcessorBase):
    def __init__(self, shared_state):
        super().__init__()  # FIX 1: Always call super().__init__()
        self.shared_state = shared_state
        self._start_time = time.time()
        self._last_ts_ms = -1  # FIX 2: Track last timestamp to guarantee monotonic increase

        # FIX 3: Create detector PER INSTANCE, not globally.
        # The global detector is not thread-safe. Each processor thread
        # must own its own detector.
        base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:  # FIX 4: Correct type hints
        img = frame.to_ndarray(format="bgr24")

        try:
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # FIX 5: Guarantee strictly monotonic timestamps.
            # time.time() alone can repeat on fast machines or after sleep.
            ts_ms = int((time.time() - self._start_time) * 1000)
            if ts_ms <= self._last_ts_ms:
                ts_ms = self._last_ts_ms + 1
            self._last_ts_ms = ts_ms

            result = self.detector.detect_for_video(mp_image, ts_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                h, w, _ = img.shape

                l_ear, r_ear   = landmarks[7], landmarks[8]
                l_shldr, r_shldr = landmarks[11], landmarks[12]

                current_score = (
                    (l_shldr.y + r_shldr.y) / 2
                ) - (
                    (l_ear.y + r_ear.y) / 2
                )

                with self.shared_state.lock:
                    if self.shared_state.calibration_requested:
                        self.shared_state.baseline_score = current_score
                        self.shared_state.calibration_requested = False
                    baseline = self.shared_state.baseline_score

                if baseline is not None:
                    if current_score < (baseline * 0.85):
                        status, color = "SLOUCHING!", (0, 0, 255)
                    else:
                        status, color = "GOOD POSTURE", (0, 255, 0)
                else:
                    status, color = "Press Calibrate", (255, 255, 255)

                cv2.putText(img, status, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                for idx in [7, 8, 11, 12]:
                    pt = landmarks[idx]
                    cv2.circle(img, (int(pt.x * w), int(pt.y * h)), 5, color, -1)

        except Exception as e:
            print(f"Frame processing error: {e}")

        # FIX 6: Return an av.VideoFrame, NOT a raw numpy array.
        # streamlit-webrtc requires av.VideoFrame from recv().
        # Returning a plain ndarray causes the pipeline to stall → frozen output.
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def __del__(self):
        # Clean up the detector when the processor is garbage-collected
        if hasattr(self, "detector"):
            self.detector.close()


# --- 3. STREAMLIT UI ---
st.title("PostureSense")
st.write("Real-time posture monitoring system")

if st.sidebar.button("Calibrate Good Posture"):
    with app_state.lock:
        app_state.calibration_requested = True
    st.sidebar.success("Calibrating on next frame…")

if app_state.baseline_score is not None:
    st.sidebar.metric("Target Score", round(app_state.baseline_score, 3))

webrtc_streamer(
    key="posture-system",
    video_processor_factory=lambda: PostureProcessor(app_state),
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
