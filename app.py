import streamlit as st
import numpy as np
import mediapipe as mp
import cv2
import time
import sqlite3
import hashlib

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from mediapipe.tasks import python
from mediapipe.tasks.python import vision




async_processing=True



class PostureDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, timestamp)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            h, w, _ = img.shape

            left = landmarks[11]
            right = landmarks[12]

            if abs(left.y - right.y) > 0.05:
                posture = "Bad Posture"
                color = (0, 0, 255)
            else:
                posture = "Good Posture"
                color = (0, 255, 0)

            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, color, -1)

            cv2.putText(img, posture, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return img
    




def main_app():
    st.title(f"👋 Welcome {st.session_state.get('name', 'User')}")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.write("🧍 Your Posture Correcction App Starts Here")

    # ✅ MOVE YOUR CAMERA CODE HERE
    webrtc_streamer(
        key="posture",
        video_transformer_factory=PostureDetector
    )

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="PostureSense", layout="centered")


# -------------------- DATABASE SETUP --------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    name TEXT,
    phone TEXT PRIMARY KEY,
    password TEXT
)
""")
conn.commit()

# -------------------- HASH FUNCTION --------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# -------------------- REGISTER --------------------
def register_user(name, phone, password):
    if len(phone) != 10 or not phone.isdigit():
        return "Invalid phone number"

    try:
        c.execute("INSERT INTO users (name, phone, password) VALUES (?, ?, ?)",
                  (name, phone, hash_password(password)))
        conn.commit()
        return "Success"
    except:
        return "Phone already exists"

# -------------------- LOGIN --------------------
def login_user(phone, password):
    c.execute("SELECT * FROM users WHERE phone=? AND password=?",
              (phone, hash_password(password)))
    return c.fetchone()

# -------------------- UI STYLE --------------------
st.markdown("""
<style>
.login-box {
    background-color: #1c1f26;
    padding: 35px;
    border-radius: 12px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.4);
}
.title {
    text-align: center;
    font-size: 28px;
    color: white;
    margin-bottom: 20px;
}
.stButton>button {
    width: 100%;
    border-radius: 8px;
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------- SESSION --------------------


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    
if "name" not in st.session_state:
    st.session_state.name = ""

# -------------------- AUTH PAGE --------------------
def auth_page():
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.markdown('<div class="title">🧍 PostureSense</div>', unsafe_allow_html=True)

    menu = ["Login", "Register"]
    choice = st.radio("", menu, horizontal=True)

    if choice == "Register":
        name = st.text_input("Full Name")
        phone = st.text_input("Phone Number")
        password = st.text_input("Password", type="password")

        if st.button("Create Account"):
            result = register_user(name, phone, password)

            if result == "Success":
                st.success("Account created! You can now login.")
            elif result == "Phone already exists":
                st.error("Phone number already registered")
            else:
                st.error("Enter valid 10-digit phone number")

    if choice == "Login":
        phone = st.text_input("Phone Number")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = login_user(phone, password)

            if user:
                st.session_state.logged_in = True
                st.session_state.name = user[0]
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid phone or password")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- MAIN APP --------------------

    # 👉 Put your posture detection code here

# -------------------- FLOW CONTROL --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    auth_page()
    st.stop()   # 🚨 THIS STOPS APP FROM RUNNING
else:
    main_app()

# Load model
base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.PoseLandmarker.create_from_options(options)





# st.title("PostureSense")
# st.write("Real-time posture monitoring system")

