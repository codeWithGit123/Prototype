import streamlit as st
import pymongo
from bson import ObjectId
import bcrypt
from PIL import Image
import io
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("Pred.pt")

# MongoDB Connection
client = pymongo.MongoClient("mongodb+srv://mdabdur2004:ArFeb2004@cluster0.zq6ldvu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["agroguard"]
users_collection = db["users"]
images_collection = db["images"]

# --------------------------
# Helper Functions
# --------------------------
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def detect_weeds(image):
    results = model(image)
    detected = False
    result_image = None
    weeds_info = []
    for result in results:
        if len(result.boxes) > 0:
            detected = True
            result_image = result.plot()
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                weeds_info.append((model.names[cls], conf))
    return detected, result_image, weeds_info

# --------------------------
# Main Application
# --------------------------
def main():
    st.set_page_config(page_title="AgroGuard", layout="wide")
    st.title("AgroGuard Cotton Weed Detection System")

    menu = ["Home", "Signup", "Login", "History", "Webcam Detection"]
    choice = st.sidebar.selectbox("Menu", menu)

    if 'user' in st.session_state:
        st.sidebar.markdown(f"**Welcome, {st.session_state['user']['username']}!**")
        if st.sidebar.button("Logout"):
            del st.session_state["user"]
            st.sidebar.success("Logged out successfully!")

    if choice == "Webcam Detection":
        st.subheader("Live Weed Detection using Webcam")
        start_button = st.button("Start Webcam")
        stop_button = st.button("Stop Webcam")
        frame_placeholder = st.empty()
        weeds_placeholder = st.empty()

        if start_button:
            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to capture frame")
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detected, result_image, weeds_info = detect_weeds(frame)
                
                if detected and result_image is not None:
                    frame_placeholder.image(result_image, caption="Weed Detection", use_column_width=True)
                    weeds_placeholder.write("**Detected Weeds:**")
                    for name, conf in weeds_info:
                        weeds_placeholder.write(f"{name}: {conf:.2f}")
                else:
                    frame_placeholder.image(frame, caption="No Weed Detected", use_column_width=True)
                
                if stop_button:
                    cap.release()
                    cv2.destroyAllWindows()
                    break
        
if __name__ == '__main__':
    main()
