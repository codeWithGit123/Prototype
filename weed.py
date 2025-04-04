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
@st.cache_resource
def load_model():
    return YOLO("Pred.pt")

model = load_model()

# MongoDB Connection
client = pymongo.MongoClient("mongodb+srv://mdabdur2004:ArFeb2004@cluster0.zq6ldvu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["agroguard"]
users_collection = db["users"]
images_collection = db["images"]

def save_image(user_id, image_bytes):
    images_collection.insert_one({"user_id": user_id, "image": image_bytes})

def get_user_images(user_id):
    return list(images_collection.find({"user_id": user_id}))

def delete_image(image_id):
    images_collection.delete_one({"_id": ObjectId(image_id)})

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def detect_objects(frame):
    results = model(frame)
    for box in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = map(int, box[:6])
        label = f"{model.names[int(cls)]}: {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def main():
    st.set_page_config(page_title="AgroGuard", layout="wide")
    st.title("AgroGuard Cotton Weed Detection System")
    
    menu = ["Home", "Signup", "Login", "History", "Mobile Camera Detection"]
    choice = st.sidebar.selectbox("Menu", menu)

    if 'user' in st.session_state:
        st.sidebar.markdown(f"**Welcome, {st.session_state['user']['username']}!**")
        if st.sidebar.button("Logout"):
            del st.session_state["user"]
            st.sidebar.success("Logged out successfully!")

    if choice == "Home":
        st.subheader("Cotton Weed Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Detect Weeds"):
                detected_image = detect_objects(np.array(image))
                st.image(detected_image, caption="Detected Weeds", use_column_width=True)
    
    elif choice == "Mobile Camera Detection":
        st.subheader("Live Mobile Camera Detection")
        camera_index = st.selectbox("Select Camera", [0, 1], index=0)
        start_button = st.button("▶️ Start Camera")
        stop_button = st.button("⛔ Stop Camera")
        frame_placeholder = st.empty()
        
        if start_button:
            cap = cv2.VideoCapture(camera_index)
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    st.warning("❌ Could not access the camera. Please check permissions.")
                    break
                frame = detect_objects(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame, channels="RGB", use_column_width=True)
        
        if stop_button:
            cap.release()
            cv2.destroyAllWindows()
            frame_placeholder.empty()

if __name__ == '__main__':
    main()
