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
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="AgroGuard", layout="wide")
st.title("AgroGuard Cotton Weed Detection System")

# Load YOLO model
model = YOLO("Pred.pt")

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
    return results[0].plot()

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.annotate_kwargs = {
            'conf': True,          # Show confidence
            'line_width': 2,        # Bounding box thickness
            'font_size': 0.6,       # Font size
            'font': 'Arial.ttf',    # Font type
            'labels': True          # Show class labels
        }
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, verbose=False)
        
        # Use YOLO's native plotting with custom parameters
        annotated_frame = results[0].plot(**self.annotate_kwargs)
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

def main():
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
            st.image(image, caption="Uploaded Image", use_container_width=True)
            if st.button("Detect Weeds"):
                detected_image = detect_objects(np.array(image))
                st.image(detected_image, caption="Detected Weeds", use_container_width=True)
    
    elif choice == "Signup":
         st.subheader("Signup")
         new_username = st.text_input("Username")
         new_password = st.text_input("Password", type="password")
 
         if st.button("Signup"):
             if users_collection.find_one({"username": new_username}):
                 st.warning("Username already exists")
             else:
                 hashed_pw = hash_password(new_password)
                 users_collection.insert_one({"username": new_username, "password": hashed_pw})
                 st.success("Account created successfully!")
 
    elif choice == "Login":
         st.subheader("Login")
         username = st.text_input("Username")
         password = st.text_input("Password", type="password")
 
         if st.button("Login"):
             user = users_collection.find_one({"username": username})
             if user and verify_password(password, user['password']):
                 st.session_state['user'] = user
                 st.success(f"Logged in successfully! Welcome, {username}.")
             else:
                 st.error("Invalid username or password")
 
    elif choice == "History":
         st.subheader("Detection History")
         if 'user' in st.session_state:
             user_id = st.session_state['user']['_id']
             images = list(get_user_images(user_id))
             if images:
                 for img in images:
                     col1, col2, col3 = st.columns([3, 1, 1])
                     with col1:
                         image_bytes = img['image']
                         hist_image = Image.open(io.BytesIO(image_bytes))
                         st.image(hist_image, caption="Previous Detection", use_container_width=True)
                     with col2:
                         img_bytes = io.BytesIO(image_bytes)
                         st.download_button("Download", img_bytes, file_name=f"detection_{img['_id']}.jpg", mime="image/jpeg", key=f"download_{img['_id']}")
                     with col3:
                         if st.button("Delete", key=str(img['_id'])):
                             delete_image(img['_id'])
                             st.success("Image deleted successfully!")
             else:
                 st.info("No detection history found.")
         else:
             st.warning("Please log in to view history")
    
    elif choice == "Mobile Camera Detection":
        st.title("Real-Time Cotton Weed Detection (Mobile)")
        st.markdown("""
        ### Instructions:
        1. Click 'Start Camera' below
        2. Allow camera access when prompted
        3. Point camera at cotton field
        4. See real-time weed detection
        """)
        
        # Camera component with start/stop controls
        webrtc_ctx = webrtc_streamer(
            key="example",
            video_processor_factory=VideoProcessor,
            rtc_configuration={  # Required for production
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": True,
                "audio": False
            }
        )
        
        st.markdown("---")
        st.info("ℹ️ Switch between front/rear camera using your device's camera toggle button")

if __name__ == '__main__':
    main()
