import streamlit as st
import pymongo
from bson import ObjectId
import bcrypt
from PIL import Image
import io
import time
import numpy as np
import cv2
from ultralytics import YOLO

# --------------------------
# Load YOLO Model
# --------------------------
model = YOLO("Pred.pt")  # Replace with your trained model path

# --------------------------
# MongoDB Connection
# --------------------------
client = pymongo.MongoClient("mongodb+srv://mdabdur2004:ArFeb2004@cluster0.zq6ldvu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["agroguard"]
users_collection = db["users"]
images_collection = db["images"]

# --------------------------
# Helper Functions
# --------------------------
def save_image(user_id, image_bytes):
    """Save detected image to MongoDB"""
    images_collection.insert_one({"user_id": user_id, "image": image_bytes})

def get_user_images(user_id):
    """Retrieve user's detection history"""
    return list(images_collection.find({"user_id": user_id}))

def delete_image(image_id):
    """Delete an image from history"""
    images_collection.delete_one({"_id": ObjectId(image_id)})

def hash_password(password):
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    """Verify hashed password"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# --------------------------
# Weed Detection Function
# --------------------------
def detect_weeds(image):
    """Detect weeds using the YOLO model"""
    results = model(image)
    detected = False
    result_image = None
    for result in results:
        if len(result.boxes) > 0:
            detected = True
            result_image = result.plot()  # Annotated image
    return detected, result_image

# --------------------------
# Live Camera Detection (Using OpenCV)
# --------------------------
def mobile_camera_detection(camera_source=0):
    """Live weed detection using OpenCV"""
    st.subheader("Live Weed Detection using OpenCV")

    source_type = st.radio("Select Camera Source:", ["USB Webcam", "Mobile (IP Camera)"])
    
    if source_type == "Mobile (IP Camera)":
        ip_url = st.text_input("Enter Mobile Camera URL", "http://192.168.1.100:8080/video")
        camera_source = ip_url  # Use IP camera stream
    
    start_button = st.button("Start Camera")
    stop_button = st.button("Stop Camera")

    frame_placeholder = st.empty()  # Placeholder for video frames

    if start_button:
        cap = cv2.VideoCapture(camera_source)  # Open camera stream
        
        if not cap.isOpened():
            st.error("Error: Unable to access the camera.")
            return
        
        st.success("Camera started successfully!")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture frame")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            detected, result_image = detect_weeds(frame_rgb)

            if detected and result_image is not None:
                frame_rgb = result_image  # Show detected frame

            img_pil = Image.fromarray(frame_rgb)  # Convert NumPy array to PIL image
            frame_placeholder.image(img_pil, caption="Live Weed Detection", use_column_width=True)

            if stop_button:
                break

        cap.release()
        st.warning("Camera stopped!")

# --------------------------
# Streamlit App
# --------------------------
def main():
    st.set_page_config(page_title="AgroGuard", layout="wide")
    st.title("🌱 AgroGuard - Cotton Weed Detection System")

    menu = ["Home", "Signup", "Login", "History", "Mobile Camera Detection"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Show logged-in user info
    if 'user' in st.session_state:
        st.sidebar.markdown(f"**Welcome, {st.session_state['user']['username']}!**")
        if st.sidebar.button("Logout"):
            del st.session_state["user"]
            st.sidebar.success("Logged out successfully!")

    # --------------------------
    # Home: Upload Image for Detection
    # --------------------------
    if choice == "Home":
        st.subheader("Cotton Weed Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Detect Weeds"):
                with st.spinner("Processing..."):
                    time.sleep(1)
                    detected, result_array = detect_weeds(np.array(image))

                if detected and result_array is not None:
                    st.image(result_array, caption="Detected Weeds", use_container_width=True)
                    if 'user' in st.session_state:
                        img_bytes = io.BytesIO()
                        Image.fromarray(result_array).save(img_bytes, format="JPEG")
                        save_image(st.session_state['user']['_id'], img_bytes.getvalue())
                        st.success("Detection saved successfully!")
                else:
                    st.warning("No Weed Detected")

    # --------------------------
    # Signup
    # --------------------------
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

    # --------------------------
    # Login
    # --------------------------
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

    # --------------------------
    # Mobile Camera Detection (Using OpenCV)
    # --------------------------
    elif choice == "Mobile Camera Detection":
        mobile_camera_detection()  # Call OpenCV-based detection

    # --------------------------
    # View Detection History
    # --------------------------
    elif choice == "History":
        st.subheader("Detection History")
        if 'user' in st.session_state:
            images = get_user_images(st.session_state['user']['_id'])
            if images:
                for img in images:
                    st.image(Image.open(io.BytesIO(img['image'])), caption="Previous Detection", use_container_width=True)
                    if st.button("Delete", key=str(img['_id'])):
                        delete_image(img['_id'])
                        st.success("Image deleted successfully!")
            else:
                st.info("No detection history found.")
        else:
            st.warning("Please log in to view history")

if __name__ == '__main__':
    main()
