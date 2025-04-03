import streamlit as st
import pymongo
from bson import ObjectId
import bcrypt
from PIL import Image
import io
import time
import requests
from ultralytics import YOLO
import numpy as np
import cv2  # OpenCV for preprocessing
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


# Load YOLO model
model = YOLO("Pred.pt")  # Replace with your trained model path

# MongoDB Connection (update the connection string as needed)
client = pymongo.MongoClient("mongodb+srv://mdabdur2004:ArFeb2004@cluster0.zq6ldvu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["agroguard"]
users_collection = db["users"]
images_collection = db["images"]

# --------------------------
# Helper Functions
# --------------------------
def save_image(user_id, image_bytes):
    images_collection.insert_one({
        "user_id": user_id,
        "image": image_bytes
    })

def get_user_images(user_id):
    return list(images_collection.find({"user_id": user_id}))

def delete_image(image_id):
    images_collection.delete_one({"_id": ObjectId(image_id)})

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

class WeedDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detected, result_image, weeds_info = detect_weeds(image_rgb)
        if detected and result_image is not None:
            return cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        return image

def preprocess_image(image):
    """
    Preprocess the image by converting it to a NumPy array, resizing it,
    enhancing sharpness to remove blurriness, and applying random brightness augmentation.
    """
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Resize image to 640x640
    resized = cv2.resize(image_np, (640, 640))
    
    # Sharpening using an unsharp mask (helps remove blurriness)
    # Apply Gaussian blur
    gaussian = cv2.GaussianBlur(resized, (0, 0), 3)
    # Sharpen: add weighted difference
    sharpened = cv2.addWeighted(resized, 1.5, gaussian, -0.5, 0)
    
    # Augmentation: Random brightness adjustment
    # Convert from RGB to HSV color space for brightness control
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_RGB2HSV)
    # Random brightness factor between 0.8 and 1.2
    brightness_factor = np.random.uniform(0.8, 1.2)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255).astype(np.uint8)
    # Convert back to RGB
    augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return augmented


def detect_weeds(image):
    """
    Pass the preprocessed image to the YOLO model and return detection result.
    """
    results = model(image)
    detected = False
    result_image = None
    for result in results:
        if len(result.boxes) > 0:
            detected = True
            result_image = result.plot()  # Annotated image as a NumPy array
    return detected, result_image
# --------------------------
# Main Application
# --------------------------
def main():
    st.set_page_config(page_title="AgroGuard", layout="wide")
    st.title("AgroGuard Cotton Weed Detection System")

    # Sidebar Menu
    menu = ["Home", "Signup", "Login", "History", "Mobile Camera Detection"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Display welcome message and logout option if logged in
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
                with st.spinner("Processing..."):
                    time.sleep(1)
                    # Preprocess the image before detection
                    preprocessed_image = preprocess_image(image)
                    detected, result_array = detect_weeds(preprocessed_image)
                
                if detected and result_array is not None:
                    st.image(result_array, caption="Detected Weeds", use_container_width=True)
                    # Convert NumPy array to PIL image for saving
                    pil_img = Image.fromarray(result_array)
                    if 'user' in st.session_state:
                        img_bytes = io.BytesIO()
                        pil_img.save(img_bytes, format="JPEG")
                        save_image(st.session_state['user']['_id'], img_bytes.getvalue())
                        st.success("Detection saved successfully!")
                else:
                    st.warning("No Weed Detected")

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

    if choice == "Mobile Camera Detection":
        st.subheader("Live Weed Detection using Mobile Camera")
        start_button = st.button("Start Camera")
        stop_button = st.button("Stop Camera")
        frame_placeholder = st.empty()
        weeds_placeholder = st.empty()

        if start_button:
            st.success("Camera started!")
            webrtc_ctx = webrtc_streamer(
                key="camera",
                video_transformer_factory=WeedDetectionTransformer
            )

            if webrtc_ctx and webrtc_ctx.state.playing:
                while webrtc_ctx.video_receiver:
                    frame = webrtc_ctx.video_receiver.get_frame()
                    if frame is None:
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
                        st.warning("Camera stopped!")
                        break


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

if __name__ == '__main__':
    main()
