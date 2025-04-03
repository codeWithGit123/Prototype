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
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --------------------------
# Load YOLO Model
# --------------------------
model = YOLO("Pred.pt")  # Replace with your model path

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
    images_collection.insert_one({"user_id": user_id, "image": image_bytes})

def get_user_images(user_id):
    return list(images_collection.find({"user_id": user_id}))

def delete_image(image_id):
    images_collection.delete_one({"_id": ObjectId(image_id)})

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# --------------------------
# Weed Detection Transformer for Live Camera
# --------------------------
class WeedDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detected, result_image = detect_weeds(image_rgb)
        if detected and result_image is not None:
            return cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        return image

# --------------------------
# Weed Detection for Uploaded Images
# --------------------------
def detect_weeds(image):
    results = model(image)
    detected = False
    result_image = None
    for result in results:
        if len(result.boxes) > 0:
            detected = True
            result_image = result.plot()  # Returns annotated image
    return detected, result_image

# --------------------------
# Streamlit Application
# --------------------------
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
    # Live Camera Detection (Fixed WebRTC Permissions)
    # --------------------------
    elif choice == "Mobile Camera Detection":
        st.subheader("Live Weed Detection using Mobile Camera")

        # Ensure Camera Permissions
        st.write(
            """
            - **Ensure HTTPS** (WebRTC doesn't work on HTTP)
            - **Manually Allow Camera Access** in browser settings.
            """
        )

        # Fix WebRTC State Issues
        if "camera_active" not in st.session_state:
            st.session_state["camera_active"] = False

        try:
            webrtc_ctx = webrtc_streamer(
                key="camera",
                mode=WebRtcMode.SENDRECV,
                video_transformer_factory=WeedDetectionTransformer,
                async_processing=True,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                video_html_attrs={"autoplay": True, "muted": True, "playsinline": True}
            )
        except Exception as e:
            st.error(f"Error starting camera: {e}")
            st.warning("Please allow camera access in your browser settings and refresh the page.")

        if webrtc_ctx and webrtc_ctx.state.playing:
            if not st.session_state["camera_active"]:
                st.session_state["camera_active"] = True
                st.success("Camera started! Grant permission if prompted.")
        else:
            if st.session_state["camera_active"]:
                st.session_state["camera_active"] = False
                st.error("Camera access denied! Please allow camera access in browser settings.")

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
