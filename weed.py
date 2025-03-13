import streamlit as st
import requests
import io
from PIL import Image

# Flask API URL
FLASK_API_URL = "https://flaskbackend-7lj5.onrender.com/"

st.title("Weed Detection System 🌿🚜")

st.write("Upload an image to detect weeds.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert image to bytes
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to bytes for API request
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()

    # Send request to Flask API
    if st.button("Detect Weeds"):
        st.write("Processing...")
        response = requests.post(FLASK_API_URL, files={"image": ("image.jpg", image_bytes, "image/jpeg")})

        if response.status_code == 200:
            # Show the result
            result_image = Image.open(io.BytesIO(response.content))
            st.image(result_image, caption="Detected Weeds", use_container_width=True)
        else:
            st.error("Failed to process the image. Try again!")

