import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
# import threading
import time

model = YOLO('yolo.pt')

st.title("Weed Detection System")

up_img = st.file_uploader("Upload an Image",type=["jpg","png","jpeg"])
# up_vid = st.file_uploader("Upload a Video File",type=["mp4","mov","avi","mkv","webm"])


if up_img is not None:
    image_pil = Image.open(up_img).convert("RGB")  # Converts RGBA → RGB
    image_np = np.array(image_pil)
    results = model(image_np)

    for r in results:
        im_array = r.plot()
        oimg = Image.fromarray(im_array)
    
    st.image(oimg,caption="Detected Weeds",use_column_width=True)

# if up_vid is not None:
#     tfile = tempfile.NamedTemporaryFile(delete=False)
#     tfile.write(up_vid.read())

#     cap = cv2.VideoCapture(tfile.name)
#     stframe = st.empty()
    
#     # Reduce processing load by skipping frames
#     frame_skip = 2  # Process every 2nd frame for better speed
#     fps = int(cap.get(cv2.CAP_PROP_FPS)) // frame_skip
    
#     def process_video():
#         frame_count = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Skip frames to improve performance
#             if frame_count % frame_skip == 0:
#                 frame = cv2.resize(frame, (640, 360))  # Resize for faster processing
#                 results = model(frame)

#                 annotated_frame = results[0].plot()
#                 annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

#                 stframe.image(annotated_frame, channels="RGB", use_column_width=True)
            
#             frame_count += 1
#             time.sleep(1 / fps)  # Maintain normal playback speed

#         cap.release()

#     # Run the video processing in a separate thread
#     thread = threading.Thread(target=process_video)
#     thread.start()