import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO('yolo.pt')

st.title("Weed Detection System")

up_img = st.file_uploader("Upload an Image",type=["jpg","png","jpeg"])


if up_img is not None:
    image_pil = Image.open(up_img).convert("RGB")  # Converts RGBA → RGB
    image_np = np.array(image_pil)
    results = model(image_np)

    for r in results:
        im_array = r.plot()
        oimg = Image.fromarray(im_array)
    
    st.image(oimg,caption="Detected Weeds",use_column_width=True)
