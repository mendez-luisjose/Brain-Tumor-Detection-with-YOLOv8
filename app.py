import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
from utils import set_background

set_background("./imgs/background.png")

MODEL_PATH = "./model/brain_tumor_detection_model.pt"

header = st.container()
body = st.container()

model = YOLO(MODEL_PATH)

def model_prediction(img):
    pred = model.predict(img)[0]
    img_wth_box = pred.plot()

    return img_wth_box

    
with header :
    _, col1, _ = st.columns([0.05,1,0.1])
    col1.title("💥 Brain Tumor Detection 🧠")

    _, col4, _ = st.columns([0.1,1,0.2])
    col4.subheader("Computer Vision Detection with YoloV8 🧪")

    _, col5, _ = st.columns([0.05,1,0.1])
    col5.image("./imgs/train_batch.jpg")

    st.write("This are the Brain Images that the Model was trained, during 50 epochs, using more than 300 images.")

    st.header("Brain Tumors Images Examples 🧑‍🔬:")
    _, col2, _ = st.columns([0.1,1,0.2])
    col2.image("./imgs/val_batch.jpg")

    st.write("The Model was trained with the Yolov8 Architecture, for 50 epochs, using the Google Colab GPU, and with more tha 300 Images.")

with body :
    _, col1, _ = st.columns([0.15,1,0.2])
    col1.subheader("Check It-out the Brain Detection Model 🔎!")

    img = st.file_uploader("Upload a Brain Image: ", type=["png", "jpg", "jpeg"])

    _, col2, _ = st.columns([0.3,1,0.2])

    _, col5, _ = st.columns([0.8,1,0.2])

    
    if img is not None:
        image = np.array(Image.open(img))    
        col2.image(image, width=400)

        if col5.button("Detect Tumors"):
            prediction = model_prediction(image)

            _, col3, _ = st.columns([0.4,1,0.2])
            col3.header("Detection Results ✅:")

            col9, col10, col11, col12 = st.columns([0.3, 1, 0.2, 1])
            col10.subheader("Brain Image:")
            col12.subheader("Tumors Detected:")

            col7, col8 = st.columns([1,1])
            col7.image(image, width=350)
            col8.image(prediction, width=350)




