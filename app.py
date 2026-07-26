import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image


# Load Saved Objects in a Cached Function
@st.cache_resource
def load_artifacts():
     model = tf.keras.models.load_model("Model/model.keras")
     return model
model= load_artifacts()

# Page Config

st.set_page_config(
    page_title="CIFAR-10 Image Classification and Prediction",
    page_icon="🖼️",
    layout="centered"
)
class_dist=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
st.title("CIFAR-10 Image Classification and Prediction")
st.write("Upload an image to classify it.")

# User Inputs

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("")
    st.write("Classifying...")

    image = image.resize((32, 32,))
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
# Prediction

if st.button("Predict"):
    if uploaded_file is not None:
                prediction = model.predict(image)
                st.success(
                        f" {class_dist[np.argmax(prediction)]}"
                    )
    else:
        st.warning("Please upload an image file to make a prediction.")



   