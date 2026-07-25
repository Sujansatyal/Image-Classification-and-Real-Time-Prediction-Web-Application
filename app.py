import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf


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


image =  np.reshape(uploaded_file, ( 32, 32))
# Prediction

if st.button("Predict"):
    if uploaded_file is not None:
                prediction = model.predict(image)
 





# Display Prediction
    st.success(
        f" {class_dist[np.argmax(prediction)]}"
    )