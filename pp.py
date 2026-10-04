import streamlit as st
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from PIL import Image
import numpy as np
import os


model_path = 'best_model.h5'

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    st.error("Model file not found. Please ensure 'best_model.h5' is in the correct directory.")
    model = None

IMG_SIZE = (224, 224)

st.title("Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image to detect if it's NORMAL or PNEUMONIA.")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Chest X-ray.', use_column_width=True)

    # Preprocess the image
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        st.success(f"Prediction: NORMAL (Confidence: {1 - prediction:.2f})")
    else:
        st.error(f"Prediction: PNEUMONIA (Confidence: {prediction:.2f})")
elif uploaded_file is not None:
    st.error("Model is not loaded. Unable to make predictions.")