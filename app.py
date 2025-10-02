import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model once and cache it
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

st.title("Coronary Image Classifier")
st.write("Upload an image and let the model predict!")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show preview
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    labels = ['Left', 'Right']
    confidence = prediction[class_index] * 100

    st.success(f"✅ Prediction: **{labels[class_index]}** ({confidence:.2f}%)")
