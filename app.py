import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import requests
from PIL import Image

# App title
st.title("Dirty vs. Clean Panel Classifier")

# Model download setup
MODEL_URL = "https://raw.githubusercontent.com/Elma7e/Solar-Dust-Classification/ba95751a356979a1dec79946354c78f67fc88c2a/binaryV1.pt"
MODEL_PATH = "binaryV1.pt"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        st.success("Model downloaded.")

# Load the YOLO classification model
model = YOLO(MODEL_PATH)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save to a temp file for YOLO
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    # Run prediction
    with st.spinner("Classifying..."):
        results = model.predict(source=tmp_path, save=False)

    # Parse and display results
    if results:
        res = results[0]
        if hasattr(res, "probs"):
            top1_idx = res.probs.top1
            top1_conf = res.probs.top1conf
            label = model.names[top1_idx]
            st.success(f"Prediction: {label} ({top1_conf*100:.1f}%)")
        else:
            st.error("Model did not return classification probabilities.")
    else:
        st.error("No results from model inference.")
