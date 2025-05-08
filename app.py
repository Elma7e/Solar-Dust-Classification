import streamlit as st
from ultralytics import YOLO
import torch
import tempfile
from PIL import Image

# App title
st.title("Dirty vs. Clean Panel Classifier")

# Load the YOLO classification model
device = "cuda" if torch.cuda.is_available() else "cpu"
# Adjust the path to your model file as needed
model = YOLO("runs/classify/train/weights/best.pt", device=device)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to a temp file for YOLO
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    # Run prediction
    with st.spinner("Classifying..."):
        results = model.predict(source=tmp_path, save=False)

    # Parse classification results
    if results and len(results) > 0:
        res = results[0]
        # For classification models, .probs gives probabilities
        if hasattr(res, "probs"):
            probs = res.probs.cpu().numpy()
            # Map class index to name
            class_names = model.names
            best_idx = int(probs.argmax())
            best_name = class_names[best_idx]
            best_score = float(probs[best_idx])
            st.success(f"Prediction: {best_name} ({best_score*100:.1f}%)")
        else:
            st.error("Could not extract probabilities from model output.")
    else:
        st.error("No results returned by the model.")
