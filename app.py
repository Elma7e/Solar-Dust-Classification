import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import requests
from PIL import Image
import hashlib

# App title
st.title("Dirty vs. Clean Panel Classifier")

# Model download setup
MODEL_URL = "https://raw.githubusercontent.com/Elma7e/Solar-Dust-Classification/ba95751a356979a1dec79946354c78f67fc88c2a/binaryV1.pt"
MODEL_PATH = "binaryV1.pt"
MODEL_CHECKSUM = "8c7e2b0f36cba8f6ea6e2cc8f9b4d5d1"  # Example checksum (update with the actual checksum)


def verify_checksum(file_path, expected_checksum):
    """Verify the checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == expected_checksum


def download_model():
    """Download the model and verify its integrity."""
    with st.spinner("Downloading model..."):
        response = requests.get(MODEL_URL)
        response.raise_for_status()  # Raise an error for HTTP issues
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        st.success("Model downloaded.")

        # Verify the checksum
        if not verify_checksum(MODEL_PATH, MODEL_CHECKSUM):
            os.remove(MODEL_PATH)
            st.error("Downloaded model file is corrupted. Please try again.")
            st.stop()


# Download model if not already present
if not os.path.exists(MODEL_PATH):
    download_model()

# Load the YOLO classification model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save to a temp file for YOLO
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_file_path = tmp.name
        image.save(temp_file_path)

    try:
        # Run prediction
        with st.spinner("Classifying..."):
            results = model.predict(source=temp_file_path, save=False)

        # Parse and display results
        if results:
            res = results[0]
            if hasattr(res, "probs"):
                top1_idx = res.probs.top1
                top1_conf = res.probs.top1conf
                label = model.names[top1_idx]
                st.success(f"Prediction: {label} ({top1_conf * 100:.1f}%)")
            else:
                st.error("Model did not return classification probabilities.")
        else:
            st.error("No results from model inference.")

    except Exception as e:
        st.error(f"An error occurred during classification: {e}")

    finally:
        # Cleanup temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)