import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from pathlib import Path
import numpy as np
from PIL import Image
import io

# Configure Streamlit page
st.set_page_config(
    page_title="Image Classification: VGG16 and Custom CNN",
    page_icon="🖼️",
    layout="wide",
)

# Header section
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #4CAF50;">🖼️ Image Classification</h1>
        <p style="font-size: 18px; color: #555;">Upload an image to get predictions using your VGG16 and Custom CNN models.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar for user input
st.sidebar.header("Model Selection")
model_options = ["VGG16", "Custom CNN", "Both"]
model_choice = st.sidebar.radio("Choose a model:", model_options)

# File uploader
upload = st.file_uploader("Upload an image (PNG, JPG, JPEG):", type=['png', 'jpg', 'jpeg'])

# Function to load VGG16 model
@st.cache_resource
def load_vgg16_model():
    model_path = Path("D:/webmachinelearning/src/model/vgg16_model.keras")
    return tf.keras.models.load_model(str(model_path))


# Function to load custom CNN model
@st.cache_resource
def load_custom_model():
    model_path = Path("D:\webmachinelearning\src\model\cnn_model.keras")  # Adjust path to your .keras model
    return tf.keras.models.load_model(str(model_path))

# Prediction function
# Prediction function for both models
def predict_image(uploaded_image, vgg16_model=None, custom_cnn_model=None):
    # Normalisasi input
    preprocess_input = lambda x: x / 255.0  # Normalisasi untuk CNN custom
    class_names = ['Bug', 'Dubas', 'Healthy', 'Honey']  # Sesuaikan dengan nama kelas Anda

    try:
        # Load dan proses gambar
        img = load_img(uploaded_image, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        results = []

        # Prediksi dengan VGG16 model jika dipilih
        if vgg16_model:
            vgg16_predictions = vgg16_model.predict(img_array)
            vgg16_probabilities = tf.nn.softmax(vgg16_predictions[0])
            vgg16_top_class = np.argmax(vgg16_probabilities)
            results.append(("VGG16", class_names[vgg16_top_class], float(vgg16_probabilities[vgg16_top_class])))

        # Prediksi dengan Custom CNN model jika dipilih
        if custom_cnn_model:
            custom_predictions = custom_cnn_model.predict(img_array)
            custom_probabilities = tf.nn.softmax(custom_predictions[0])
            custom_top_class = np.argmax(custom_probabilities)
            results.append(("Custom CNN", class_names[custom_top_class], float(custom_probabilities[custom_top_class])))

        return results
    
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
        return []

# Main content
if st.button("Predict"):
    if upload is not None:
        # Check for correct file type (PNG, JPG, JPEG)
        try:
            img = Image.open(upload)
            img_format = img.format.lower()
            if img_format not in ['png', 'jpg', 'jpeg']:
                st.error("Invalid image format. Please upload a PNG, JPG, or JPEG image.")
            else:
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(upload, caption="Uploaded Image", use_container_width=True)

                with col2:
                    st.subheader("Prediction Results")

                    with st.spinner("Loading models and processing image..."):
                        results = []
                        
                        # Load both models if needed
                        if model_choice in ["VGG16", "Both"]:
                            vgg16_model = load_vgg16_model()
                            vgg16_results = predict_image(upload, vgg16_model=vgg16_model)
                            results.extend(vgg16_results)

                        if model_choice in ["Custom CNN", "Both"]:
                            custom_model = load_custom_model()
                            custom_results = predict_image(upload, custom_cnn_model=custom_model)
                            results.extend(custom_results)

                        for model_name, label, prob in results:
                            st.write(f"### Results from {model_name}:")
                            st.write(f"**{label}**: {prob:.2%}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload an image first!")


# Footer
st.markdown(
    """
    <hr style='border:1px solid #4CAF50'>
    <div style='text-align: center; font-size: 14px; color: #888;'>
        Built with ❤️ using Streamlit and TensorFlow
    </div>
    """,
    unsafe_allow_html=True,
)
