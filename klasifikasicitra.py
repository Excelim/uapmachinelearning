import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Class labels
class_labels = ["Bug", "Dubas", "Healthy", "Honey"]

def load_models():
    """Load VGG16 and CNN models from disk."""
    vgg16_model = tf.keras.models.load_model('D:/webmachinelearningmluap/src/model/vgg16_model.keras')
    cnn_model = tf.keras.models.load_model('D:/webmachinelearningmluap/src/model/cnn_model.keras')
    return vgg16_model, cnn_model

def predict(image, model):
    """Predict the class of the uploaded image using the selected model."""
    # Preprocess the image
    image = image.resize((224, 224))  # Resize to the input size of the model
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform prediction
    predictions = model.predict(image)[0]  # Get predictions for the first batch
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    return predicted_class, confidence, predictions

def plot_confidence_summary(class_labels, all_predictions):
    """Generate a bar plot for average confidence by class."""
    # Average predictions across all uploaded images
    avg_predictions = np.mean(all_predictions, axis=0)

    # Plot bar chart
    fig, ax = plt.subplots()
    ax.bar(class_labels, avg_predictions, color="skyblue")
    ax.set_ylabel("Average Confidence")
    ax.set_title("Average Confidence by Class")
    ax.set_ylim(0, 1)  # Confidence is between 0 and 1
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Load the models
vgg16_model, cnn_model = load_models()

def main():
    st.set_page_config(page_title="Prediksi Penyakit Daun Palam", layout="wide")

    # Add custom CSS to change background color to soft pink pastel
    st.markdown(
        """
        <style>
        body {
            background-color: #FADADD;  /* Soft Pink Pastel */
        }
        .reportview-container {
            background-color: #FADADD;  /* Soft Pink Pastel */
        }
        .sidebar .sidebar-content {
            background-color: #FADADD;  /* Soft Pink Pastel */
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # Sidebar menu
    with st.sidebar:
        st.title("Select Options")
        selected_model = st.radio(
            "Choose a Model:",
            ('VGG16', 'CNN')
        )
        model = vgg16_model if selected_model == 'VGG16' else cnn_model

    st.title("Prediksi Penyakit Daun Palam")
    st.write("Upload up to three images of money plants to predict their condition.")

    # Multi-upload images
    uploaded_files = st.file_uploader(
        "Upload up to 3 Images",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
        help="You can upload up to three images.",
    )

    if uploaded_files:
        # Limit to three images
        uploaded_files = uploaded_files[:3]

        # Display uploaded images
        st.write("Uploaded Images:")
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"{uploaded_file.name}", use_container_width=True)

        # Button to trigger prediction
        if st.button("Predict"):
            # Initialize summary table and all_predictions list
            summary_data = []
            all_predictions = []

            for uploaded_file in uploaded_files:
                # Open image and make predictions
                image = Image.open(uploaded_file)
                predicted_class, confidence, predictions = predict(image, model)

                # Append results to the summary table
                summary_data.append({
                    "Image Name": uploaded_file.name,
                    "Predicted Class": class_labels[predicted_class],
                    "Confidence": f"{confidence:.2f}"
                })

                # Collect predictions for plotting
                all_predictions.append(predictions)

            # Display summary table
            st.markdown("### Summary of Predictions")
            summary_df = pd.DataFrame(summary_data)  # Create DataFrame for summary
            st.table(summary_df)

            # Plot confidence summary
            st.markdown("### Average Confidence by Class")
            plot_confidence_summary(class_labels, all_predictions)

    # Footer section with additional explanation or links
    st.markdown("---")
    st.markdown(
        "Developed with ❤ by excelim (#). Powered by Streamlit and TensorFlow."
    )

if __name__ == "__main__":
    main()
