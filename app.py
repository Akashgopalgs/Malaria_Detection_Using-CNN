import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the classification model
@st.cache_resource
# def load_classification_model():
#     model = load_model('maleria_detection_model.tf')
#     # Save the model in .keras format
#     model.save('maleria_detection_model.keras', save_format='keras')
#     return model

    model = load_model('maleria_detection_model.tf')
    # Save the model in .keras format
    model.save('maleria_detection_model.keras', save_format='keras')

def load_classification_model():
    model = load_model('maleria_detection_model.keras')  # Load the .keras file
    return model




# Load the VGG16 feature extractor
@st.cache_resource
def load_feature_extractor():
    vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    vgg16_base.trainable = False  # Freeze the base
    return vgg16_base

classification_model = load_classification_model()
feature_extractor = load_feature_extractor()

# Preprocess the uploaded image and extract features
def preprocess_and_extract_features(image):
    img = image.resize((64, 64))  # Resize to match VGG16 input
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Apply VGG16 preprocessing
    features = feature_extractor.predict(img_array)  # Extract features
    return features

# Streamlit UI
st.title("Malaria Detection Web App")

uploaded_file = st.file_uploader("Upload a blood smear image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=250)
    st.markdown("</div>", unsafe_allow_html=True)

    # Extract features and predict
    try:
        processed_features = preprocess_and_extract_features(image)
        prediction = classification_model.predict(processed_features)[0][0]  # Assuming binary classification

        # Display result
        if prediction > 0.5:
            st.success("### Prediction: Malaria Detected")
        else:
            st.success("### Prediction: No Malaria Detected")
    except Exception as e:
        st.error(f"Error in processing: {e}")
