import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Plant Disease Guardian", layout="wide", page_icon="🌿")

# Local disease information as a fallback
DISEASE_INFO = {
    'pepperbell_bacterial_spot': {
        'description': "Bacterial spot in pepperbell plants is caused by Xanthomonas euvesicatoria, leading to dark, water-soaked lesions with yellowish halos.",
        'prevention': "Use disease-free seeds, apply copper-based bactericides, and avoid overhead watering."
    },
    'pepper_bell_healthy': {
        'description': "Your pepper bell plant is healthy and showing no signs of disease!",
        'prevention': "Keep up consistent watering, spacing, and monitoring."
    },
    'potato_early_blight': {
        'description': "Early blight in potato plants is caused by Alternaria solani, causing dark spots with concentric rings on leaves.",
        'prevention': "Rotate crops, use fungicides like chlorothalonil, and remove infected debris."
    },
    'potato_late_blight': {
        'description': "Late blight in potato plants is caused by Phytophthora infestans, leading to dark, water-soaked lesions and rapid crop loss.",
        'prevention': "Use resistant varieties, apply fungicides like metalaxyl, and ensure good air circulation."
    },
    'potato_healthy': {
        'description': "Your potato plant is healthy and showing no signs of disease!",
        'prevention': "Keep up consistent watering, spacing, and monitoring."
    },
    'tomato_early_blight': {
        'description': "Early blight in tomato plants is caused by Alternaria solani, causing dark spots with concentric rings on leaves.",
        'prevention': "Rotate crops, use fungicides like mancozeb, and stake plants for better airflow."
    },
    'tomato_late_blight': {
        'description': "Late blight in tomato plants is caused by Phytophthora infestans, leading to dark, water-soaked lesions and rapid fruit rot.",
        'prevention': "Use resistant varieties, apply fungicides like chlorothalonil, and avoid overhead watering."
    },
    'tomato_healthy': {
        'description': "Your tomato plant is healthy and showing no signs of disease!",
        'prevention': "Keep up consistent watering, spacing, and monitoring."
    }
}

# Configure Gemini API using environment variable
api_key = os.getenv("GEMINI_API_KEY")
model = None
if api_key:
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-001")
        st.write("✅ Using Gemini AI model for disease details.")
    except Exception as e:
        st.warning(f"Error initializing Gemini model: {e}. Using local data.")

# Load the TensorFlow model with error handling
MODEL_PATH = "1.keras"
potato_disease_model = None
if os.path.exists(MODEL_PATH):
    try:
        potato_disease_model = tf.keras.models.load_model(MODEL_PATH)
        st.write("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
else:
    st.error("⚠️ Model file not found! Ensure '1.keras' is in the project directory.")

# Class names from the model (ensure consistency)
CLASS_NAMES = [
    'pepperbell_bacterial_spot', 'pepper_bell_healthy', 'potato_early_blight', 
    'potato_late_blight', 'potato_healthy', 'tomato_early_blight', 'tomato_late_blight', 'tomato_healthy'
]

# Function to extract disease name and check if healthy
def process_class_name(predicted_class):
    parts = predicted_class.split('_')
    plant = parts[0].lower()
    condition = '_'.join(parts[1:]).lower()
    is_healthy = "healthy" in condition
    disease = condition if not is_healthy else None
    return plant, disease, is_healthy

# Function to get disease info (Gemini AI or fallback to local data)
def get_disease_info(plant, disease):
    disease_key = f"{plant}_{disease}" if disease else f"{plant}_healthy"
    
    # Try Gemini API
    if model and disease:
        prompt = f"Provide a concise description and prevention tips for {disease} in {plant} plants."
        try:
            response = model.generate_content(prompt)
            return {"description": response.text, "prevention": "Follow agricultural best practices."}
        except Exception as e:
            st.warning(f"Error fetching info from Gemini API: {e}. Using local data.")

    # Fallback to local data
    return DISEASE_INFO.get(disease_key, {
        "description": "No description available.",
        "prevention": "Consult an agricultural expert."
    })

# Main app content
st.title('🌿 Plant Disease Guardian')

# File uploader
uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file and potato_disease_model:
    with st.spinner("🔍 Processing image..."):
        try:
            # Process the image
            img = Image.open(uploaded_file)
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            # Predict
            predictions = potato_disease_model.predict(img_array)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])

            # Extract details
            plant, disease, is_healthy = process_class_name(predicted_class)
            display_name = f"{plant.capitalize()} - {disease.replace('_', ' ').capitalize()}" if disease else f"{plant.capitalize()} - Healthy"
            disease_info = get_disease_info(plant, disease)

            # Display results
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"### 🌱 Prediction: **{display_name}**")
                st.markdown(f"**Confidence:** `{confidence:.2f}`")
                st.image(img, caption='📷 Uploaded Leaf Image', width=300)

            with col2:
                if is_healthy:
                    st.subheader("✅ Plant Health Status")
                    st.markdown(disease_info["description"])
                else:
                    st.subheader("📋 Disease Details")
                    st.markdown(disease_info["description"])

        except Exception as e:
            st.error(f"⚠️ Error processing image: {e}")

# Custom CSS for better UI
st.markdown("""
    <style>
    h1 { text-align: center; color: #2ecc71; }
    .stMarkdown { font-size: 16px; color: #e0e0e0; }
    .stFileUploader > div > button {
        background-color: #2ecc71; color: white; border-radius: 5px;
    }
    .stFileUploader > div > button:hover { background-color: #27ae60; }
    </style>
""", unsafe_allow_html=True)
