import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Plant Disease Guardian", layout="wide", page_icon="🌿")

# Local Disease Information (Fallback)
DISEASE_INFO = {
    'pepperbell_bacterial_spot': {
        'description': "Bacterial spot in pepperbell plants is caused by Xanthomonas euvesicatoria.",
        'prevention': "Use disease-free seeds, apply copper-based bactericides."
    },
    'tomato_late_blight': {
        'description': "Late blight in tomato plants is caused by Phytophthora infestans.",
        'prevention': "Use resistant varieties, apply fungicides like chlorothalonil."
    },
    'potato_healthy': {
        'description': "Your potato plant is healthy!",
        'prevention': "Keep up consistent watering and monitoring."
    }
}

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro-001")
    except Exception as e:
        st.warning(f"Gemini AI error: {e}. Using local data.")

# Load the TensorFlow model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("1.keras")

model_path = "1.keras"
potato_disease_model = None
if os.path.exists(model_path):
    potato_disease_model = load_model()
    st.write("✅ Model loaded successfully!")
else:
    st.error("⚠️ Model file not found!")

# Define class names
CLASS_NAMES = [
    'pepperbell_bacterial_spot', 'pepper_bell_healthy', 'potato_early_blight', 
    'potato_late_blight', 'potato_healthy', 'tomato_early_blight', 'tomato_late_blight', 'tomato_healthy'
]

# Function to process class name
def process_class_name(predicted_class):
    parts = predicted_class.split('_')
    plant = parts[0].lower()
    condition = '_'.join(parts[1:]).lower()
    is_healthy = "healthy" in condition
    disease = condition if not is_healthy else None
    return plant, disease, is_healthy

# Function to get disease info from AI or local
def get_disease_info(plant, disease):
    disease_key = f"{plant}_{disease}" if disease else f"{plant}_healthy"

    if model and disease:
        prompt = f"Provide a description, prevention, and treatment for {disease} in {plant} plants."
        try:
            response = model.generate_content(prompt)
            return {"description": response.text, "prevention": "Follow agricultural best practices."}
        except Exception as e:
            st.warning(f"Gemini AI error: {e}. Using local data.")

    return DISEASE_INFO.get(disease_key, {
        "description": "No description available.",
        "prevention": "Consult an agricultural expert."
    })

# Main App Layout
st.title('🌿 Plant Disease Guardian')

# Use Tabs for Better Organization
tab1, tab2 = st.tabs(["🖼️ Upload & Predict", "ℹ️ About"])

with tab1:
    # Image Upload
    uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file and potato_disease_model:
        with st.spinner("🔍 Processing image..."):
            try:
                # Progress Bar
                progress_bar = st.progress(0)

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

                # Update progress bar
                for percent in range(100):
                    progress_bar.progress(percent + 1)

                # Show results
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

                # Show Alternative Prediction if Confidence is Low
                if confidence < 0.7:
                    alt_class = CLASS_NAMES[np.argsort(predictions[0])[-2]]
                    st.warning(f"⚠️ Model is uncertain. Alternative prediction: **{alt_class.capitalize()}**")

            except Exception as e:
                st.error(f"⚠️ Error processing image: {e}")

with tab2:
    st.markdown("""
        ## ℹ️ About This App
        **Plant Disease Guardian** is an AI-powered tool for detecting plant diseases from leaf images.  
        - 📷 **Upload a photo** using your device.  
        - 🌿 **Get instant predictions** on plant diseases.  
        - 🤖 **Uses AI (Gemini API)** for additional disease details and treatment solutions.  
        - 🚀 **Optimized for speed** with TensorFlow & Streamlit.  

        ### How It Works:
        1. **User Uploads an Image** - The image of a plant leaf is uploaded.  
        2. **Image Preprocessing** - The image is resized and converted into an array for model input.  
        3. **Disease Classification** - The trained TensorFlow model predicts the disease.  
        4. **AI-Powered Explanation** - The Gemini AI provides detailed disease information.  
        5. **Results Displayed** - The app shows the diagnosis, confidence score, and treatment suggestions.  
    """)

# Custom CSS for Styling
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
