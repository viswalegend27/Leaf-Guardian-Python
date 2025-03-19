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
    'pepper_bell_healthy': {
        'description': "Your pepper-bell plant is healthy!",
        'prevention': "Keep up consistent watering and monitoring."
    },
    'tomato_healthy': {
        'description': "Your tomato plant is healthy!",
        'prevention': "Keep up consistent watering and monitoring."
    },
    'potato_healthy': {
        'description': "Your potato plant is healthy!",
        'prevention': "Keep up consistent watering and monitoring."
    }
}

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY"))

if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro-001")
    except Exception as e:
        st.warning(f"⚠️ Gemini AI error: {e}. Using local data.")
        model = None
else:
    st.warning("⚠️ No Gemini API Key found! Using local disease information.")
    model = None

# Load the TensorFlow model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("1.keras")

model_path = "1.keras"
potato_disease_model = None
if os.path.exists(model_path):
    potato_disease_model = load_model()
    st.success("✅ Model loaded successfully!")
else:
    st.error("⚠️ Model file not found! Please upload a valid model.")

# Define class names and their symptom descriptions
CLASS_NAMES = [
    'pepper_bell_bacterial_spot', 'pepper_bell_healthy', 'potato_early_blight',
    'potato_late_blight', 'potato_healthy', 'tomato_early_blight', 'tomato_late_blight', 'tomato_healthy'
]

# Symptom descriptions for each class to improve API accuracy
CLASS_DESCRIPTIONS = {
    'pepper_bell_bacterial_spot': "Small, water-soaked spots on leaves that enlarge, turn brown with a yellow halo, and may merge, causing leaf drop.",
    'pepper_bell_healthy': "Leaves are uniformly green, with no spots, discoloration, or wilting.",
    'potato_early_blight': "Small, brown spots with concentric rings (target-like) on older, lower leaves, often with yellowing around the spots.",
    'potato_late_blight': "Dark, water-soaked spots on leaves that turn black or brown, often with white fuzzy mold on the underside in wet conditions.",
    'potato_healthy': "Leaves are uniformly green, with no spots, discoloration, or wilting.",
    'tomato_early_blight': "Small, brown spots with concentric rings (target-like) on older, lower leaves, often with yellowing around the spots.",
    'tomato_late_blight': "Dark, water-soaked spots on leaves that turn black or brown, often with white fuzzy mold on the underside in wet conditions.",
    'tomato_healthy': "Leaves are uniformly green, with no spots, discoloration, or wilting."
}

# Function to process class name
def process_class_name(predicted_class):
    healthy_suffix = "healthy"
    parts = predicted_class.lower().split('_')

    if predicted_class.endswith(healthy_suffix):
        plant = '_'.join(parts[:-1])
        condition = healthy_suffix
        is_healthy = True
        disease = None
    else:
        plant = parts[0]
        condition = '_'.join(parts[1:])
        is_healthy = False
        disease = condition

    return plant, disease, is_healthy

# Function to get disease info
def get_disease_info(plant, disease):
    disease_key = f"{plant}_{disease}" if disease else f"{plant}_healthy"

    if model and disease:
        prompt = f"Provide a description, symptoms, prevention, and treatment for {disease} in {plant} plants."
        try:
            response = model.generate_content(prompt)
            return {"description": response.text, "prevention": "Follow agricultural best practices."}
        except Exception as e:
            st.warning(f"⚠️ Gemini AI error: {e}. Using local data.")

    return DISEASE_INFO.get(disease_key, {
        "description": "No description available.",
        "prevention": "Consult an agricultural expert."
    })

# Main App Layout
st.title('🌿 Plant Disease Guardian')

# Tabs for organization
tab1, tab2 = st.tabs(["🖼️ Upload & Predict", "ℹ️ About"])

with tab1:
    uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file and potato_disease_model:
        with st.spinner("🔍 Processing image..."):
            try:
                # Progress bar for visual feedback
                progress_bar = st.progress(0)

                # Process the image
                img = Image.open(uploaded_file)
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)

                # Predict with TensorFlow model
                predictions = potato_disease_model.predict(img_array)
                predictions_prob = tf.nn.softmax(predictions[0]).numpy()  # Ensure probabilities
                model_predicted_class = CLASS_NAMES[np.argmax(predictions_prob)]
                confidence = np.max(predictions_prob)

                # Initialize variables
                predicted_class = model_predicted_class
                used_api = False

                # Use Gemini API if confidence is low or very high but potentially incorrect
                if confidence < 0.7 or (confidence >= 0.95 and model):
                    # Create a detailed prompt with symptom descriptions
                    prompt = "Identify which of the following classes this leaf image most likely belongs to based on the symptoms described:\n"
                    for class_name in CLASS_NAMES:
                        prompt += f"- {class_name}: {CLASS_DESCRIPTIONS[class_name]}\n"
                    prompt += "Respond with only the class name exactly as listed (e.g., 'tomato_late_blight')."
                    try:
                        response = model.generate_content([prompt, img])
                        api_predicted_class = response.text.strip().replace(' ', '_')
                        if api_predicted_class in CLASS_NAMES:
                            if api_predicted_class != model_predicted_class:
                                predicted_class = api_predicted_class
                                used_api = True
                                if confidence >= 0.95:
                                    st.info("Model was highly confident, but API provided a different classification.")
                                else:
                                    st.info("Using Gemini API prediction due to low model confidence.")
                            else:
                                st.info("API agreed with model's prediction.")
                        else:
                            st.warning(f"Gemini API returned an invalid class: {api_predicted_class}. Using model's prediction.")
                    except Exception as e:
                        st.warning(f"Gemini API error: {e}. Using model's prediction.")

                # Process the predicted class
                plant, disease, is_healthy = process_class_name(predicted_class)
                display_name = f"{plant.capitalize()} - {disease.replace('_', ' ').capitalize()}" if disease else f"{plant.replace('_', ' ').capitalize()} - Healthy"
                disease_info = get_disease_info(plant, disease)

                # Update progress bar
                for percent in range(100):
                    progress_bar.progress(percent + 1)

                # Display results
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"### 🌱 Prediction: **{display_name}**")
                    if used_api:
                        st.markdown("**Source:** Gemini API")
                    else:
                        st.markdown(f"**Confidence:** `{confidence:.2f}`")
                    st.image(img, caption='📷 Uploaded Leaf Image', width=300)

                with col2:
                    if is_healthy:
                        st.subheader("✅ Plant is Healthy")
                        st.markdown(disease_info["description"])
                    else:
                        st.subheader("📋 Disease Details")
                        st.markdown(f"**Description:** {disease_info['description']}")
                        st.markdown(f"**Prevention:** {disease_info['prevention']}")

            except Exception as e:
                st.error(f"⚠️ Error processing image: {e}")

with tab2:
    st.markdown("""
    ## ℹ️ About This App
    **Plant Disease Guardian** is an AI-powered tool for detecting plant diseases from leaf images.
    - 📷 **Upload a photo** using your device.
    - 🌿 **Get instant predictions** on plant diseases.
    - 🤖 **Uses AI (Gemini API)** for additional disease details and treatment solutions when needed.
    - 🚀 **Optimized for speed** with TensorFlow & Streamlit.

    ### How It Works:
    1. **User Uploads an Image** - The image of a plant leaf is uploaded.
    2. **Image Preprocessing** - The image is resized and converted into an array for model input.
    3. **Disease Classification** - The trained TensorFlow model predicts the disease; Gemini API is used if confidence is low or potentially incorrect.
    4. **AI-Powered Explanation** - The Gemini AI provides detailed disease information for diseases.
    5. **Results Displayed** - The app shows the diagnosis, source (model or API), and treatment suggestions.
    """)

# Custom CSS for styling
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