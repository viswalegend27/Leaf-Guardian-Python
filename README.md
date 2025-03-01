# 🌿 Plant Disease Guardian

## 🚀 Overview
**Plant Disease Guardian** is an AI-powered tool for detecting plant diseases from leaf images. It uses a deep learning model trained with TensorFlow to classify diseases and integrates the Gemini AI API to provide detailed information about the detected diseases.

## 🎯 Features
- 📷 **Upload Leaf Images** – Users can upload images of plant leaves.
- 🌱 **AI-Powered Disease Detection** – A trained deep learning model classifies the disease.
- 🤖 **Gemini AI Integration** – Provides detailed descriptions, symptoms, and treatments.
- 📊 **Confidence Score Display** – Shows how confident the model is about its prediction.
- ⚠️ **Alternative Prediction** – If confidence is low, an alternative result is displayed.
- 🎨 **User-Friendly UI** – Built with Streamlit for an interactive experience.

---

## 📦 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/viswalegend27/Leaf-Guardian-Python.git
cd Leaf-Guardian-Python
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up API Keys (Optional for AI Integration)
Create a `.env` file or use `.streamlit/secrets.toml` and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

---

## 🚀 Running the App
```bash
streamlit run main.py
```

---

## 📷 Usage
1️⃣ **Upload an image** of a plant leaf.
2️⃣ **AI model processes the image** and predicts the disease.
3️⃣ **Get instant results** with confidence scores.
4️⃣ **Additional AI insights** (if Gemini API is configured).

---

## 🏗️ Directory Structure
```
Leaf-Guardian-Python/
│── .streamlit/
│   ├── secrets.toml (optional for API keys)
│── app.py
│── main.py
│── requirements.txt
│── runtime.txt
│── README.md
│── 1.keras (Trained TensorFlow Model)
```

---

## 🛠 Technologies Used
- **Python** – Core programming language
- **TensorFlow/Keras** – Deep learning model for classification
- **Streamlit** – Web framework for interactive UI
- **Google Gemini AI** – For enhanced disease information
- **PIL (Pillow)** – Image processing
- **dotenv** – For handling environment variables

---

## 📌 To-Do
✅ Improve UI with real-time updates  
✅ Optimize model performance  
🔜 Add more plant disease categories  
🔜 Deploy using Docker/Cloud services  

---

## 🏆 Contributing
Feel free to contribute by submitting issues or pull requests! 😊

---

## 📜 License
MIT License © 2025 viswalegend27

