
---

# 🌿 Plant Disease Guardian

**Plant Disease Guardian** is an AI-powered web application designed to detect plant diseases from leaf images using deep learning. Built with **TensorFlow**, **Streamlit**, and **Google's Gemini AI**, this tool provides instant predictions along with disease details, prevention tips, and treatment suggestions.

---

## 🚀 Features
- 📷 **Upload a Leaf Image**: Easily upload a plant leaf image for analysis.
- 🌱 **AI-Powered Disease Detection**: Uses a deep learning model to classify plant diseases.
- 🤖 **AI-Generated Disease Insights**: Fetches additional details using Google's Gemini AI.
- 📊 **Confidence Score**: Displays model confidence in the prediction.
- 📋 **Alternative Predictions**: Shows secondary predictions if confidence is low.
- 🌍 **User-Friendly UI**: Simple and mobile-friendly interface powered by **Streamlit**.

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/viswalegend27/plant-disease-guardian.git
cd plant-disease-guardian
```

### 2️⃣ Install Dependencies
Make sure you have **Python 3.8+** installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up API Keys (Optional for AI-generated insights)
Create a `.env` file in the project root and add your **Gemini API key**:
```
GEMINI_API_KEY=your_api_key_here
```

### 4️⃣ Run the Application
```bash
streamlit run app.py
```

---

## 📚 How It Works
1. **User Uploads an Image** - A plant leaf image is uploaded to the application.
2. **Image Preprocessing** - The image is resized and converted into an array for model input.
3. **Disease Classification** - A trained TensorFlow model predicts the disease.
4. **AI-Powered Explanation** - The Gemini AI provides additional disease details (if enabled).
5. **Results Displayed** - The app shows the prediction, confidence score, and treatment suggestions.

---

## 🏗️ Project Structure
```
📂 plant-disease-guardian
│── 📄 main.py           # Main Streamlit application
│── 📂 models            # Contains trained TensorFlow models
│── 📂 assets            # Images and UI assets
│── 📄 requirements.txt  # Dependencies list
│── 📄 .env.example      # Provide your Gemini API Key here.
│── 📄 README.md         # Project documentation
│── 📂 .streamlit        # Streamlit folder
        │── secrets.toml # You can provide you Gemini API Key here to. Must have create this folder to work with the code as it is or else alter the main.py      
```
---

## 📌 Supported Plant Diseases
| **Plant**     | **Disease**                          |
|---------------|--------------------------------------|
| Tomato       | Late Blight, Early Blight, Healthy  |
| Potato       | Late Blight, Early Blight, Healthy  |
| Pepper Bell  | Bacterial Spot, Healthy             |

---

## 🏆 Contributing
We welcome contributions! If you'd like to improve the project:

1. Fork the repository.
2. Create a new branch (`feature-xyz`).
3. Commit your changes (`git commit -m "Added feature xyz"`).
4. Push to your branch (`git push origin feature-xyz`).
5. Create a Pull Request 🎉

---

## 📜 License
This project is open-source and available under the **MIT License**.

---

## 💡 Acknowledgments
- **TensorFlow** for deep learning model support.
- **Streamlit** for UI development.
- **Google Gemini AI** for AI-powered insights.

---

💚 Happy Farming! 🌱  

---
And also this whole project would only be a dream if not i did not follow anything in the field
this playlist i have provide is my key to achieve success in my project.

https://www.linkedin.com/in/dhavalsays/ --> Thank you sir ! ❤️❤️
---
https://youtube.com/playlist?list=PLeo1K3hjS3ut49PskOfLnE6WUoOp_2lsD&si=lqOIkHbvMHcj_01m 

Channel name : codebasics

Do subscribe to this channel and share their work to others.
