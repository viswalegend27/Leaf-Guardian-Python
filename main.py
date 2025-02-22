import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Guardian",
    layout="wide",
    page_icon="🌿"
)

# Load the potato disease classification model with error handling
try:
    potato_disease_model = tf.keras.models.load_model("1.keras")
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    potato_disease_model = None

CLASS_NAMES = ['Pepperbell-Bacterial spot', 'Pepperbell-Healthy', 'Potato-Early blight', 'Potato-Lateblight', 
               'Potato-healthy', 'Tomato-Earlyblight', 'Tomato-Lateblight', 'Tomato-healthy']

# Highly detailed disease and prevention information (unchanged from previous response)
DISEASE_INFO = {
    'Pepperbell-Bacterial spot': {
        'description': """
        **Pepper Bell Bacterial Spot**  
        Pepper bell bacterial spot is a highly destructive bacterial disease caused primarily by *Xanthomonas euvesicatoria*, with related species like *Xanthomonas vesicatoria* also implicated. It severely impacts bell peppers and other pepper varieties, leading to significant economic losses due to reduced yield and quality. Detailed symptoms include:  
        - Initial small, water-soaked lesions on leaves that quickly develop into dark brown or black spots, often encircled by a characteristic yellowish halo, resembling a target pattern.  
        - As the disease progresses, lesions may merge, causing extensive necrosis, premature leaf drop, and diminished photosynthetic capacity, which weakens the plant and stunts growth.  
        - On fruits, sunken, scab-like, or raised lesions form, particularly near the calyx, rendering peppers unmarketable and susceptible to secondary infections. Stems and petioles may exhibit dark, elongated, water-soaked lesions that can girdle and kill branches.  
        - The pathogen thrives in warm, humid environments, with optimal temperatures of 75–85°F (24–29°C) and high relative humidity (above 85%). It spreads rapidly through rain splashes, irrigation water, contaminated tools, infected seeds, or insect vectors such as aphids and whiteflies. In regions with frequent rainfall or high humidity (e.g., tropical or subtropical climates), outbreaks can reduce yields by 30–50% or more if left unmanaged. The bacteria can survive in plant debris or soil for months, posing a persistent threat.
        """,
        'prevention': """
        **Comprehensive Prevention and Management Strategies for Pepper Bell Bacterial Spot:**  
        - **Seed Health Management**: Source certified disease-free seeds from reputable suppliers, as infected seeds are a primary source of introduction. Treat seeds with hot water (122°F/50°C for 25–30 minutes) to eliminate bacterial contamination, ensuring thorough drying afterward to prevent germination issues.  
        - **Crop Rotation and Diversification**: Implement a strict 2–3 year rotation with non-host crops such as cereals (e.g., wheat, corn), legumes (e.g., beans, peas), or brassicas to reduce soil-borne bacterial populations and disrupt the disease cycle. Avoid planting peppers or solanaceous crops (e.g., tomatoes, eggplants) in the same field consecutively.  
        - **Rigorous Sanitation Practices**: Remove and destroy all infected plant material, including leaves, stems, fruits, and roots, by burning or deep burial to prevent overwintering of the bacteria. Disinfect pruning tools, harvest equipment, and hands with a 10% bleach solution, 70% ethanol, or quaternary ammonium compounds after each use to prevent cross-contamination. Regularly clean greenhouse surfaces and irrigation systems to eliminate bacterial reservoirs.  
        - **Precision Irrigation Techniques**: Eliminate overhead irrigation, which promotes leaf wetness and bacterial spread, and adopt drip irrigation to deliver water directly to the root zone. Water early in the morning (before 9 AM) to allow foliage to dry quickly, reducing the duration of leaf surface humidity (ideally below 4 hours) that favors bacterial growth.  
        - **Optimal Plant Spacing and Canopy Management**: Space pepper plants 12–18 inches apart in rows, ensuring adequate air circulation to minimize humidity within the canopy. Prune lower leaves touching the soil and remove excessive vegetation to enhance light penetration and airflow, reducing microclimates conducive to bacterial proliferation. Control weeds to further improve ventilation and reduce competition.  
        - **Targeted Bactericide Applications**: Apply copper-based bactericides (e.g., copper hydroxide, copper oxychloride) preventively, starting before symptoms appear and continuing every 7–14 days during wet or humid periods. Rotate with other bactericides like mancozeb or streptomycin to prevent resistance development, following label instructions for timing, frequency, dosage, and safety precautions. Use integrated pest management (IPM) to monitor weather conditions and apply treatments only when necessary, minimizing environmental impact.  
        - **Selection of Resistant Cultivars**: Where available, select pepper varieties with partial resistance to bacterial spot, such as ‘Aristotle’ or ‘Revolution,’ though complete resistance is rare. Consult local agricultural extension services, seed catalogs, or research institutions for region-specific cultivars adapted to your climate and pathogen strains. Conduct small-scale trials to evaluate performance under local conditions.  
        - **Vigilant Monitoring and Early Intervention**: Inspect plants weekly or biweekly for early symptoms (water-soaked spots) using a magnifying glass if needed, focusing on lower leaves first. Remove and destroy infected leaves immediately, sealing them in plastic bags to prevent spore dispersal. Avoid working with plants during or after rain, as wet foliage increases bacterial transfer via tools or hands. Use digital tools or apps (e.g., plant disease trackers) to log observations and track disease progression.  
        - **Soil and Nutrient Optimization**: Maintain well-drained, fertile soil with balanced nutrient levels, emphasizing adequate potassium and calcium to enhance plant resilience against bacterial infections. Conduct soil tests annually to address deficiencies and avoid over-fertilization with nitrogen, which can promote lush, susceptible growth. Incorporate organic amendments like compost or well-rotted manure to improve soil structure, water retention, and beneficial microbial activity that suppresses pathogens.  
        - **Integrated Pest and Environmental Management**: Control insect vectors like aphids and whiteflies using biological controls (e.g., ladybugs, lacewings), sticky traps, or targeted insecticides (e.g., insecticidal soaps) to reduce bacterial transmission. Monitor weather forecasts for high-humidity or rainy periods, applying preventive measures proactively. Use raised beds or plastic mulch to improve drainage and reduce soil splash onto leaves, further limiting bacterial spread.
        """
    },
    'Pepperbell-Healthy': {
        'description': "No disease detected. Your pepper bell plant is healthy, vibrant, and showing no signs of infection or stress, with lush green foliage and robust growth!",
        'prevention': """
        **Maintaining Optimal Plant Health:**  
        - Ensure consistent, deep watering using drip irrigation at the base, applied early in the morning to minimize leaf wetness.  
        - Maintain 12–18 inch plant spacing and prune lower leaves to enhance airflow and light exposure, preventing potential stress.  
        - Apply organic mulch (e.g., straw or wood chips) and use balanced fertilizers (e.g., 10-10-10) to support vigorous growth without over-fertilization.  
        - Monitor plants biweekly for early signs of stress, pests, or diseases, adjusting care based on weather conditions or soil health.
        """
    },
    'Potato-Early blight': {
        'description': """
        **Potato Early Blight**  
        Potato early blight, caused by the fungal pathogen *Alternaria solani*, is a pervasive disease affecting potato crops worldwide, particularly in warm, humid regions. It targets leaves, stems, and tubers, leading to significant yield reductions if uncontrolled. Symptoms include:  
        - Small, dark brown spots with concentric rings (resembling a target) on older leaves, typically starting at the base of the plant due to proximity to soil-borne spores.  
        - These spots enlarge, causing yellowing, wilting, and defoliation, which impairs photosynthesis and weakens the plant, reducing tuber size and quality.  
        - On stems, dark, elongated lesions may form, while tubers develop shallow, sunken, brown areas under the skin, often visible after harvest or during storage, increasing susceptibility to rot.  
        - The fungus thrives in temperatures of 68–86°F (20–30°C) with high humidity (above 85%), spreading via wind, rain splashes, or infected plant debris. In poorly managed fields, early blight can reduce yields by 20–40%, especially during prolonged wet seasons.
        """,
        'prevention': """
        **Comprehensive Prevention and Management Strategies for Potato Early Blight:**  
        - **Extended Crop Rotation**: Rotate potatoes with non-host crops such as cereals (e.g., oats, barley), legumes (e.g., peas, beans), or brassicas for at least 2–3 years to deplete soil-borne fungal inoculum and break the disease cycle. Avoid planting solanaceous crops like tomatoes or eggplants in rotation.  
        - **Thorough Sanitation**: Remove and destroy all infected plant debris, including leaves, stems, and tubers, by burning or deep burial to prevent fungal survival in soil or residue. Clean and disinfect tools, equipment, and harvest machinery with a 10% bleach solution or 70% ethanol to prevent spore transfer. Regularly clear field borders of volunteer plants or weeds that may harbor the pathogen.  
        - **Precision Irrigation Techniques**: Avoid overhead irrigation, which promotes leaf wetness and fungal spore germination, and adopt drip irrigation to deliver water directly to the root zone. Water early in the morning (before 9 AM) to allow foliage to dry within 4–6 hours, minimizing the humid conditions favored by *Alternaria solani*.  
        - **Optimal Plant Spacing and Hilling**: Space potato plants 12–15 inches apart in rows to ensure adequate airflow, reducing canopy humidity. Hill soil around plants as they grow to cover lower stems, preventing soil splash onto leaves and reducing infection risk from soil-borne spores. Remove any lower leaves touching the ground to further limit exposure.  
        - **Strategic Fungicide Applications**: Apply protective fungicides like chlorothalonil, mancozeb, or copper-based products preventively, starting before symptoms appear and continuing every 7–14 days during humid or rainy periods. Use weather-based forecasting tools (e.g., TOM-CAST) to optimize spray timing, minimizing unnecessary applications while maximizing efficacy. Rotate fungicide classes to prevent resistance development, following label instructions for dosage and safety.  
        - **Selection of Resistant Cultivars**: Choose potato varieties with moderate resistance to early blight, such as ‘Kennebec,’ ‘Yukon Gold,’ or ‘Russet Burbank,’ if available, though resistance may vary by region or strain. Consult local agricultural extension services or seed suppliers for cultivars suited to your climate and soil conditions, and conduct small-scale field trials to assess performance.  
        - **Vigilant Monitoring and Early Intervention**: Inspect plants weekly or biweekly for early symptoms (small dark spots on lower leaves), using a hand lens if needed to detect subtle lesions. Remove and destroy infected leaves immediately, sealing them in plastic bags to prevent spore dispersal. Avoid working with plants during or after rain, as wet foliage increases spore spread via tools or hands. Use digital disease tracking apps or field notebooks to log observations and track disease progression.  
        - **Soil and Nutrient Optimization**: Maintain well-drained, fertile soil with balanced nutrient levels, emphasizing adequate potassium and magnesium to enhance plant resilience against fungal infections. Conduct annual soil tests to identify deficiencies and avoid excessive nitrogen, which can promote lush, susceptible growth. Incorporate organic amendments like compost or well-rotted manure to improve soil structure, water retention, and beneficial microbial activity that suppresses pathogens.  
        - **Integrated Environmental Management**: Monitor weather patterns for prolonged warm, humid periods, applying preventive measures proactively. Use plastic mulch or raised beds to improve drainage and reduce soil splash onto foliage, further limiting infection. Control volunteer potatoes or nearby solanaceous weeds, which can serve as alternate hosts for *Alternaria solani*.
        """
    },
    'Potato-Lateblight': {
        'description': """
        **Potato Late Blight**  
        Potato late blight, caused by the oomycete *Phytophthora infestans*, is one of the most devastating diseases of potatoes, historically responsible for the Irish Potato Famine. It affects leaves, stems, and tubers, causing rapid crop loss in cool, wet conditions. Symptoms include:  
        - Dark, water-soaked lesions on leaves that expand quickly, often with a white, fuzzy mold (sporangia) on the undersides, especially during humid nights.  
        - Stems develop dark, rotted areas that can girdle and kill branches, while tubers exhibit brown, mushy rot, typically under the skin, spreading internally and rendering them unmarketable.  
        - The disease spreads explosively in temperatures of 50–75°F (10–24°C) with high moisture, via wind, rain, or infected plant material, often devastating entire fields within days. In severe outbreaks, losses can exceed 70%, particularly in regions with frequent fog, dew, or rainfall.
        """,
        'prevention': """
        **Comprehensive Prevention and Management Strategies for Potato Late Blight:**  
        - **Extended Crop Rotation**: Rotate potatoes with non-host crops such as grains (e.g., wheat, barley) or legumes for 3–4 years to reduce soil-borne oomycete populations and eliminate overwintering inoculum. Avoid planting solanaceous crops like tomatoes or eggplants in rotation, as they can host *Phytophthora infestans*.  
        - **Strict Sanitation Protocols**: Remove and destroy all infected plants, including leaves, stems, and tubers, immediately upon detection, by burning or deep burial to prevent spore dispersal. Disinfect tools, equipment, and harvest machinery with a 10% bleach solution, 70% ethanol, or quaternary ammonium compounds to avoid cross-contamination. Clear field borders of volunteer potatoes or weeds that may harbor the pathogen.  
        - **Controlled Irrigation and Drainage**: Eliminate overhead irrigation, which promotes leaf wetness and oomycete sporulation, and use drip irrigation to deliver water directly to the root zone. Ensure fields have excellent drainage to prevent standing water, using raised beds or trenches if necessary. Water early in the morning to allow foliage to dry quickly, reducing the humid conditions favored by *Phytophthora infestans*.  
        - **Proactive Fungicide Applications**: Apply protective fungicides like chlorothalonil, metalaxyl, or mancozeb preventively, starting before symptoms appear and continuing every 5–10 days during cool, wet weather. Use weather-based forecasting models (e.g., BlightCast, SimCast) to optimize spray timing, targeting periods of high humidity, rainfall, or fog. Rotate fungicide modes of action to prevent resistance, following label instructions for dosage, safety, and environmental considerations.  
        - **Selection of Resistant Cultivars**: Plant potato varieties with strong resistance to late blight, such as ‘Kennebec,’ ‘Yukon Gold,’ or ‘Sarpo Mira,’ though resistance may vary by *Phytophthora* strain. Consult local agricultural extension services or seed suppliers for cultivars adapted to your region’s climate, soil, and prevalent pathogen strains, and conduct field trials to evaluate performance.  
        - **Intensive Monitoring and Rapid Response**: Inspect plants daily during wet or cool weather for early symptoms (water-soaked lesions), using a magnifying glass if needed to detect subtle signs. Remove and destroy infected plants immediately, sealing them in plastic bags to prevent spore spread. Avoid working with plants during or after rain, as wet foliage increases oomycete transmission via tools or hands. Use digital tools or apps (e.g., disease trackers) to log observations, track weather conditions, and trigger interventions.  
        - **Soil and Environmental Optimization**: Maintain well-drained, fertile soil with balanced nutrients, emphasizing potassium and phosphorus to enhance plant resilience. Conduct annual soil tests to address deficiencies and avoid excessive nitrogen, which can promote susceptible growth. Use raised beds, plastic mulch, or contour plowing to improve drainage and reduce soil splash onto foliage, minimizing infection risk. Plant in well-ventilated areas, avoiding low-lying or shaded fields where humidity lingers.  
        - **Integrated Pest and Environmental Management**: Monitor weather forecasts for prolonged cool, wet periods, applying preventive measures proactively. Control volunteer potatoes or nearby solanaceous weeds, which can serve as alternate hosts for *Phytophthora infestans*. Use biological controls or cultural practices (e.g., row covers) to reduce humidity and spore dispersal, complementing chemical strategies for a holistic approach.
        """
    },
    'Potato-healthy': {
        'description': "No disease detected. Your potato plant is healthy, robust, and exhibiting lush, green foliage with no signs of infection, stress, or pest damage!",
        'prevention': """
        **Maintaining Optimal Plant Health:**  
        - Provide consistent, deep watering using drip irrigation at the base, applied early in the morning to minimize leaf wetness.  
        - Space plants 12–15 inches apart and hill soil around stems to enhance airflow and prevent soil splash, promoting vigorous growth.  
        - Apply organic mulch (e.g., straw) and use balanced fertilizers (e.g., 10-10-10) to support nutrient uptake without over-fertilization.  
        - Monitor plants biweekly for early signs of stress, pests, or diseases, adjusting care based on weather, soil conditions, or growth stage.
        """
    },
    'Tomato-Earlyblight': {
        'description': """
        **Tomato Early Blight**  
        Tomato early blight, caused by the fungal pathogen *Alternaria solani*, is a common and damaging disease affecting tomato plants, particularly in warm, humid regions. It primarily targets older leaves, stems, and fruits, leading to reduced yield and quality. Symptoms include:  
        - Small, dark brown spots with concentric rings (target-like) on older leaves, typically starting at the base of the plant due to proximity to soil-borne spores.  
        - These spots enlarge, causing yellowing, wilting, and defoliation, which impairs photosynthesis, weakens the plant, and reduces fruit development.  
        - On stems, dark, elongated lesions may form, while fruits develop sunken, leathery, brown spots, especially near the stem end, lowering marketability.  
        - The fungus thrives in temperatures of 68–86°F (20–30°C) with high humidity (above 85%), spreading via wind, rain splashes, or infected plant debris. In poorly managed fields, early blight can reduce tomato yields by 20–40%, particularly during prolonged wet seasons.
        """,
        'prevention': """
        **Comprehensive Prevention and Management Strategies for Tomato Early Blight:**  
        - **Extended Crop Rotation**: Rotate tomatoes with non-host crops such as cereals (e.g., oats, barley), legumes (e.g., peas, beans), or brassicas for at least 2–3 years to deplete soil-borne fungal inoculum and break the disease cycle. Avoid planting solanaceous crops like potatoes or eggplants in rotation.  
        - **Thorough Sanitation**: Remove and destroy all infected plant debris, including leaves, stems, and fruits, by burning or deep burial to prevent fungal survival in soil or residue. Clean and disinfect tools, equipment, and harvest machinery with a 10% bleach solution or 70% ethanol to prevent spore transfer. Regularly clear field borders of volunteer plants or weeds that may harbor the pathogen.  
        - **Precision Irrigation Techniques**: Avoid overhead irrigation, which promotes leaf wetness and fungal spore germination, and adopt drip irrigation to deliver water directly to the root zone. Water early in the morning (before 9 AM) to allow foliage to dry within 4–6 hours, minimizing the humid conditions favored by *Alternaria solani*.  
        - **Optimal Plant Spacing and Training**: Space tomato plants 18–24 inches apart in rows to ensure adequate airflow, reducing canopy humidity. Use stakes, cages, or trellises to support plants, pruning lower leaves touching the soil and removing excessive foliage to enhance light penetration and ventilation, limiting disease pressure.  
        - **Strategic Fungicide Applications**: Apply protective fungicides like chlorothalonil, mancozeb, or copper-based products preventively, starting before symptoms appear and continuing every 7–14 days during humid or rainy periods. Use weather-based forecasting tools (e.g., TOM-CAST) to optimize spray timing, minimizing unnecessary applications while maximizing efficacy. Rotate fungicide classes to prevent resistance development, following label instructions for dosage and safety.  
        - **Selection of Resistant Cultivars**: Choose tomato varieties with moderate resistance to early blight, such as ‘Mountain Magic,’ ‘Jasper,’ or ‘Defiant,’ if available, though resistance may vary by region or strain. Consult local agricultural extension services or seed suppliers for cultivars suited to your climate and soil conditions, and conduct small-scale field trials to assess performance.  
        - **Vigilant Monitoring and Early Intervention**: Inspect plants weekly or biweekly for early symptoms (small dark spots on lower leaves), using a hand lens if needed to detect subtle lesions. Remove and destroy infected leaves immediately, sealing them in plastic bags to prevent spore dispersal. Avoid working with plants during or after rain, as wet foliage increases spore spread via tools or hands. Use digital disease tracking apps or field notebooks to log observations and track disease progression.  
        - **Soil and Nutrient Optimization**: Maintain well-drained, fertile soil with balanced nutrient levels, emphasizing adequate potassium and magnesium to enhance plant resilience against fungal infections. Conduct annual soil tests to identify deficiencies and avoid excessive nitrogen, which can promote lush, susceptible growth. Incorporate organic amendments like compost or well-rotted manure to improve soil structure, water retention, and beneficial microbial activity that suppresses pathogens.  
        - **Integrated Environmental Management**: Monitor weather patterns for prolonged warm, humid periods, applying preventive measures proactively. Use plastic mulch or raised beds to improve drainage and reduce soil splash onto foliage, further limiting infection. Control volunteer tomatoes or nearby solanaceous weeds, which can serve as alternate hosts for *Alternaria solani*.
        """
    },
    'Tomato-Lateblight': {
        'description': """
        **Tomato Late Blight**  
        Tomato late blight, caused by the oomycete *Phytophthora infestans*, is a highly destructive disease, closely related to potato late blight, and capable of devastating tomato crops in cool, wet conditions. It affects leaves, stems, and fruits, leading to rapid crop loss if unchecked. Symptoms include:  
        - Dark, water-soaked lesions on leaves that expand rapidly, often with a white, fuzzy mold (sporangia) on the undersides, especially during humid nights.  
        - Stems develop dark, rotted areas that can girdle and kill branches, while fruits exhibit large, brown, mushy rot, particularly near the stem end, rendering them unmarketable.  
        - The disease spreads explosively in temperatures of 50–75°F (10–24°C) with high moisture, via wind, rain, or infected plant material, often destroying entire fields within days. In severe outbreaks, losses can exceed 70%, especially in regions with frequent fog, dew, or rainfall.
        """,
        'prevention': """
        **Comprehensive Prevention and Management Strategies for Tomato Late Blight:**  
        - **Extended Crop Rotation**: Rotate tomatoes with non-host crops such as grains (e.g., wheat, barley) or legumes for 3–4 years to reduce soil-borne oomycete populations and eliminate overwintering inoculum. Avoid planting solanaceous crops like potatoes or eggplants in rotation, as they can host *Phytophthora infestans*.  
        - **Strict Sanitation Protocols**: Remove and destroy all infected plants, including leaves, stems, and fruits, immediately upon detection, by burning or deep burial to prevent spore dispersal. Disinfect tools, equipment, and harvest machinery with a 10% bleach solution, 70% ethanol, or quaternary ammonium compounds to avoid cross-contamination. Clear field borders of volunteer tomatoes or weeds that may harbor the pathogen.  
        - **Controlled Irrigation and Drainage**: Eliminate overhead irrigation, which promotes leaf wetness and oomycete sporulation, and use drip irrigation to deliver water directly to the root zone. Ensure fields have excellent drainage to prevent standing water, using raised beds or trenches if necessary. Water early in the morning to allow foliage to dry quickly, reducing the humid conditions favored by *Phytophthora infestans*.  
        - **Proactive Fungicide Applications**: Apply protective fungicides like chlorothalonil, metalaxyl, or mancozeb preventively, starting before symptoms appear and continuing every 5–10 days during cool, wet weather. Use weather-based forecasting models (e.g., BlightCast, SimCast) to optimize spray timing, targeting periods of high humidity, rainfall, or fog. Rotate fungicide modes of action to prevent resistance, following label instructions for dosage, safety, and environmental considerations.  
        - **Selection of Resistant Cultivars**: Plant tomato varieties with strong resistance to late blight, such as ‘Defiant PhR,’ ‘Mountain Merit,’ or ‘Ferne,’ though resistance may vary by *Phytophthora* strain. Consult local agricultural extension services or seed suppliers for cultivars adapted to your region’s climate, soil, and prevalent pathogen strains, and conduct field trials to evaluate performance.  
        - **Intensive Monitoring and Rapid Response**: Inspect plants daily during wet or cool weather for early symptoms (water-soaked lesions), using a magnifying glass if needed to detect subtle signs. Remove and destroy infected plants immediately, sealing them in plastic bags to prevent spore spread. Avoid working with plants during or after rain, as wet foliage increases oomycete transmission via tools or hands. Use digital tools or apps (e.g., disease trackers) to log observations, track weather conditions, and trigger interventions.  
        - **Soil and Environmental Optimization**: Maintain well-drained, fertile soil with balanced nutrients, emphasizing potassium and phosphorus to enhance plant resilience. Conduct annual soil tests to address deficiencies and avoid excessive nitrogen, which can promote susceptible growth. Use raised beds, plastic mulch, or contour plowing to improve drainage and reduce soil splash onto foliage, minimizing infection risk. Plant in well-ventilated areas, avoiding low-lying or shaded fields where humidity lingers.  
        - **Integrated Pest and Environmental Management**: Monitor weather forecasts for prolonged cool, wet periods, applying preventive measures proactively. Control volunteer tomatoes or nearby solanaceous weeds, which can serve as alternate hosts for *Phytophthora infestans*. Use stakes, cages, or trellises to improve air circulation and reduce canopy humidity, complementing chemical and cultural strategies for a holistic approach.
        """
    },
    'Tomato-healthy': {
        'description': "No disease detected. Your tomato plant is healthy, vigorous, and displaying lush, green foliage with no signs of infection, stress, or pest damage!",
        'prevention': """
        **Maintaining Optimal Plant Health:**  
        - Provide consistent, deep watering using drip irrigation at the base, applied early in the morning to minimize leaf wetness.  
        - Space plants 18–24 inches apart and use stakes, cages, or trellises to enhance airflow and support growth, preventing soil contact.  
        - Apply organic mulch (e.g., straw) and use balanced fertilizers (e.g., 10-10-10) to support nutrient uptake without over-fertilization.  
        - Monitor plants biweekly for early signs of stress, pests, or diseases, adjusting care based on weather, soil conditions, or growth stage.
        """
    }
}

# Add content to your Streamlit app
st.title('🌿 Plant Disease Guardian')

# Upload image
uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and potato_disease_model is not None:
    try:
        
        img = Image.open(uploaded_file)
        img = img.resize((256, 256), Image.Resampling.LANCZOS)  # Use LANCZOS for better image quality
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        
        predictions = potato_disease_model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        print(predicted_class)

        
        col1, col2 = st.columns([1, 1])

        with col1:
            
            st.markdown("<div style='display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%;'>", unsafe_allow_html=True)
            st.markdown(f"<div class='prediction'>Predicted Class: {predicted_class}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='confidence'>Confidence: {confidence:.2f}</div>", unsafe_allow_html=True)
            st.image(img, caption='Uploaded Leaf Image', width=400)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            
            if predicted_class in DISEASE_INFO:
                st.subheader("📋 Disease Details")
                st.markdown(DISEASE_INFO[predicted_class]['description'], unsafe_allow_html=True)

                st.subheader("🛡️ Prevention & Management")
                st.markdown(DISEASE_INFO[predicted_class]['prevention'], unsafe_allow_html=True)
            else:
                st.write("Information for this disease is not available yet. Please check back later or contact an expert!")
    except Exception as e:
        st.error(f"Error processing image or prediction: {e}")

st.markdown("""
    <style>
    /* Dark background for the entire app */
    body {
        background-color: #1a1a1a;
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background: transparent;
    }

    /* Title styling (green with leaf emoji) */
    h1 {
        color: #2ecc71; /* Bright green */
        font-family: 'Georgia', serif;
        text-align: center;
        padding: 15px;
        background-color: #2d2d2d;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }

    /* Prediction and confidence styling */
    .prediction {
        font-size: 24px;
        font-weight: bold;
        color: #2ecc71; /* Green for prediction */
        margin-bottom: 10px;
        text-align: center;
    }
    .confidence {
        font-size: 18px;
        color: #e74c3c; /* Red for confidence */
        text-align: center;
        margin-bottom: 20px;
    }

    /* Subheader styling */
    h2 {
        color: #2ecc71;
        font-family: 'Arial', sans-serif;
        border-bottom: 2px solid #2ecc71;
        padding-bottom: 5px;
    }

    /* Text formatting for disease and prevention */
    .stMarkdown {
        font-size: 16px;
        line-height: 1.6;
        color: #e0e0e0;
        background: rgba(45, 45, 45, 0.8);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .stMarkdown strong {
        color: #2ecc71;
    }

    /* Uploader button styling (dark with green hover) */
    .stFileUploader {
        margin: 20px auto;
        text-align: center;
    }
    .stFileUploader label {
        display: none;
    }
    .stFileUploader > div > button {
        background-color: #34495e; /* Dark gray */
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .stFileUploader > div > button:hover {
        background-color: #2ecc71; /* Green on hover */
        box-shadow: 0 4px 8px rgba(46, 204, 113, 0.5);
    }

    /* Image and caption styling */
    .stImage {
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
    }
    .stImage > figcaption {
        color: #e0e0e0;
        font-size: 14px;
        text-align: center;
        background: rgba(45, 45, 45, 0.8);
        padding: 5px;
        border-radius: 0 0 8px 8px;
    }

    /* Column spacing */
    .stColumns > div {
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)