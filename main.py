import re  
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
import base64
from io import BytesIO

load_dotenv()

# Light theme configuration
st.set_page_config(
    page_title="Plant Guardian", 
    layout="wide", 
    page_icon="🌿",
    initial_sidebar_state="collapsed"
)

# Disease information dictionary (fallback if API is unavailable)
DISEASE_INFO = {
    'bacterial_spot': {
        'description': "Bacterial spot causes dark, raised lesions on leaves, stems, and fruits. Spots may appear water-soaked initially, then turn brown to black with a yellow halo. In severe cases, leaves may drop prematurely.",
        'prevention': "Use disease-free seeds and transplants, practice crop rotation for 2-3 years, apply copper-based fungicides early in the season, avoid overhead irrigation, maintain proper plant spacing for air circulation."
    },
    'early_blight': {
        'description': "Early blight causes brown spots with concentric rings (target-like patterns) on lower, older leaves first. Infected leaves turn yellow, then brown and may drop. Dark lesions may also appear on stems and fruits.",
        'prevention': "Remove infected leaves immediately, practice crop rotation for 3-4 years, improve air circulation between plants, apply fungicides preventatively before disease appears, mulch around plants to prevent soil splashing."
    },
    'late_blight': {
        'description': "Late blight causes water-soaked lesions that quickly turn brown with white fuzzy growth (pathogen's spores) on undersides. It spreads rapidly in cool, wet conditions and can destroy entire plants within days.",
        'prevention': "Remove and destroy infected plants, avoid overhead watering, apply fungicides before infection occurs, plant resistant varieties, increase plant spacing, avoid working with plants when wet."
    },
    'healthy': {
        'description': "This plant appears healthy! The leaves show good coloration without spots, lesions, discoloration, or abnormal growth patterns. Healthy plants typically display vibrant green leaves with proper structure.",
        'prevention': "Continue regular monitoring, proper watering (at soil level), balanced fertilization, adequate sunlight exposure, and good air circulation. Remove any yellowing leaves promptly and watch for early signs of pest activity."
    }
}

# Detailed blight comparison chart for reference
BLIGHT_COMPARISON = {
    'early_blight': {
        'pathogen': "Alternaria solani (fungus)",
        'key_visual': "Dark brown to black concentric rings (target-like pattern)",
        'location': "Starts on older/lower leaves first, then progresses upward",
        'pattern': "Angular lesions often bounded by leaf veins",
        'growth': "NO white fuzzy growth on leaf undersides",
        'halo': "Distinct yellow halo around lesions",
        'texture': "Spots may appear dry and papery",
        'progression': "Slow to moderate progression",
        'conditions': "Favored by warm (75-85°F), humid conditions with alternating wet/dry periods"
    },
    'late_blight': {
        'pathogen': "Phytophthora infestans (oomycete)",
        'key_visual': "Irregular pale/dark green water-soaked patches that turn brown/black",
        'location': "Can start anywhere on plant, affects all leaf ages equally",
        'pattern': "Lesions often begin at leaf margins and tips, with NO concentric rings",
        'growth': "White fuzzy growth on leaf undersides in humid conditions",
        'halo': "May have light green to pale yellow border (not a distinct halo)",
        'texture': "Patches appear wet, greasy or water-soaked",
        'progression': "Rapid progression that can destroy plants within days",
        'conditions': "Favored by cool (60-70°F) temperatures and wet, humid conditions"
    }
}

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY"))

if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        st.sidebar.success("✅ Gemini API connected")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Gemini AI error: {e}. Using local data.")
        model = None
else:
    st.sidebar.warning("⚠️ No Gemini API Key found! Using local disease information.")
    model = None

# Load TensorFlow model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("1.keras")

model_path = "1.keras"
disease_model = None
if os.path.exists(model_path):
    disease_model = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
else:
    st.sidebar.error("⚠️ Model file not found! Please upload a valid model.")

# Class names from the model
CLASS_NAMES = [
    'pepper_bell_bacterial_spot', 'pepper_bell_healthy', 'potato_early_blight',
    'potato_late_blight', 'potato_healthy', 'tomato_early_blight', 'tomato_late_blight', 'tomato_healthy'
]

def extract_disease_only(predicted_class):
    """Extract only the disease part from the prediction, ignoring plant type"""
    parts = predicted_class.lower().split('_')
    
    if 'healthy' in parts:
        return 'healthy', True
    
    # Remove plant name and return only disease
    for disease in ['bacterial_spot', 'early_blight', 'late_blight']:
        if disease in predicted_class:
            return disease, False
    
    return "unknown", False

def encode_image_to_base64(image):
    """Convert PIL image to base64 string for API compatibility"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_image_with_gemini(image, tf_prediction=None, confidence=None, detailed_report=False):
    """
    Enhanced multi-stage Gemini API analysis for plant disease detection with significantly 
    improved accuracy, feature extraction, and contextual awareness.
    
    Args:
        image: Image to analyze (PIL Image or compatible format)
        tf_prediction: Optional TensorFlow model prediction to augment analysis
        confidence: Confidence score of TF prediction if available
        detailed_report: Whether to return detailed analysis report in addition to classification
        
    Returns:
        tuple: (disease_classification, is_healthy, detailed_analysis_dict if detailed_report=True)
    """
    if not model:
        return "unknown", False, {} if detailed_report else "unknown", False
    
    try:
        # Standardize image format for API
        if isinstance(image, Image.Image):
            img_bytes = BytesIO()
            image.save(img_bytes, format='JPEG')
            img_bytes = img_bytes.getvalue()
        
        # Create a tracking dictionary to collect evidence throughout multi-stage analysis
        analysis_evidence = {
            'early_blight': {'markers': [], 'confidence': 0.0, 'counter_evidence': []},
            'late_blight': {'markers': [], 'confidence': 0.0, 'counter_evidence': []},
            'bacterial_spot': {'markers': [], 'confidence': 0.0, 'counter_evidence': []},
            'healthy': {'markers': [], 'confidence': 0.0, 'counter_evidence': []},
            'stage_results': []
        }
        
        # Stage 1: Initial comprehensive analysis with enhanced visual marker detection
        stage1_prompt = """
        I need you to analyze this plant leaf image with extreme precision for disease identification.
        
        ANALYZE FOR THESE SPECIFIC VISUAL MARKERS:
        
        EARLY BLIGHT (Alternaria solani):
        1. PRIMARY: Bull's-eye/target-like CONCENTRIC RINGS pattern in lesions
        2. Lesions with dry, papery texture
        3. Yellow chlorotic halos around necrotic areas
        4. Angular lesions bounded by leaf veins
        5. Older/lower leaves affected first
        6. Lesions may coalesce but retain target-pattern identity
        7. Brown to black necrotic tissue in center of lesions
        8. Spots might show cracks as they age
        9. NO water-soaking appearance
        10. Absence of white fungal growth on undersides
        
        LATE BLIGHT (Phytophthora infestans):
        1. PRIMARY: Water-soaked/greasy-looking irregular patches
        2. WHITE FUZZY GROWTH on leaf undersides in humid conditions
        3. Pale green to brown lesions with NO concentric patterns
        4. Lesions often begin at leaf tips/margins
        5. Affects all leaf ages (not just older leaves)
        6. Rapid spreading across entire leaflets
        7. Dark brown/black necrosis develops
        8. Lesions may appear translucent when held to light
        9. Wet appearance even in dry conditions
        10. Irregular, non-angular lesion boundaries
        
        BACTERIAL SPOT (Xanthomonas):
        1. Small (1-3mm), dark, water-soaked circular spots
        2. Pronounced yellow halos around lesions 
        3. Spots densely clustered together
        4. Shot-hole appearance as centers fall out
        5. Spots visible on both leaf surfaces
        6. Lesions remain small and don't coalesce like blights
        7. No fuzzy growth on leaf undersides
        8. No concentric ring patterns
        9. Often affects fruit and stems too
        10. Shiny appearance of spots in early stages
        
        HEALTHY LEAF:
        1. Uniform green coloration without lesions or spots
        2. No discoloration, chlorosis or necrosis
        3. No visible fungal/bacterial growth
        4. Regular leaf shape without distortions
        5. No abnormal texture changes
        
        THOROUGHLY EXAMINE MULTIPLE AREAS of the leaf before concluding.
        
        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        
        OBSERVED FEATURES:
        * [List ALL detected features, BOTH presence AND absence of key markers]
        
        CLASSIFICATION: [early_blight/late_blight/bacterial_spot/healthy]
        
        CONFIDENCE: [high/medium/low]
        
        EXPLANATION: [Brief justification focusing on primary diagnostic features observed]
        """
        
        # Include TF prediction if available
        if tf_prediction and confidence:
            stage1_prompt += f"""
            
            NOTE: A machine learning model predicts {tf_prediction.replace('_', ' ')} with {confidence:.2f} confidence.
            Use this information as a reference point only - your analysis should be based ENTIRELY on visual evidence.
            If you see clear evidence contradicting the model, explain specifically what visual markers conflict.
            """
        
        # Make stage 1 API call
        if isinstance(image, Image.Image):
            response = model.generate_content([stage1_prompt, {"mime_type": "image/jpeg", "data": img_bytes}])
        else:
            response = model.generate_content([stage1_prompt, image])
        
        stage1_text = response.text.lower()
        analysis_evidence['stage_results'].append({'stage': 1, 'response': stage1_text})
        
        # Extract classification and confidence from structured response
        classification = extract_classification(stage1_text)
        confidence_level = extract_confidence(stage1_text)
        
        # Update evidence collection with features from stage 1
        update_evidence_dictionary(analysis_evidence, stage1_text, classification, confidence_level)
        
        # Always proceed to stage 2 for more robust analysis
        # Stage 2: High-resolution differential analysis with improved markers
        stage2_prompt = """
        I need you to perform a highly detailed differential diagnosis focusing specifically on distinguishing disease features in this plant leaf.
        
        PERFORM THIS SYSTEMATIC DIFFERENTIAL ANALYSIS:
        
        1. EXAMINE PRIMARY DIAGNOSTIC MARKERS:
           a) CONCENTRIC RINGS/BULL'S-EYE PATTERN:
              - Present? [Yes/No/Partial/Unclear]
              - Description: [Pattern clarity, size, distribution]
              - Coverage: [Percentage of lesions showing this pattern]
           
           b) WATER-SOAKED APPEARANCE:
              - Present? [Yes/No/Partial/Unclear]
              - Description: [Extent, location, intensity]
              - Texture: [Is it actually wet-looking or just dark?]
           
           c) WHITE FUZZY GROWTH ON UNDERSIDES:
              - Present? [Yes/No/Partial/Unclear]
              - Description: [Coverage, density, location]
              - Alternative explanation: [Could this be misidentified debris/reflection?]
           
           d) SMALL CIRCULAR SPOTS WITH YELLOW HALOS:
              - Present? [Yes/No/Partial/Unclear]
              - Description: [Size, distribution, halo intensity]
              - Size range: [Estimate in mm if possible]
        
        2. ASSESS LESION CHARACTERISTICS:
           a) TEXTURE: [Dry/papery vs. Wet/greasy vs. Other]
           b) PATTERN: [Angular/vein-bounded vs. Irregular vs. Circular]
           c) DISTRIBUTION: [Older leaves vs. All leaf ages vs. Clustered]
           d) COLORATION: [Brown with yellow halos vs. Dark water-soaked vs. Other]
           e) LESION BOUNDARIES: [Angular/sharp vs. Irregular/diffuse]
        
        3. DETECT MUTUALLY EXCLUSIVE FEATURES:
           a) Angular lesions bounded by veins? [Yes/No/Partial]
           b) Irregular water-soaked patches? [Yes/No/Partial]
           c) Uniformly small circular spots? [Yes/No/Partial]
           d) Concentric ring patterns? [Yes/No/Partial]
           e) White growth on underside? [Yes/No/Partial/Not visible]
        
        4. ESTIMATE SEVERITY AND PROGRESSION:
           a) Coverage percentage: [Approximate % of leaf affected]
           b) Stage of progression: [Early/Middle/Advanced]
        
        RESPOND WITH EXACT FORMAT:
        
        PRIMARY MARKERS:
        * Concentric rings: [Present/Absent/Partial] - [Description]
        * Water-soaked patches: [Present/Absent/Partial] - [Description]
        * White fuzzy growth: [Present/Absent/Partial] - [Description]
        * Small circular spots with halos: [Present/Absent/Partial] - [Description]
        
        LESION PROFILE:
        * Texture: [Description]
        * Pattern: [Description]
        * Distribution: [Description]
        * Coloration: [Description]
        * Boundaries: [Description]
        
        MUTUALLY EXCLUSIVE FEATURES:
        * [List which mutually exclusive features are present/absent]
        
        DISEASE PROBABILITY RANKING:
        1. [Disease name] - [High/Medium/Low] - [Key evidence]
        2. [Disease name] - [High/Medium/Low] - [Key evidence]
        3. [Disease name] - [High/Medium/Low] - [Key evidence]
        
        FINAL DIAGNOSIS: [early_blight/late_blight/bacterial_spot/healthy]
        CONFIDENCE: [high/medium/low]
        RULING OUT: [Specific reasons for ruling out other conditions]
        """
        
        # Add conditional focus based on stage 1 results
        if classification in ["early_blight", "late_blight"]:
            stage2_prompt += """
            
            CRITICAL BLIGHT DIFFERENTIATION:
            * Look specifically for water-soaking AND white growth (confirming late blight)
            * VS concentric rings AND dry texture (confirming early blight)
            * These are mutually exclusive features - identify which set is present
            * Examine lesion boundaries: angular (early blight) vs. irregular (late blight)
            * Check lesion texture: papery/dry (early blight) vs. greasy/wet (late blight)
            """
        elif classification == "bacterial_spot":
            stage2_prompt += """
            
            CRITICAL BACTERIAL SPOT VERIFICATION:
            * Confirm uniform size of lesions (1-3mm)
            * Verify bright yellow halos around dark centers
            * Check for shot-hole appearance in older lesions
            * Confirm absence of both concentric rings AND water-soaked spreading patches
            * Verify spots are visible similarly on both leaf surfaces
            """
        
        # Stage 2 API call with high-resolution image examination instructions
        stage2_prompt += """
        
        EXAMINATION TECHNIQUE:
        * Zoom in to examine lesion details closely
        * Check multiple areas of the leaf, not just the most obvious lesions
        * Compare different lesions for consistency of features
        * Look specifically at lesion edges and boundaries
        * Examine transitions between healthy and diseased tissue
        """
        
        if isinstance(image, Image.Image):
            stage2_response = model.generate_content([stage2_prompt, {"mime_type": "image/jpeg", "data": img_bytes}])
        else:
            stage2_response = model.generate_content([stage2_prompt, image])
        
        stage2_text = stage2_response.text.lower()
        analysis_evidence['stage_results'].append({'stage': 2, 'response': stage2_text})
        
        # Extract stage 2 classification and confidence
        stage2_classification = extract_classification(stage2_text)
        stage2_confidence = extract_confidence(stage2_text)
        
        # Extract counter-evidence against each diagnosis
        extract_counter_evidence(analysis_evidence, stage2_text)
        
        # Update evidence with stage 2 findings
        update_evidence_dictionary(analysis_evidence, stage2_text, stage2_classification, stage2_confidence)
        
        # Always perform stage 3 for improved accuracy
        # Stage 3: Final high-specificity analysis with enhanced disease marker detection
        # Determine competing diagnoses for focused analysis
        competing_diagnoses = []
        
        # Always include stage 1 and stage 2 classifications
        if classification not in competing_diagnoses and classification != "unknown" and classification != "healthy":
            competing_diagnoses.append(classification)
        if stage2_classification not in competing_diagnoses and stage2_classification != "unknown" and stage2_classification != "healthy":
            competing_diagnoses.append(stage2_classification)
        
        # Include TF prediction if available and relevant
        if tf_prediction and tf_prediction not in competing_diagnoses and tf_prediction != "healthy":
            competing_diagnoses.append(tf_prediction)
        
        # Fallback if no clear competitors
        if not competing_diagnoses:
            competing_diagnoses = ["early_blight", "late_blight", "bacterial_spot"]
        
        # Create highly targeted prompt with enhanced differential diagnostics
        stage3_prompt = f"""
        FINAL DECISIVE ANALYSIS - perform a highly specialized examination focused exclusively on definitive diagnostic markers.
        
        I need you to distinguish between: {', '.join([d.replace('_', ' ') for d in competing_diagnoses])}
        
        EXAMINE THESE ABSOLUTELY DEFINITIVE DIAGNOSTIC FEATURES IN DETAIL:
        
        FOR EARLY BLIGHT:
        * DEFINITIVE: Bull's-eye/target CONCENTRIC RING pattern in lesions (MOST RELIABLE FEATURE)
        * DEFINITIVE: DRY, papery texture to lesions
        * DEFINITIVE: Lesions bounded by leaf veins (angular)
        * DEFINITIVE: NO water-soaked appearance whatsoever
        * DEFINITIVE: Brown to black necrotic tissue with yellow chlorotic halo
        * DEFINITIVE: Any coalescence of lesions still shows individual concentric patterns
        * COUNTER-INDICATOR: Presence of white fuzzy growth on underside
        * COUNTER-INDICATOR: Wet/greasy lesion appearance
        
        FOR LATE BLIGHT:
        * DEFINITIVE: Water-soaked/greasy appearance to lesions (MOST RELIABLE FEATURE)
        * DEFINITIVE: WHITE FUZZY growth on undersides (if visible in humid conditions)
        * DEFINITIVE: NO concentric ring patterns whatsoever
        * DEFINITIVE: Lesions often starting at leaf margins/tips
        * DEFINITIVE: Irregular, non-angular lesion boundaries not limited by veins
        * DEFINITIVE: Pale green to dark brown/black color progression
        * COUNTER-INDICATOR: Presence of any concentric ring patterns
        * COUNTER-INDICATOR: Dry/papery lesion texture
        
        FOR BACTERIAL SPOT:
        * DEFINITIVE: Small (1-3mm), uniform circular spots (MOST RELIABLE FEATURE)
        * DEFINITIVE: Bright yellow halos around small dark centers
        * DEFINITIVE: Clustered appearance with potential shot-holes
        * DEFINITIVE: Spots same size/appearance on both leaf surfaces
        * DEFINITIVE: Small lesions that remain distinct (don't merge into larger patches)
        * DEFINITIVE: Shiny appearance in early stages
        * COUNTER-INDICATOR: Any concentric ring patterns
        * COUNTER-INDICATOR: Large spreading water-soaked patches
        
        PERFORM THIS CRITICAL DIAGNOSTIC TEST:
        For each disease, assign a numerical score (0-10) for presence of EACH definitive feature:
        - 0: Feature definitely absent
        - 1-3: Feature slightly/questionably present
        - 4-6: Feature moderately present
        - 7-10: Feature strongly/definitely present
        
        THEN assign a numerical score (0-10) for presence of EACH counter-indicator:
        - 0: Counter-indicator definitely present (strong evidence AGAINST)
        - 1-3: Counter-indicator moderately present
        - 4-6: Counter-indicator slightly present
        - 7-10: Counter-indicator definitely absent
        
        SHOW YOUR SCORING WORK:
        
        EARLY BLIGHT SCORE:
        * Concentric rings: [0-10] - [reasoning]
        * Dry/papery texture: [0-10] - [reasoning]
        * Angular lesions: [0-10] - [reasoning]
        * No water-soaking: [0-10] - [reasoning]
        * Yellow chlorotic halo: [0-10] - [reasoning]
        * No white growth: [0-10] - [reasoning]
        TOTAL: [Sum/60] = [percentage]%
        
        LATE BLIGHT SCORE:
        * Water-soaked appearance: [0-10] - [reasoning]
        * Potentially white growth: [0-10] - [reasoning]
        * No concentric patterns: [0-10] - [reasoning]
        * Irregular boundaries: [0-10] - [reasoning]
        * Color progression: [0-10] - [reasoning]
        * No dry/papery texture: [0-10] - [reasoning]
        TOTAL: [Sum/60] = [percentage]%
        
        BACTERIAL SPOT SCORE:
        * Small uniform spots: [0-10] - [reasoning]
        * Yellow halos: [0-10] - [reasoning]
        * Clustered appearance: [0-10] - [reasoning]
        * Same on both surfaces: [0-10] - [reasoning]
        * Remain distinct: [0-10] - [reasoning]
        * No large patches: [0-10] - [reasoning]
        TOTAL: [Sum/60] = [percentage]%
        
        HIGHEST SCORING DISEASE: [disease name] - [score]%
        
        FINAL DIAGNOSIS: [early_blight/late_blight/bacterial_spot/healthy]
        
        DIAGNOSTIC CERTAINTY: [90%+ certain / 75-90% certain / 50-75% certain]
        
        RULING OUT ALTERNATIVES: [Explain specifically why alternatives were ruled out based on absence of their definitive features and/or presence of counter-indicators]
        """
        
        # Stage 3 API call
        if isinstance(image, Image.Image):
            stage3_response = model.generate_content([stage3_prompt, {"mime_type": "image/jpeg", "data": img_bytes}])
        else:
            stage3_response = model.generate_content([stage3_prompt, image])
        
        stage3_text = stage3_response.text.lower()
        analysis_evidence['stage_results'].append({'stage': 3, 'response': stage3_text})
        
        # Extract stage 3 classification and map certainty to confidence
        stage3_classification = extract_classification(stage3_text)
        stage3_certainty = extract_certainty(stage3_text)
        
        # Map certainty expressions to confidence levels
        if "90%+ certain" in stage3_text:
            stage3_confidence = "high"
        elif "75-90% certain" in stage3_text:
            stage3_confidence = "medium"
        else:
            stage3_confidence = "low"
        
        # Update evidence with stage 3 findings
        update_evidence_dictionary(analysis_evidence, stage3_text, stage3_classification, stage3_confidence)
        
        # Determine the final classification through weighted voting
        # Stage 3 has highest weight, followed by stage 2, then stage 1
        classifications = {
            classification: 1,
            stage2_classification: 2,
            stage3_classification: 3
        }
        
        # Include TF model prediction with appropriate weight if available
        if tf_prediction and confidence:
            # Weight TF prediction based on its confidence
            tf_weight = 2 if confidence > 0.8 else 1
            classifications[tf_prediction] = tf_weight
        
        # Find the classification with the highest combined weight
        final_classification = max(classifications, key=classifications.get)
        
        # Determine if leaf is healthy
        is_healthy = final_classification == "healthy"
        
        # Determine final confidence level based on consistency and individual confidences
        if stage3_classification == stage2_classification == classification:
            final_confidence = "high"  # All stages agree
        elif stage3_classification == stage2_classification or stage3_classification == classification:
            final_confidence = stage3_confidence  # Use stage 3 confidence if it agrees with at least one other stage
        else:
            final_confidence = "medium" if stage3_confidence == "high" else "low"  # Lower confidence if stages disagree
        
        # Compile final detailed analysis if requested
        if detailed_report:
            # Create a detailed analysis dictionary
            detailed_analysis = {
                'classification': final_classification,
                'confidence': final_confidence,
                'is_healthy': is_healthy,
                'stage_results': analysis_evidence['stage_results'],
                'disease_markers': {
                    'early_blight': analysis_evidence['early_blight'],
                    'late_blight': analysis_evidence['late_blight'],
                    'bacterial_spot': analysis_evidence['bacterial_spot'],
                    'healthy': analysis_evidence['healthy']
                },
                'tf_prediction': tf_prediction if tf_prediction else None,
                'tf_confidence': confidence if confidence else None
            }
            return final_classification, is_healthy, detailed_analysis
        
        # Return simpler result if detailed report not needed
        return final_classification, is_healthy
    except Exception as e:
        st.error(f"Error in Gemini API analysis: {str(e)}")
        if detailed_report:
            return "unknown", False, {
                'error': str(e),
                'classification': "unknown",
                'confidence': "low",
                'is_healthy': False,
                'stage_results': []
            }
        return "unknown", False

def extract_classification(text):
    """Extract disease classification from structured API response"""
    text = text.lower()
    
    # Look for classification/diagnosis/final diagnosis sections
    for marker in ["classification:", "diagnosis:", "final diagnosis:"]:
        if marker in text:
            classification_text = text.split(marker)[1].strip().split("\n")[0]
            
            # Check for disease keywords
            if "early_blight" in classification_text or "early blight" in classification_text:
                return "early_blight"
            elif "late_blight" in classification_text or "late blight" in classification_text:
                return "late_blight"
            elif "bacterial_spot" in classification_text or "bacterial spot" in classification_text:
                return "bacterial_spot"
            elif "healthy" in classification_text:
                return "healthy"
    
    # Alternative extraction if structured format wasn't followed
    if "early_blight" in text or "early blight" in text:
        return "early_blight"
    elif "late_blight" in text or "late blight" in text:
        return "late_blight"
    elif "bacterial_spot" in text or "bacterial spot" in text:
        return "bacterial_spot"
    elif "healthy" in text and not any(d in text for d in ["early_blight", "early blight", "late_blight", "late blight", "bacterial_spot", "bacterial spot"]):
        return "healthy"
    
    return "unknown"

def extract_confidence(text):
    """
    Extract confidence level from Gemini API response text.
    
    Args:
        text: Response text from Gemini API
    
    Returns:
        str: Confidence level (high, medium, or low)
    """
    # Look for confidence statement
    confidence_pattern = r"confidence:\s*(high|medium|low)"
    match = re.search(confidence_pattern, text)
    
    if match:
        return match.group(1)
    
    # Check for certainty expressions
    if re.search(r"(high confidence|very confident|strongly believe|clear case|definitive)", text):
        return "high"
    elif re.search(r"(medium confidence|moderately confident|likely)", text):
        return "medium"
    else:
        return "low"

def extract_certainty(text):
    """
    Extract diagnostic certainty from Gemini API stage 3 response.
    
    Args:
        text: Response text from Gemini API stage 3
    
    Returns:
        str: Certainty level
    """
    certainty_pattern = r"diagnostic certainty:\s*(90\%\+ certain|75-90\% certain|50-75\% certain)"
    match = re.search(certainty_pattern, text)
    
    if match:
        return match.group(1)
    
    return "50-75% certain"  # Default to lower certainty if not found

def update_evidence_dictionary(evidence_dict, text, classification, confidence_level):
    """
    Update the evidence dictionary with information extracted from API response.
    
    Args:
        evidence_dict: Dictionary tracking evidence for various diseases
        text: Response text from Gemini API
        classification: Extracted classification
        confidence_level: Confidence level for the classification
    """
    # Define feature detection patterns for each disease
    feature_patterns = {
        'early_blight': [
            r"concentric rings.*present",
            r"bull'?s.?eye pattern",
            r"dry.*papery texture",
            r"angular lesions",
            r"yellow.*halos",
            r"bounded by leaf veins"
        ],
        'late_blight': [
            r"water.soaked.*present",
            r"greasy.looking",
            r"white fuzzy growth",
            r"pale green to brown",
            r"irregular.*not bounded by veins",
            r"wet appearance"
        ],
        'bacterial_spot': [
            r"small circular spots",
            r"yellow halos",
            r"clustered appearance",
            r"shot.hole appearance",
            r"small \(1-3mm\)",
            r"spots.*both leaf surfaces"
        ],
        'healthy': [
            r"uniform green",
            r"no lesions",
            r"no discoloration",
            r"absence of spots",
            r"normal leaf texture",
            r"no necrosis"
        ]
    }
    
    # Update markers for each disease based on text
    for disease, patterns in feature_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if match not in evidence_dict[disease]['markers']:
                        evidence_dict[disease]['markers'].append(match)
    
    # Update confidence for classified disease
    if classification in evidence_dict:
        # Convert confidence to numerical value
        confidence_value = {'high': 0.9, 'medium': 0.7, 'low': 0.5}.get(confidence_level, 0.5)
        
        # Update using maximum confidence across stages
        evidence_dict[classification]['confidence'] = max(
            evidence_dict[classification]['confidence'],
            confidence_value
        )

def extract_counter_evidence(evidence_dict, text):
    """
    Extract counter-evidence against diagnoses from stage 2 and 3 responses.
    
    Args:
        evidence_dict: Dictionary tracking evidence for various diseases
        text: Response text from Gemini API
    """
    # Define counter-evidence patterns for each disease
    counter_patterns = {
        'early_blight': [
            r"absence of concentric rings",
            r"no bull'?s.?eye pattern",
            r"wet.*not dry texture",
            r"non.angular lesions",
            r"water.soaked appearance"
        ],
        'late_blight': [
            r"no water.soaking",
            r"absence of greasy appearance",
            r"no white fuzzy growth",
            r"concentric patterns present",
            r"dry texture"
        ],
        'bacterial_spot': [
            r"lesions larger than 3mm",
            r"no yellow halos",
            r"lesions coalesce",
            r"concentric patterns present",
            r"water.soaked patches"
        ],
        'healthy': [
            r"lesions present",
            r"discoloration observed",
            r"necrotic tissue",
            r"chlorosis detected",
            r"abnormal texture"
        ]
    }
    
    # Extract counter-evidence statements
    ruling_out_match = re.search(r"ruling out alternatives?:(.+?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)
    if ruling_out_match:
        ruling_out_text = ruling_out_match.group(1).lower()
        
        # Check for disease-specific rulings
        for disease in evidence_dict:
            if disease == 'stage_results':
                continue
                
            disease_text = disease.replace('_', ' ')
            if f"ruling out {disease_text}" in ruling_out_text or f"{disease_text} ruled out" in ruling_out_text:
                # Extract the specific reason
                reason_match = re.search(fr"{disease_text}.*?because(.*?)(?:\.|$)", ruling_out_text, re.IGNORECASE)
                if reason_match and reason_match.group(1) not in evidence_dict[disease]['counter_evidence']:
                    evidence_dict[disease]['counter_evidence'].append(reason_match.group(1).strip())
    
    # Check for specific counter-evidence patterns
    for disease, patterns in counter_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if match not in evidence_dict[disease]['counter_evidence']:
                        evidence_dict[disease]['counter_evidence'].append(match)


def extract_classification(text):
    """Extract disease classification from structured API response"""
    text = text.lower()
    
    # Look for classification/diagnosis/final diagnosis sections
    for marker in ["classification:", "diagnosis:", "final diagnosis:"]:
        if marker in text:
            classification_text = text.split(marker)[1].strip().split("\n")[0]
            
            # Check for disease keywords
            if "early_blight" in classification_text or "early blight" in classification_text:
                return "early_blight"
            elif "late_blight" in classification_text or "late blight" in classification_text:
                return "late_blight"
            elif "bacterial_spot" in classification_text or "bacterial spot" in classification_text:
                return "bacterial_spot"
            elif "healthy" in classification_text:
                return "healthy"
    
    # Alternative extraction if structured format wasn't followed
    if "early_blight" in text or "early blight" in text:
        return "early_blight"
    elif "late_blight" in text or "late blight" in text:
        return "late_blight"
    elif "bacterial_spot" in text or "bacterial spot" in text:
        return "bacterial_spot"
    elif "healthy" in text and not any(d in text for d in ["early_blight", "early blight", "late_blight", "late blight", "bacterial_spot", "bacterial spot"]):
        return "healthy"
    
    return "unknown"


def extract_confidence(text):
    """Extract confidence level from API response"""
    text = text.lower()
    
    # Look for confidence/certainty markers
    for marker in ["confidence:", "confidence level:", "diagnostic certainty:"]:
        if marker in text:
            confidence_text = text.split(marker)[1].strip().split("\n")[0]
            
            if any(term in confidence_text for term in ["high", "90%", "strong", "certain"]):
                return "high"
            elif any(term in confidence_text for term in ["medium", "moderate", "75-90%", "75%"]):
                return "medium"
            elif any(term in confidence_text for term in ["low", "50-75%", "uncertain", "unclear"]):
                return "low"
    
    # Default if not found
    return "medium"


def map_certainty_to_confidence(text):
    """Map certainty percentage ranges to confidence levels"""
    text = text.lower()
    
    if "90%+ certain" in text or "highly certain" in text:
        return "high"
    elif "75-90% certain" in text or "moderately certain" in text:
        return "medium"
    elif "50-75% certain" in text or "somewhat certain" in text:
        return "low"
    
    # Fall back to standard confidence extraction
    return extract_confidence(text)


def extract_disease_markers(text, disease):
    """Extract positive markers for a specific disease from response text"""
    markers = []
    disease_name = disease.replace('_', ' ')
    
    # Check for primary markers based on disease
    if disease == "early_blight":
        if "concentric rings" in text or "bull's-eye" in text or "target pattern" in text:
            markers.append("concentric rings/bull's-eye pattern present")
        if "dry" in text and "papery" in text:
            markers.append("dry, papery texture")
        if "angular" in text and "vein" in text:
            markers.append("angular lesions bounded by veins")
        if "yellow halo" in text:
            markers.append("yellow halos around lesions")
    
    elif disease == "late_blight":
        if "water-soaked" in text or "greasy" in text:
            markers.append("water-soaked/greasy appearance")
        if "white" in text and ("fuzzy" in text or "growth" in text):
            markers.append("white fuzzy growth")
        if "margins" in text or "tips" in text:
            markers.append("lesions at leaf margins/tips")
        if "irregular" in text and "patch" in text:
            markers.append("irregular patches")
    
    elif disease == "bacterial_spot":
        if "small" in text and "circular" in text and "spot" in text:
            markers.append("small circular spots")
        if "yellow halo" in text:
            markers.append("yellow halos around spots")
        if "cluster" in text:
            markers.append("clustered spots")
        if "shot-hole" in text:
            markers.append("shot-hole appearance")
    
    elif disease == "healthy":
        if "uniform" in text and "color" in text:
            markers.append("uniform coloration")
        if "no" in text and ("spot" in text or "lesion" in text):
            markers.append("no spots or lesions")
        if "no" in text and "discolor" in text:
            markers.append("no discoloration")
    
    # Look for disease name with evidence
    for line in text.split('\n'):
        if disease_name in line and ":" in line:
            evidence_part = line.split(":", 1)[1].strip()
            markers.append(evidence_part)
    
    return markers


def extract_negative_markers(text, disease):
    """Extract negative markers for a disease from response text"""
    negative_markers = []
    disease_name = disease.replace('_', ' ')
    
    # Check for negative markers based on disease
    if disease == "early_blight":
        if "no concentric rings" in text or "absence of concentric rings" in text:
            negative_markers.append("no concentric rings/bull's-eye pattern")
        if "not dry" in text or "wet texture" in text:
            negative_markers.append("lacks dry, papery texture")
    
    elif disease == "late_blight":
        if "no water-soaked" in text or "absence of water-soaked" in text:
            negative_markers.append("no water-soaked/greasy appearance")
        if "no white fuzzy growth" in text or "absence of white growth" in text:
            negative_markers.append("no white fuzzy growth")
    
    elif disease == "bacterial_spot":
        if "no small circular spots" in text or "spots are large" in text:
            negative_markers.append("lacks small circular spots")
        if "no yellow halos" in text:
            negative_markers.append("no yellow halos around spots")
    
    # Look for negative references to the disease
    for line in text.split('\n'):
        if ("not " + disease_name) in line or ("no " + disease_name) in line:
            negative_markers.append(f"ruled out {disease_name}: {line.strip()}")
        if "absent" in line and disease_name in line:
            negative_markers.append(f"features absent: {line.strip()}")
    
    return negative_markers


def generate_evidence_summary(evidence_dict):
    """Generate a summary of collected evidence for final decision making"""
    summary = ""
    
    for disease, data in evidence_dict.items():
        if disease != "stage_results" and data['markers']:
            summary += f"\n{disease.replace('_', ' ')} evidence:\n"
            
            # Get unique markers (remove duplicates)
            unique_markers = list(set(data['markers']))
            
            # Add markers with bullet points
            for marker in unique_markers:
                summary += f"* {marker}\n"
            
            summary += f"Overall confidence: {data['confidence']:.2f}\n"
    
    return summary


def weighted_decision(evidence_dict, tf_prediction=None, tf_confidence=None):
    """Make a weighted decision based on collected evidence when no clear diagnosis emerged"""
    # Calculate weights for each disease
    weights = {}
    
    for disease, data in evidence_dict.items():
        if disease != "stage_results":
            # Base weight on confidence and number of supporting markers
            weight = data['confidence'] * len(data['markers'])
            
            # Count negative markers for competing diseases as supporting evidence
            for other_disease, other_data in evidence_dict.items():
                if other_disease != disease and other_disease != "stage_results":
                    negative_markers = [m for m in other_data['markers'] if "no " in m or "absence" in m]
                    weight += 0.2 * len(negative_markers)
            
            weights[disease] = weight
    
    # Add TF model prediction as additional weight if available
    if tf_prediction and tf_prediction in weights and tf_confidence:
        weights[tf_prediction] += tf_confidence * 0.5  # Add half the ML confidence as weight
    
    # Determine the disease with the highest weight
    if weights:
        max_disease = max(weights, key=weights.get)
        if weights[max_disease] > 0:
            return max_disease
    
    return "unknown"


def prepare_detailed_report(evidence_dict, final_classification, final_confidence):
    """Prepare detailed analysis report with evidence and confidence scores"""
    report = {
        'final_diagnosis': final_classification,
        'confidence_level': final_confidence,
        'confidence_score': {'high': 0.9, 'medium': 0.7, 'low': 0.5}.get(final_confidence, 0.5),
        'diagnostic_evidence': {},
        'differential_diagnosis': []
    }
    
    # Add normalized confidence scores for all diseases
    total_weight = 0
    disease_weights = {}
    
    for disease, data in evidence_dict.items():
        if disease != "stage_results":
            # Calculate weight based on confidence and marker count
            marker_count = len(data['markers'])
            weight = data['confidence'] * (1 + 0.1 * marker_count)  # Add 10% per marker
            disease_weights[disease] = weight
            total_weight += weight
    
    # Normalize to get relative confidence
    if total_weight > 0:
        for disease, weight in disease_weights.items():
            normalized_score = weight / total_weight
            
            # Add to differential diagnosis list in sorted order
            report['differential_diagnosis'].append({
                'disease': disease,
                'relative_confidence': round(normalized_score, 2),
                'supporting_evidence': evidence_dict[disease]['markers'][:5]  # Top 5 markers
            })
    
    # Sort differential diagnosis by confidence
    report['differential_diagnosis'] = sorted(
        report['differential_diagnosis'], 
        key=lambda x: x['relative_confidence'], 
        reverse=True
    )
    
    # Add detailed evidence for the final diagnosis
    if final_classification in evidence_dict:
        report['diagnostic_evidence'] = {
            'primary_markers': [m for m in evidence_dict[final_classification]['markers'] if any(term in m.lower() for term in ['primary', 'definitive', 'key'])],
            'supporting_markers': [m for m in evidence_dict[final_classification]['markers'] if not any(term in m.lower() for term in ['primary', 'definitive', 'key'])],
            'ruling_out_evidence': []
        }
        
        # Add evidence ruling out competing diagnoses
        for disease, data in evidence_dict.items():
            if disease != final_classification and disease != "stage_results":
                negative_markers = [m for m in data['markers'] if "no " in m or "absence" in m or "ruled out" in m]
                if negative_markers:
                    report['diagnostic_evidence']['ruling_out_evidence'].extend(negative_markers)
    
    return report

def get_detailed_blight_analysis(image, disease):
    """Get a detailed analysis specifically for early vs late blight"""
    if model and disease in ["early_blight", "late_blight"]:
        try:
            # Convert image for API compatibility if needed
            if isinstance(image, Image.Image):
                img_bytes = BytesIO()
                image.save(img_bytes, format='JPEG')
                img_bytes = img_bytes.getvalue()
            
            # Detailed blight analysis prompt
            blight_analysis_prompt = f"""
            Analyze this leaf image that shows {disease.replace('_', ' ')}. 
            
            Provide a SHORT analysis of WHY this is {disease.replace('_', ' ')} by identifying the specific visual markers present:
            
            1. For early blight, list evidence of:
               - Concentric rings/target patterns
               - Distribution on older/lower leaves
               - Angular bounded lesions
               - Dry papery texture
            
            2. For late blight, list evidence of:
               - Water-soaked/greasy appearance
               - White fuzzy growth on undersides
               - Irregular unbounded lesions
               - Location at leaf margins/tips
            
            Keep your response under 100 words and focus ONLY on the visual evidence visible in THIS specific image.
            """
            
            # Make API call
            response = model.generate_content([blight_analysis_prompt, {"mime_type": "image/jpeg", "data": img_bytes}])
            
            return response.text.strip()
        except Exception as e:
            return None
    
    return None

def get_disease_info(disease, enhanced=False):
    """Get detailed information about a disease using API for enhanced descriptions"""
    if model and disease != "unknown":
        # Enhanced prompt for better disease descriptions and prevention methods
        prompt = f"""
        Provide comprehensive information about {disease.replace('_', ' ')} in plants.
        
        Format your response in two clearly labeled sections:
        
        DESCRIPTION:
        - What is {disease.replace('_', ' ')}?
        - What causes it?
        - What are the specific visual symptoms? (Be detailed about appearance)
        - How does it progress over time?
        - What conditions favor this disease?
        
        PREVENTION AND TREATMENT:
        - List 5-7 specific prevention methods (with detailed implementation steps)
        - Include cultural, chemical, and biological control options
        - Include both preventative and treatment measures
        - Mention timing considerations for treatments
        - Suggest resistant varieties if applicable
        
        Keep the total response under 250 words, but make it detailed and actionable.
        """
        
        # If we're asking about a blight and want enhanced info, add specific comparative elements
        if enhanced and disease in ["early_blight", "late_blight"]:
            prompt += f"""
            
            DIFFERENTIATING FEATURES:
            - List 3-5 key visual features that distinguish {disease.replace('_', ' ')} from {('late' if 'early' in disease else 'early')}_blight
            - Explain why these distinguishing features occur (biological reasons)
            - Explain common confusion points between the two blights
            """
        
        try:
            response = model.generate_content(prompt)
            text = response.text
            
            # Split into description and prevention
            parts = text.split("PREVENTION AND TREATMENT")
            if len(parts) > 1:
                description = parts[0].replace("DESCRIPTION:", "").strip()
                
                # If enhanced, check for differentiating features section
                if enhanced and "DIFFERENTIATING FEATURES" in parts[1]:
                    treatment_parts = parts[1].split("DIFFERENTIATING FEATURES")
                    prevention = treatment_parts[0].strip()
                    differentiating = treatment_parts[1].strip()
                    return {"description": description, "prevention": prevention, "differentiating": differentiating}
                else:
                    prevention = parts[1].strip()
            else:
                # Alternative parsing if the format isn't as expected
                if "DESCRIPTION" in text and "PREVENTION" in text:
                    desc_start = text.find("DESCRIPTION")
                    prev_start = text.find("PREVENTION")
                    
                    description = text[desc_start:prev_start].replace("DESCRIPTION:", "").strip()
                    
                    if enhanced and "DIFFERENTIATING FEATURES" in text:
                        diff_start = text.find("DIFFERENTIATING FEATURES")
                        prevention = text[prev_start:diff_start].replace("PREVENTION AND TREATMENT:", "").strip()
                        differentiating = text[diff_start:].replace("DIFFERENTIATING FEATURES:", "").strip()
                        return {"description": description, "prevention": prevention, "differentiating": differentiating}
                    else:
                        prevention = text[prev_start:].replace("PREVENTION AND TREATMENT:", "").strip()
                else:
                    # Fallback if structure doesn't match expectations
                    description = text[:len(text)//2].strip()
                    prevention = text[len(text)//2:].strip()
                
            return {"description": description, "prevention": prevention}
        except Exception as e:
            st.warning(f"⚠️ Gemini AI error: {e}. Using local data.")
    
    return DISEASE_INFO.get(disease, {
        "description": "No description available.",
        "prevention": "Consult an agricultural expert."
    })

def load_comparison_table():
    """Load the pre-defined blight comparison information as a Streamlit table"""
    data = BLIGHT_COMPARISON
    
    st.markdown("### 📊 Early Blight vs Late Blight Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Early Blight (Alternaria solani)")
        for key, value in data['early_blight'].items():
            st.markdown(f"**{key.replace('_', ' ').title()}{value}")
    
    with col2:
        st.markdown("#### Late Blight (Phytophthora infestans)")
        for key, value in data['late_blight'].items():
            st.markdown(f"**{key.replace('_', ' ').title()}{value}")

# UI Layout
st.title('🌿 Plant Guardian')

tab1, tab2, tab3 = st.tabs(["🖼️ Upload & Predict", "🔍 Blight Comparison", "ℹ️ About"])

with tab1:
    uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])
    
    # Add prediction method selection
    prediction_method = st.radio(
        "Select Prediction Method:",
        ["TensorFlow + AI (recommended)", "TensorFlow Only", "AI Only (if available)"],
        horizontal=True
    )

    if uploaded_file and disease_model:
        with st.spinner("🔍 Processing image..."):
            try:
                progress_bar = st.progress(0)
                
                # Process image for model prediction
                img = Image.open(uploaded_file)
                img_display = img.copy()  # Save a copy for display
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                
                # Update progress
                progress_bar.progress(20)
                
                # TensorFlow prediction (run for both TensorFlow and combined methods)
                tf_disease = "unknown"
                confidence = 0
                tf_is_healthy = False
                
                if prediction_method != "AI Only (if available)":
                    # Make prediction with TensorFlow
                    predictions = disease_model.predict(img_array)
                    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
                    confidence = np.max(predictions[0])
                    
                    # Extract only disease information
                    tf_disease, tf_is_healthy = extract_disease_only(predicted_class)
                
                # Update progress
                progress_bar.progress(40)
                
                # AI prediction (run for both AI and combined methods)
                api_disease = "unknown"
                api_is_healthy = False
                api_used = False
                
                if prediction_method != "TensorFlow Only" and model:
                    st.info("⚙️ Using AI image analysis to enhance prediction accuracy...")
                    
                    # For combined method, pass TensorFlow prediction to Gemini for context
                    if prediction_method == "TensorFlow + AI (recommended)":
                        api_disease, api_is_healthy = analyze_image_with_gemini(img_display, tf_disease, confidence)
                    else:
                        # For AI-only, don't pass the TensorFlow prediction
                        api_disease, api_is_healthy = analyze_image_with_gemini(img_display)
                        
                    api_used = True
                
                # Determine final result based on prediction method
                if prediction_method == "TensorFlow Only":
                    disease = tf_disease
                    is_healthy = tf_is_healthy
                elif prediction_method == "AI Only (if available)":
                    if api_disease != "unknown":
                        disease = api_disease
                        is_healthy = api_is_healthy
                    else:
                        disease = "unknown"
                        is_healthy = False
                else:  # Combined method
                    if api_disease != "unknown":
                        disease = api_disease
                        is_healthy = api_is_healthy
                    else:
                        disease = tf_disease
                        is_healthy = tf_is_healthy
                
                # Update progress
                progress_bar.progress(60)
                
                # Get detailed disease information with enhanced info for blights
                disease_info = get_disease_info(disease, enhanced=(disease in ["early_blight", "late_blight"]))
                
                # Get specific blight analysis if applicable (only for AI methods)
                blight_analysis = None
                if disease in ["early_blight", "late_blight"] and model and prediction_method != "TensorFlow Only":
                    progress_bar.progress(75)
                    blight_analysis = get_detailed_blight_analysis(img_display, disease)
                
                # Update progress bar
                progress_bar.progress(100)
                
                # Display results
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"### 🌱 Diagnosis: **{disease.replace('_', ' ').title()}**")
                    
                    # Display confidence for TensorFlow predictions
                    if prediction_method != "AI Only (if available)":
                        st.markdown(f"**Model Confidence:`{confidence:.2f}`")
                    
                    # Show comparison between methods when appropriate
                    if prediction_method == "TensorFlow + AI (recommended)" and api_used and tf_disease != api_disease:
                        st.markdown(f"**TensorFlow Prediction:`{tf_disease.replace('_', ' ').title()}`")
                        st.markdown(f"**Gemini AI Prediction:`{api_disease.replace('_', ' ').title()}`")
                        st.markdown(f"**Final Diagnosis:`{disease.replace('_', ' ').title()}` (AI-enhanced)")
                    else:
                        analysis_method = ""
                        if prediction_method == "TensorFlow Only":
                            analysis_method = "TensorFlow Model"
                        elif prediction_method == "AI Only (if available)":
                            analysis_method = "Gemini AI"
                        else:
                            analysis_method = "TensorFlow Model + Gemini AI"
                        
                        st.markdown(f"**Analysis Method:`{analysis_method}`")
                    
                    st.image(img_display, caption='📷 Uploaded Leaf Image', width=300)
                    
                    # Show blight-specific analysis if available
                    if blight_analysis:
                        st.markdown("### 🔬 Image Analysis")
                        st.markdown(blight_analysis)
                
                with col2:
                    if is_healthy:
                        st.subheader("✅ Plant is Healthy")
                        st.markdown(disease_info["description"])
                    else:
                        st.subheader("📋 Disease Details")
                        st.markdown("**Description:**")
                        st.markdown(disease_info["description"])
                        
                        st.subheader("🛡️ Prevention & Treatment")
                        st.markdown(disease_info["prevention"])
                        
                        # Show differentiating features if available
                        if "differentiating" in disease_info:
                            st.subheader("🔍 Differentiating Features")
                            st.markdown(disease_info["differentiating"])
                
                # Show confidence warning if needed
                if prediction_method != "AI Only (if available)" and confidence < 0.7 and disease == "unknown":
                    alt_class = CLASS_NAMES[np.argsort(predictions[0])[-2]]
                    alt_disease, _ = extract_disease_only(alt_class)
                    st.warning(f"⚠️ Model is uncertain. Alternative possibility: **{alt_disease.replace('_', ' ').title()}**")
            
            except Exception as e:
                st.error(f"⚠️ Error processing image: {e}")

with tab2:
    st.markdown("""
    ## 🔍 Early vs Late Blight Differentiation
    
    Distinguishing between early blight and late blight is critical for effective treatment. These diseases share some similarities but have distinct characteristics.
    """)
    
    load_comparison_table()
    
    st.markdown("""
    ### 📷 Visual Identification Tips
    
    When examining leaf images:
    
    1. **Look for concentric rings first** - The most reliable indicator of early blight
    2. **Check for white fuzzy growth** - A definitive sign of late blight
    3. **Notice lesion locations** - Early blight starts on older leaves; late blight can appear anywhere
    4. **Observe leaf margins** - Late blight often begins at leaf edges
    5. **Consider recent weather** - Cool, wet conditions favor late blight; warm humid conditions favor early blight
    """)

with tab3:
    st.markdown("""
    ## ℹ️ About This App
    **Plant Guardian** is an AI-powered tool for detecting plant diseases from leaf images.
    - 📷 **Upload a photo** using your device.
    - 🌿 **Get instant disease predictions** from leaf images.
    - 🤖 **Uses AI (Gemini API)** for enhanced image analysis and detailed disease information.
    - 🚀 **Optimized for speed** with TensorFlow & Streamlit.

    ### How It Works:
    1. **User Uploads an Image** - The image of a plant leaf is uploaded.
    2. **Image Preprocessing** - The image is resized and prepared for analysis.
    3. **Primary Classification** - The trained TensorFlow model predicts the disease.
    4. **AI Enhancement** - Gemini AI verifies and refines the prediction through advanced image analysis.
    5. **Detailed Information** - AI generates comprehensive disease descriptions and treatment plans.
    6. **Results Display** - The app shows the diagnosis, confidence score, and actionable recommendations.
    
    ### Supported Diseases:
    - **Bacterial Spot** - Dark, water-soaked lesions with yellow halos
    - **Early Blight** - Brown concentric rings, target-like patterns
    - **Late Blight** - Water-soaked patches with whitish growth underneath
    - **Healthy** - Normal appearance without disease symptoms
    """)

# Apply light theme styling
st.markdown("""
<style>
    /* Light theme styles */
    .main {
        background-color: #ffffff;
        color: #333333;
    }
    h1 {
        text-align: center;
        color: #2e7d32;
    }
    h2, h3 {
        color: #1b5e20;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f8e9;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        color: #558b2f;
    }
    .stTabs [aria-selected="true"] {
        background-color: #c5e1a5;
        color: #33691e;
    }
    .stFileUploader > div > button {
        background-color: #66bb6a;
        color: white;
        border-radius: 5px;
    }
    .stFileUploader > div > button:hover {
        background-color: #43a047;
    }
    .stProgress > div > div > div > div {
        background-color: #81c784;
    }
    .stAlert {
        background-color: #f1f8e9;
        border: 1px solid #aed581;
    }
    /* Enhanced styling for disease info */
    .stMarkdown {
        line-height: 1.6;
    }
    blockquote {
        border-left: 4px solid #81c784;
        padding-left: 16px;
        margin-left: 0;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Display a footer
st.markdown("""
---
📱 **Plant Guardian** | Created with ❤️ by Plant Health Team | © 2024
""")