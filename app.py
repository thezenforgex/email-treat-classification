import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Setup page layout
st.set_page_config(
    page_title="Email Threat Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Grey & Orange UI matching screenshot
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #12141d !important;
        color: #e0e6ed;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1a1c28 !important;
        border-right: 1px solid #26293b;
        padding-top: 20px;
    }
    
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 16px;
    }
    
    .sidebar-item {
        font-size: 0.9rem;
        color: #a0a8b9;
        margin-bottom: 12px;
    }
    
    .sidebar-value {
        color: #e0e6ed;
        font-weight: 500;
    }
    
    .status-badge {
        background: rgba(46, 204, 113, 0.12);
        border: 1px solid rgba(46, 204, 113, 0.3);
        color: #2ecc71;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 600;
        margin-top: 24px;
        text-align: center;
    }

    /* Main Area Styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }
    
    .main-subtitle {
        font-size: 0.95rem;
        color: #949cb1;
        margin-bottom: 24px;
    }

    /* Textarea Styling */
    .stTextArea textarea {
        background-color: #1e2130 !important;
        color: #e6edf3 !important;
        border: 1px solid #2d3246 !important;
        border-radius: 8px !important;
        font-size: 0.95rem;
    }
    .stTextArea textarea:focus {
        border-color: #ff5252 !important;
        box-shadow: 0 0 0 1px #ff5252 !important;
    }

    /* Primary Button Styling */
    div.stButton > button {
        background: #ff5252 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: #ff3333 !important;
        box-shadow: 0 4px 12px rgba(255, 82, 82, 0.4) !important;
    }

    /* Result Banner Boxes */
    .threat-banner {
        background-color: #2c161a;
        border: 1px solid #632029;
        color: #ff6b6b;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .routine-banner {
        background-color: #142a20;
        border: 1px solid #1c4d37;
        color: #2ecc71;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .result-description {
        font-size: 0.92rem;
        color: #a0a8b9;
        margin-bottom: 24px;
    }

    /* Metric Cards */
    .metric-title {
        font-size: 0.82rem;
        color: #838ca1;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    .metric-value-large {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
    }

    .footer-text {
        font-size: 0.85rem;
        color: #636b7e;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model and scaler
@st.cache_resource
def get_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(base_dir)
    sys.path.append(os.path.join(base_dir, 'src'))
    
    from src.data_preprocessing import FEATURE_COLS
    from src.predict import load_inference_pipeline
    from src.extract_features import extract_features_from_raw_text
    
    model, scaler, meta = load_inference_pipeline()
    return model, scaler, meta, FEATURE_COLS, extract_features_from_raw_text

try:
    model, scaler, meta, FEATURE_COLS, extract_features_from_raw_text = get_artifacts()
    model_ready = True
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
    model_ready = False

# Sidebar (Model Information matching screenshot)
with st.sidebar:
    st.markdown('<div class="sidebar-header">Model Information</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-item">Model: <span class="sidebar-value">{meta.get("best_model_name", "Logistic Regression")}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-item">Feature extraction: <span class="sidebar-value">Metadata & Keywords</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-item">Model source: <span class="sidebar-value">Scikit-learn Pipeline</span></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="status-badge">Model loaded successfully</div>', unsafe_allow_html=True)

# Main Title Section
st.markdown('<div class="main-title">🛡️ Email Threat Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Enter the content of an email below to classify it as routine or potentially threatening.</div>', unsafe_allow_html=True)

# Text Area Input
email_input = st.text_area(
    "Email Content",
    height=220,
    placeholder="Paste email content here..."
)

# Analyze Button
analyze_clicked = st.button("🔍 Analyze Email")

# Output Section
if analyze_clicked:
    if not email_input or not email_input.strip():
        st.warning("Please enter email content before analyzing.")
    elif model_ready:
        features = extract_features_from_raw_text(email_input)
        input_df = pd.DataFrame([features])[FEATURE_COLS]
        
        if meta.get('requires_scaling', True):
            input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=FEATURE_COLS)
        else:
            input_df_scaled = input_df
            
        pred = model.predict(input_df_scaled)[0]
        prob = model.predict_proba(input_df_scaled)[0][1]
        
        confidence = prob * 100 if pred == 1 else (1 - prob) * 100
        msg_len = len(email_input.strip())
        num_features = len(FEATURE_COLS)
        
        if pred == 1:
            st.markdown('<div class="threat-banner">🚨 Potentially Threatening</div>', unsafe_allow_html=True)
            st.markdown('<div class="result-description">The model classified this email as potentially threatening.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="routine-banner">✅ Routine Email</div>', unsafe_allow_html=True)
            st.markdown('<div class="result-description">The model classified this email as routine.</div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Metrics Row matching screenshot layout
        col_m1, col_m2, col_m3 = st.columns([1, 1.2, 1])
        
        with col_m1:
            st.markdown('<div class="metric-title">Model Confidence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value-large">{confidence:.2f}%</div>', unsafe_allow_html=True)
            
        with col_m2:
            st.markdown('<div class="metric-title">Message Length</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value-large">{msg_len:,} characters</div>', unsafe_allow_html=True)
            
        with col_m3:
            st.markdown('<div class="metric-title">Extracted Features</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value-large">{num_features}</div>', unsafe_allow_html=True)

st.markdown('<div class="footer-text">Educational prototype for email threat classification.</div>', unsafe_allow_html=True)
