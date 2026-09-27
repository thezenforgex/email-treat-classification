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

# Robust CSS Override for Dark Grey & Orange UI matching target design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --primary-color: #e74c3c !important;
        --background-color: #12141d !important;
        --secondary-background-color: #1a1c28 !important;
        --text-color: #e0e6ed !important;
    }

    /* Entire App & Header Dark Background */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #12141d !important;
        color: #e0e6ed !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: #12141d !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        background-color: #1a1c28 !important;
        border-right: 1px solid #282c3f !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #c5cbd8 !important;
    }

    .sidebar-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 16px;
    }
    
    .sidebar-item {
        font-size: 0.92rem;
        color: #9ea7b8 !important;
        margin-bottom: 12px;
    }
    
    .sidebar-value {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .status-badge {
        background-color: rgba(46, 204, 113, 0.12) !important;
        border: 1px solid rgba(46, 204, 113, 0.4) !important;
        color: #2ecc71 !important;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 600;
        margin-top: 24px;
        text-align: center;
    }

    /* Main Area Header */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff !important;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }
    
    .main-subtitle {
        font-size: 0.95rem;
        color: #949cb1 !important;
        margin-bottom: 24px;
    }

    /* Input Text Area Override */
    [data-testid="stTextArea"] label {
        color: #e0e6ed !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    [data-testid="stTextArea"] textarea, textarea {
        background-color: #1e2130 !important;
        color: #ffffff !important;
        border: 1px solid #2e344a !important;
        border-radius: 8px !important;
        font-size: 0.95rem;
    }
    
    [data-testid="stTextArea"] textarea:focus {
        border-color: #e74c3c !important;
        box-shadow: 0 0 0 1px #e74c3c !important;
    }

    /* Primary Coral-Orange Action Button */
    div.stButton > button, button[kind="primary"] {
        background: #e74c3c !important;
        background-color: #e74c3c !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 2px 8px rgba(231, 76, 60, 0.3) !important;
    }
    
    div.stButton > button:hover {
        background: #c0392b !important;
        background-color: #c0392b !important;
        box-shadow: 0 4px 12px rgba(231, 76, 60, 0.5) !important;
    }

    /* Result Banner Boxes */
    .threat-banner {
        background-color: #381a1d !important;
        border: 1px solid #7f1d1d !important;
        color: #f87171 !important;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .routine-banner {
        background-color: #142e23 !important;
        border: 1px solid #14532d !important;
        color: #4ade80 !important;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .result-description {
        font-size: 0.92rem;
        color: #949cb1 !important;
        margin-top: 8px;
        margin-bottom: 24px;
    }

    /* Metrics Row */
    .metric-title {
        font-size: 0.85rem;
        color: #838ca1 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    .metric-value-large {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff !important;
    }

    .footer-text {
        font-size: 0.85rem;
        color: #636b7e !important;
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
        
        # Metrics Row matching target screenshot layout
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
