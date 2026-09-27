import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Setup page config
st.set_page_config(
    page_title="Email Threat & Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

# Custom CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .header-container {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 14px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: 800;
        color: #4facfe;
        margin-bottom: 6px;
    }
    
    .header-desc {
        font-size: 0.95rem;
        color: #cfd8dc;
    }
    
    .result-threat {
        background: linear-gradient(135deg, #eb3b5a 0%, #fa8231 100%);
        color: white;
        padding: 22px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 800;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(235, 59, 90, 0.4);
    }
    
    .result-safe {
        background: linear-gradient(135deg, #20bf6b 0%, #0fb9b1 100%);
        color: white;
        padding: 22px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 800;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(32, 191, 107, 0.4);
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

# App Header
st.markdown("""
<div class="header-container">
    <div class="header-title">🛡️ Email Threat & Spam Detector</div>
    <div class="header-desc">Paste raw email text below to instantly analyze if it is Spam / Threat or Routine (Safe)</div>
</div>
""", unsafe_allow_html=True)

# Preset Samples
col_sample1, col_sample2, col_sample3 = st.columns(3)

if 'email_text' not in st.session_state:
    st.session_state['email_text'] = ""

with col_sample1:
    if st.button("🔴 Load Phishing Email Sample"):
        st.session_state['email_text'] = (
            "URGENT SECURITY NOTICE: Your bank account has been SUSPENDED!\n"
            "Verify your account credentials immediately to prevent permanent account termination.\n"
            "Click here: http://secure-login-bank-verification.xyz/verify to update your password now."
        )
with col_sample2:
    if st.button("🟢 Load Routine Email Sample"):
        st.session_state['email_text'] = (
            "Hi Alex,\n\n"
            "Here is the draft of the project report for our weekly sync on Thursday.\n"
            "Please review it when you have a moment and let me know your thoughts.\n\n"
            "Best regards,\nSarah"
        )
with col_sample3:
    if st.button("🧹 Clear Text"):
        st.session_state['email_text'] = ""

# Input Text Area
email_input = st.text_area(
    "Paste Email Content Below:",
    value=st.session_state['email_text'],
    height=220,
    placeholder="Paste email body, subject line, or raw email text here..."
)

# Predict Action
if st.button("🔍 Check Email Threat Status", type="primary", use_container_width=True):
    if not email_input or not email_input.strip():
        st.warning("Please paste some email text to analyze!")
    elif model_ready:
        features = extract_features_from_raw_text(email_input)
        input_df = pd.DataFrame([features])[FEATURE_COLS]
        
        if meta.get('requires_scaling', True):
            input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=FEATURE_COLS)
        else:
            input_df_scaled = input_df
            
        pred = model.predict(input_df_scaled)[0]
        prob = model.predict_proba(input_df_scaled)[0][1]
        
        st.markdown("---")
        
        if pred == 1:
            st.markdown(
                f'<div class="result-threat">🚨 SPAM / POTENTIALLY THREATENING<br>'
                f'<span style="font-size:1.1rem; font-weight:normal;">Calculated Threat Confidence: <b>{prob*100:.1f}%</b></span></div>',
                unsafe_allow_html=True
            )
            st.progress(float(prob))
            
            st.subheader("⚠️ Suspicious Risk Factors Detected:")
            if features['suspicious_tld'] == 1:
                st.error("• Contains suspicious high-risk top-level domain (.xyz, .top, .ru, etc.)")
            if features['keyword_indicators'] >= 2:
                st.error(f"• High frequency of urgent/phishing trigger keywords ({features['keyword_indicators']} detected)")
            if features['url_count'] >= 1:
                st.error(f"• Includes embedded hyperlink(s) ({features['url_count']} link(s))")
            if features['uppercase_ratio'] > 0.15:
                st.error(f"• Elevated proportion of uppercase text ({features['uppercase_ratio']*100:.1f}%) signifying urgency/coercion")
        else:
            st.markdown(
                f'<div class="result-safe">✅ ROUTINE / LEGITIMATE EMAIL (SAFE)<br>'
                f'<span style="font-size:1.1rem; font-weight:normal;">Routine Confidence: <b>{(1-prob)*100:.1f}%</b></span></div>',
                unsafe_allow_html=True
            )
            st.progress(float(1 - prob))
            st.success("✨ No suspicious phishing triggers or malicious patterns found in email content.")
