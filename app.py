import os
import sys
import re
import pickle
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

# Free Uptime / Keep-Alive Monitor Endpoint (returns fast response for UptimeRobot / cron-job.org)
if "ping" in st.query_params or "health" in st.query_params or "uptime" in st.query_params:
    st.write("🟢 App is active and running.")
    st.stop()

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

FEATURE_COLS = [
    'message_length',
    'url_count',
    'attachment_count',
    'keyword_indicators',
    'domain_age_days',
    'suspicious_tld',
    'spf_dkim_passed',
    'uppercase_ratio',
    'has_external_links'
]

HIGH_RISK_KEYWORDS = [
    'urgent', 'urgently', 'verify', 'verification', 'account', 'bank', 'banking',
    'login', 'password', 'credential', 'credentials', 'suspended', 'suspension',
    'deactivation', 'action required', 'security alert', 'security notice',
    'click here', 'claim', 'prize', 'winner', 'wire transfer', 'update required',
    'update your account', 'unauthorized', 'confirm your', 'access restricted',
    'ssn', 'social security', 'credit card', 'gift card', 'bitcoin', 'crypto'
]

HIGH_RISK_TLDS = ['.xyz', '.top', '.work', '.ru', '.cc', '.info', '.tk', '.click', '.site', '.club', '.online', '.vip']

def extract_features_from_raw_text(text: str) -> dict:
    if not text or not text.strip():
        return None
    clean_text = text.strip()
    message_length = len(clean_text)
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', clean_text)
    urls_www = re.findall(r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', clean_text)
    url_count = len(set(urls + urls_www))
    
    suspicious_tld = 0
    for tld in HIGH_RISK_TLDS:
        if tld in clean_text.lower():
            suspicious_tld = 1
            break
            
    text_lower = clean_text.lower()
    keyword_indicators = 0
    for kw in HIGH_RISK_KEYWORDS:
        if kw in text_lower:
            keyword_indicators += text_lower.count(kw)
            
    alpha_chars = [c for c in clean_text if c.isalpha()]
    if alpha_chars:
        uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    else:
        uppercase_ratio = 0.0
        
    attachment_patterns = [r'\.exe', r'\.zip', r'\.rar', r'\.pdf', r'\.doc[x]?', r'attached', r'attachment']
    attachment_count = 0
    for pat in attachment_patterns:
        attachment_count += len(re.findall(pat, text_lower))
        
    if 'spf=pass' in text_lower or 'dkim=pass' in text_lower or 'authentication-results: pass' in text_lower:
        spf_dkim_passed = 1
    elif 'spf=fail' in text_lower or 'dkim=fail' in text_lower or 'spf=softfail' in text_lower:
        spf_dkim_passed = 0
    else:
        spf_dkim_passed = 0 if (suspicious_tld == 1 or keyword_indicators >= 3) else 1
        
    if suspicious_tld == 1 or keyword_indicators >= 4:
        domain_age_days = 45.0
    else:
        domain_age_days = 1200.0
        
    has_external_links = 1 if url_count > 0 else 0
    
    return {
        'message_length': float(message_length),
        'url_count': int(url_count),
        'attachment_count': int(attachment_count),
        'keyword_indicators': int(keyword_indicators),
        'domain_age_days': float(domain_age_days),
        'suspicious_tld': int(suspicious_tld),
        'spf_dkim_passed': int(spf_dkim_passed),
        'uppercase_ratio': float(np.round(uppercase_ratio, 4)),
        'has_external_links': int(has_external_links)
    }

# Helper function to load model and vectorizer/scaler from model/
@st.cache_resource
def get_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model', 'model.pkl')
    vec_path = os.path.join(base_dir, 'model', 'vectorizer.pkl')
    
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, 'models', 'best_model.joblib')
        vec_path = os.path.join(base_dir, 'models', 'scaler.joblib')
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vec_path, 'rb') as f:
        scaler = pickle.load(f)
        
    meta = {'best_model_name': 'Logistic Regression'}
    return model, scaler, meta

try:
    model, scaler, meta = get_artifacts()
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
        
        input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=FEATURE_COLS)
            
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
