import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Setup page layout
st.set_page_config(
    page_title="Email Threat Classification | CyberShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling with custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Fira+Code:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: #ffffff;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 25px;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #4facfe;
        margin: 0;
    }
    
    .main-subtitle {
        font-size: 1.05rem;
        color: #cfd8dc;
        margin-top: 6px;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    .status-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: #000000;
        font-weight: 800;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.3);
    }
    
    .status-threat {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
        color: #ffffff;
        font-weight: 800;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        box-shadow: 0 4px 15px rgba(239, 71, 58, 0.3);
    }
    
    .subtext {
        font-size: 0.88rem;
        color: #90a4ae;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model and scaler
@st.cache_resource
def get_artifacts():
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.data_preprocessing import FEATURE_COLS
    from src.predict import load_inference_pipeline
    model, scaler, meta = load_inference_pipeline()
    return model, scaler, meta, FEATURE_COLS

try:
    model, scaler, meta, FEATURE_COLS = get_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model artifacts: {e}")

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">🛡️ CyberShield AI: Email Threat Classifier</div>
    <div class="main-subtitle">Zero-Cost End-to-End Machine Learning Cyber Threat Detection & Security Analytics Prototype</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🔮 Real-Time Threat Predictor", "📊 Batch Threat Scanner", "📈 EDA & Model Performance", "📖 Technical Paper & Spec"])

# ----------------- TAB 1: Real-Time Predictor -----------------
with tabs[0]:
    st.subheader("Interactive Email Observation Input")
    st.markdown("Adjust measurable characteristics of the incoming email to evaluate real-time threat probability.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📧 Message & Structure")
        message_length = st.number_input("Message Length (Characters)", min_value=10, max_value=10000, value=650, step=50, help="Total character count in email body.")
        url_count = st.number_input("URL Count", min_value=0, max_value=30, value=1, step=1, help="Number of hyperlinks contained in message.")
        attachment_count = st.number_input("Attachment Count", min_value=0, max_value=10, value=0, step=1, help="Count of attached files.")
        
    with col2:
        st.markdown("#### ⚠️ Keywords & Content")
        keyword_indicators = st.slider("High-Risk Keyword Count", min_value=0, max_value=15, value=0, help="Frequency of trigger words e.g., 'urgent', 'verify account', 'bank', 'suspended'.")
        uppercase_ratio = st.slider("Uppercase Ratio", min_value=0.0, max_value=1.0, value=0.05, step=0.01, help="Ratio of uppercase characters in text.")
        spf_dkim = st.selectbox("SPF / DKIM Authentication", options=["Passed (1)", "Failed / Missing (0)"], index=0)
        spf_dkim_passed = 1 if "Passed" in spf_dkim else 0

    with col3:
        st.markdown("#### 🌐 Domain & Sender Characteristics")
        domain_age_days = st.number_input("Sender Domain Age (Days)", min_value=1, max_value=10000, value=1200, step=30, help="Age of the sending domain in days.")
        suspicious_tld_choice = st.selectbox("Sender TLD Category", options=["Standard (.com, .org, .edu)", "High-Risk (.xyz, .top, .work, .ru)"], index=0)
        suspicious_tld = 1 if "High-Risk" in suspicious_tld_choice else 0
        has_external_links = 1 if url_count > 0 else 0

    st.markdown("---")
    
    if st.button("🚀 Analyze Email Threat Risk", use_container_width=True, type="primary"):
        input_data = {
            'message_length': float(message_length),
            'url_count': int(url_count),
            'attachment_count': int(attachment_count),
            'keyword_indicators': int(keyword_indicators),
            'domain_age_days': float(domain_age_days),
            'suspicious_tld': int(suspicious_tld),
            'spf_dkim_passed': int(spf_dkim_passed),
            'uppercase_ratio': float(uppercase_ratio),
            'has_external_links': int(has_external_links)
        }
        
        input_df = pd.DataFrame([input_data])[FEATURE_COLS]
        if meta.get('requires_scaling', True):
            input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=FEATURE_COLS)
        else:
            input_df_scaled = input_df
            
        pred = model.predict(input_df_scaled)[0]
        prob = model.predict_proba(input_df_scaled)[0][1]
        
        st.markdown("### 🎯 Classification Results & Security Assessment")
        
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if pred == 1:
                st.markdown(f'<div class="status-threat">⚠️ POTENTIALLY THREATENING<br><span style="font-size:0.9rem; font-weight:normal;">Phishing / Malicious Email Detected</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-safe">✅ ROUTINE (SAFE)<br><span style="font-size:0.9rem; font-weight:normal;">Legitimate Business Communication</span></div>', unsafe_allow_html=True)
                
        with res_col2:
            st.metric("Calculated Threat Probability", f"{prob*100:.2f}%")
            st.progress(float(prob))
            st.markdown(f"**Selected Model:** `{meta.get('best_model_name', 'Logistic Regression')}`")

        # Feature contribution insights
        st.markdown("#### 🔍 Threat Factor Breakdown")
        factors = []
        if suspicious_tld == 1:
            factors.append("🚨 Sender domain uses a high-risk suspicious TLD extension.")
        if spf_dkim_passed == 0:
            factors.append("🚨 Email failed SPF/DKIM domain ownership authentication.")
        if domain_age_days < 90:
            factors.append(f"🚨 Sender domain is extremely new ({int(domain_age_days)} days old).")
        if keyword_indicators >= 3:
            factors.append(f"🚨 High count of phishing trigger keywords detected ({keyword_indicators}).")
        if url_count >= 3:
            factors.append(f"🚨 Elevated number of embedded hyperlinks ({url_count}).")
        if uppercase_ratio > 0.20:
            factors.append(f"🚨 High uppercase text proportion ({uppercase_ratio*100:.1f}%) indicates urgency/coercion.")
            
        if factors:
            for f in factors:
                st.error(f)
        else:
            st.success("✨ No suspicious anomaly triggers found in email structure or sender domain.")

# ----------------- TAB 2: Batch Scanner -----------------
with tabs[1]:
    st.subheader("📁 Batch Email CSV Scanner")
    st.markdown("Upload a CSV file containing email record observations to generate a threat report.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_df)}** records.")
            
            missing_cols = [c for c in FEATURE_COLS if c not in batch_df.columns]
            if missing_cols:
                st.warning(f"Missing required columns in uploaded CSV: {missing_cols}")
            else:
                if meta.get('requires_scaling', True):
                    scaled_batch = scaler.transform(batch_df[FEATURE_COLS])
                    batch_preds = model.predict(scaled_batch)
                    batch_probs = model.predict_proba(scaled_batch)[:, 1]
                else:
                    batch_preds = model.predict(batch_df[FEATURE_COLS])
                    batch_probs = model.predict_proba(batch_df[FEATURE_COLS])[:, 1]
                    
                batch_df['Predicted_Threat'] = batch_preds
                batch_df['Threat_Probability'] = np.round(batch_probs, 4)
                batch_df['Status'] = np.where(batch_preds == 1, 'Threat', 'Routine')
                
                threat_count = int((batch_preds == 1).sum())
                st.write(f"Scanned {len(batch_df)} emails: **{threat_count}** threats flagged ({threat_count/len(batch_df)*100:.1f}%).")
                st.dataframe(batch_df, use_container_width=True)
                
                csv_bytes = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Scanned Threat Results CSV", csv_bytes, "batch_threat_predictions.csv", "text/csv")
        except Exception as ex:
            st.error(f"Error processing CSV file: {ex}")
    else:
        st.info("Tip: You can test batch scanning by uploading `data/raw_email_threat_dataset.csv` generated by the pipeline!")

# ----------------- TAB 3: EDA & Model Performance -----------------
with tabs[2]:
    st.subheader("📊 Exploratory Data Analysis & Evaluation Metrics")
    
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        st.markdown("#### 🎯 Class Distribution & Key Features")
        if os.path.exists('figures/class_distribution.png'):
            st.image('figures/class_distribution.png', use_container_width=True)
        else:
            st.info("Run model training to generate class distribution plot.")
            
        st.markdown("#### 📉 Confusion Matrices Across Foundational Models")
        if os.path.exists('figures/confusion_matrices.png'):
            st.image('figures/confusion_matrices.png', use_container_width=True)
            
    with fig_col2:
        st.markdown("#### 📈 Receiver Operating Characteristic (ROC) Curves")
        if os.path.exists('figures/roc_curves.png'):
            st.image('figures/roc_curves.png', use_container_width=True)
            
        st.markdown("#### 🔑 Feature Importance Breakdown")
        if os.path.exists('figures/feature_importance.png'):
            st.image('figures/feature_importance.png', use_container_width=True)
            
    st.markdown("---")
    st.markdown("### 🏆 Foundational Model Benchmarking Matrix")
    if os.path.exists('models/model_comparison_results.csv'):
        comp_df = pd.read_csv('models/model_comparison_results.csv', index_col=0)
        st.dataframe(comp_df.style.highlight_max(axis=0, color='#2ecc71'), use_container_width=True)

# ----------------- TAB 4: Technical Paper & Spec -----------------
with tabs[3]:
    st.subheader("📜 Capstone Project Documentation & Technical Paper")
    if os.path.exists('TECHNICAL_PAPER.md'):
        with open('TECHNICAL_PAPER.md', 'r', encoding='utf-8') as f:
            paper_content = f.read()
        st.markdown(paper_content)
    else:
        st.info("Technical paper is available in project files.")
