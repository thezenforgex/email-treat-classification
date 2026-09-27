import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Allow imports when run directly or imported as module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import FEATURE_COLS

def load_inference_pipeline():
    """
    Loads the trained model, scaler, and metadata.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'best_model.joblib')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.joblib')
    meta_path = os.path.join(base_dir, 'models', 'model_metadata.json')
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(meta_path)):
        # Fallback to current working directory
        model_path = os.path.join('models', 'best_model.joblib')
        scaler_path = os.path.join('models', 'scaler.joblib')
        meta_path = os.path.join('models', 'model_metadata.json')
        raise FileNotFoundError("Model artifacts not found. Please run model_training.py first.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    return model, scaler, meta

def predict_email_threat(input_dict: dict):
    """
    Predicts threat status and threat probability for a single email observation dictionary.
    """
    model, scaler, meta = load_inference_pipeline()
    
    # Calculate derived features if missing
    if 'has_external_links' not in input_dict:
        input_dict['has_external_links'] = 1 if input_dict.get('url_count', 0) > 0 else 0
        
    input_df = pd.DataFrame([input_dict])[FEATURE_COLS]
    
    if meta.get('requires_scaling', True):
        scaled_input = scaler.transform(input_df)
        input_df = pd.DataFrame(scaled_input, columns=FEATURE_COLS)
        
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    threat_prob = float(probabilities[1])
    
    if threat_prob < 0.25:
        risk_level = "LOW RISK (Routine)"
    elif threat_prob < 0.60:
        risk_level = "MODERATE RISK (Suspicious)"
    else:
        risk_level = "HIGH RISK (Threat / Malicious)"
        
    return {
        'prediction': prediction,
        'label': 'Potentially Threatening' if prediction == 1 else 'Routine (Safe)',
        'threat_probability': threat_prob,
        'routine_probability': float(probabilities[0]),
        'risk_level': risk_level,
        'model_used': meta.get('best_model_name', 'Foundational Model')
    }

if __name__ == '__main__':
    sample_routine = {
        'message_length': 850,
        'url_count': 1,
        'attachment_count': 0,
        'keyword_indicators': 0,
        'domain_age_days': 1500,
        'suspicious_tld': 0,
        'spf_dkim_passed': 1,
        'uppercase_ratio': 0.05
    }
    
    sample_threat = {
        'message_length': 380,
        'url_count': 4,
        'attachment_count': 2,
        'keyword_indicators': 5,
        'domain_age_days': 12,
        'suspicious_tld': 1,
        'spf_dkim_passed': 0,
        'uppercase_ratio': 0.32
    }
    
    print("Routine Email Test:", predict_email_threat(sample_routine))
    print("Threat Email Test:", predict_email_threat(sample_threat))
