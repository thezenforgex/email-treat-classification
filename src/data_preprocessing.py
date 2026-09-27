import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

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

TARGET_COL = 'is_threat'

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw email dataset by handling missing values, removing duplicates,
    and ensuring proper data types.
    """
    cleaned_df = df.copy()
    
    # 1. Deduplication
    cleaned_df = cleaned_df.drop_duplicates()
    
    # 2. Imputation of missing numerical values using median
    for col in ['message_length', 'domain_age_days']:
        if col in cleaned_df.columns:
            median_val = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_val)
            
    # 3. Ensure integer / float types
    cleaned_df['message_length'] = cleaned_df['message_length'].astype(float)
    cleaned_df['url_count'] = cleaned_df['url_count'].astype(int)
    cleaned_df['attachment_count'] = cleaned_df['attachment_count'].astype(int)
    cleaned_df['keyword_indicators'] = cleaned_df['keyword_indicators'].astype(int)
    cleaned_df['domain_age_days'] = cleaned_df['domain_age_days'].astype(float)
    cleaned_df['suspicious_tld'] = cleaned_df['suspicious_tld'].astype(int)
    cleaned_df['spf_dkim_passed'] = cleaned_df['spf_dkim_passed'].astype(int)
    cleaned_df['uppercase_ratio'] = cleaned_df['uppercase_ratio'].astype(float)
    cleaned_df['has_external_links'] = (cleaned_df['url_count'] > 0).astype(int)
    
    return cleaned_df

def prepare_train_test_data(data_path='data/raw_email_threat_dataset.csv', test_size=0.2, random_state=42):
    """
    Loads raw data, applies cleaning, fits standard scaler, and returns split datasets.
    """
    df = pd.read_csv(data_path)
    cleaned_df = clean_data(df)
    
    X = cleaned_df[FEATURE_COLS]
    y = cleaned_df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index)
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    joblib.dump(scaler, 'models/scaler.joblib')
    cleaned_df.to_csv('data/processed_email_threat_dataset.csv', index=False)
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == '__main__':
    X_train, X_test, X_tr_s, X_te_s, y_train, y_test, scaler = prepare_train_test_data()
    print("Preprocessing completed successfully!")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
