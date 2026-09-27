import os
import numpy as np
import pandas as pd

def generate_email_threat_dataset(n_samples=3500, random_state=42):
    np.random.seed(random_state)
    
    # 0 = Routine (Ham), 1 = Threat (Phishing/Malicious)
    # Class balance: ~35% threats, ~65% routine emails
    is_threat = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])
    
    # Feature generation with class-conditional distributions
    
    # 1. message_length (chars)
    # Routine: mean ~ 800, std ~ 350
    # Threat: mean ~ 450, std ~ 250 (often shorter, urgent action calls)
    msg_len_routine = np.random.normal(800, 350, size=n_samples)
    msg_len_threat = np.random.normal(480, 220, size=n_samples)
    message_length = np.where(is_threat == 1, msg_len_threat, msg_len_routine)
    message_length = np.clip(message_length, 50, 4500).astype(int)
    
    # 2. url_count
    # Routine: Poisson lambda ~ 0.8
    # Threat: Poisson lambda ~ 3.5
    url_routine = np.random.poisson(lam=0.8, size=n_samples)
    url_threat = np.random.poisson(lam=3.6, size=n_samples)
    url_count = np.where(is_threat == 1, url_threat, url_routine)
    url_count = np.clip(url_count, 0, 15)
    
    # 3. attachment_count
    # Routine: Binomial n=3, p=0.15
    # Threat: Binomial n=4, p=0.45 (executable, malicious PDF/ZIP)
    att_routine = np.random.binomial(n=3, p=0.15, size=n_samples)
    att_threat = np.random.binomial(n=4, p=0.42, size=n_samples)
    attachment_count = np.where(is_threat == 1, att_threat, att_routine)
    
    # 4. keyword_indicators (high risk trigger word counts like 'urgent', 'account', 'verify', 'bank', 'password')
    # Routine: Poisson lambda ~ 0.4
    # Threat: Poisson lambda ~ 4.2
    kw_routine = np.random.poisson(lam=0.4, size=n_samples)
    kw_threat = np.random.poisson(lam=4.2, size=n_samples)
    keyword_indicators = np.where(is_threat == 1, kw_threat, kw_routine)
    keyword_indicators = np.clip(keyword_indicators, 0, 12)
    
    # 5. domain_age_days (sender domain age)
    # Routine: Exponential mean ~ 1800 days (~5 years)
    # Threat: Exponential mean ~ 120 days (newly registered domains)
    dom_age_routine = np.random.exponential(scale=1800, size=n_samples) + 90
    dom_age_threat = np.random.exponential(scale=150, size=n_samples) + 2
    domain_age_days = np.where(is_threat == 1, dom_age_threat, dom_age_routine)
    domain_age_days = np.clip(domain_age_days, 1, 7300).astype(int)
    
    # 6. suspicious_tld (1 if .xyz, .top, .work, .info, .tk, etc.)
    # Routine: 5% chance
    # Threat: 58% chance
    tld_routine = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    tld_threat = np.random.choice([0, 1], size=n_samples, p=[0.42, 0.58])
    suspicious_tld = np.where(is_threat == 1, tld_threat, tld_routine)
    
    # 7. spf_dkim_passed (1 if authentication passed, 0 if failed)
    # Routine: 92% pass
    # Threat: 28% pass
    auth_routine = np.random.choice([0, 1], size=n_samples, p=[0.08, 0.92])
    auth_threat = np.random.choice([0, 1], size=n_samples, p=[0.72, 0.28])
    spf_dkim_passed = np.where(is_threat == 1, auth_threat, auth_routine)
    
    # 8. uppercase_ratio (ratio of capital letters in subject/body)
    # Routine: Beta(a=2, b=25) -> mean ~ 0.07
    # Threat: Beta(a=4, b=12) -> mean ~ 0.25
    uc_routine = np.random.beta(a=2, b=25, size=n_samples)
    uc_threat = np.random.beta(a=4, b=12, size=n_samples)
    uppercase_ratio = np.where(is_threat == 1, uc_threat, uc_routine)
    uppercase_ratio = np.round(np.clip(uppercase_ratio, 0.01, 0.95), 4)
    
    # 9. has_external_links
    has_external_links = (url_count > 0).astype(int)

    df = pd.DataFrame({
        'message_length': message_length,
        'url_count': url_count,
        'attachment_count': attachment_count,
        'keyword_indicators': keyword_indicators,
        'domain_age_days': domain_age_days,
        'suspicious_tld': suspicious_tld,
        'spf_dkim_passed': spf_dkim_passed,
        'uppercase_ratio': uppercase_ratio,
        'has_external_links': has_external_links,
        'is_threat': is_threat
    })
    
    # Inject ~1.5% realistic missing values into numerical columns for data cleaning exercise
    mask_missing_len = np.random.rand(n_samples) < 0.015
    mask_missing_age = np.random.rand(n_samples) < 0.015
    
    df.loc[mask_missing_len, 'message_length'] = np.nan
    df.loc[mask_missing_age, 'domain_age_days'] = np.nan
    
    return df

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    df = generate_email_threat_dataset(n_samples=3500, random_state=42)
    output_path = os.path.join('data', 'raw_email_threat_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated raw dataset with shape {df.shape} saved to {output_path}")
    print(df.head())
    print("\nMissing values summary:")
    print(df.isnull().sum())
    print("\nClass distribution:")
    print(df['is_threat'].value_counts(normalize=True))
