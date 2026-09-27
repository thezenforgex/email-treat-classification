import re
import numpy as np

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
    """
    Parses raw pasted email text and extracts numerical features for threat classification.
    """
    if not text or not text.strip():
        return None
        
    clean_text = text.strip()
    message_length = len(clean_text)
    
    # 1. URL Count & TLD detection
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', clean_text)
    urls_www = re.findall(r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', clean_text)
    url_count = len(set(urls + urls_www))
    
    # Check suspicious TLD
    suspicious_tld = 0
    for tld in HIGH_RISK_TLDS:
        if tld in clean_text.lower():
            suspicious_tld = 1
            break
            
    # 2. Keyword Indicators Count
    text_lower = clean_text.lower()
    keyword_indicators = 0
    for kw in HIGH_RISK_KEYWORDS:
        if kw in text_lower:
            keyword_indicators += text_lower.count(kw)
            
    # 3. Uppercase Ratio
    alpha_chars = [c for c in clean_text if c.isalpha()]
    if alpha_chars:
        uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    else:
        uppercase_ratio = 0.0
        
    # 4. Attachment References Count
    attachment_patterns = [r'\.exe', r'\.zip', r'\.rar', r'\.pdf', r'\.doc[x]?', r'attached', r'attachment']
    attachment_count = 0
    for pat in attachment_patterns:
        attachment_count += len(re.findall(pat, text_lower))
        
    # 5. SPF / DKIM status check in headers if present
    if 'spf=pass' in text_lower or 'dkim=pass' in text_lower or 'authentication-results: pass' in text_lower:
        spf_dkim_passed = 1
    elif 'spf=fail' in text_lower or 'dkim=fail' in text_lower or 'spf=softfail' in text_lower:
        spf_dkim_passed = 0
    else:
        # Infer based on suspicious indicators
        spf_dkim_passed = 0 if (suspicious_tld == 1 or keyword_indicators >= 3) else 1
        
    # 6. Domain Age (Inferred or Default)
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
