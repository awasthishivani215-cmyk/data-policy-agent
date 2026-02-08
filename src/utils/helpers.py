import json
import hashlib
from datetime import datetime

def generate_policy_id(policy_text, company_name):
    content_hash = hashlib.md5(policy_text.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d")
    return f"POL_{company_name.upper()}_{timestamp}_{content_hash}"

def validate_policy_structure(policy_dict):
    required_fields = ['policy_text', 'effective_date', 'company']
    missing_fields = [field for field in required_fields if field not in policy_dict]
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    if len(policy_dict.get('policy_text', '')) < 100:
        return False, "Policy text too short (minimum 100 characters)"
    
    return True, "Valid policy structure"

def format_compliance_score(score):
    if score >= 90:
        return "Excellent"
    elif score >= 80:
        return "Good"
    elif score >= 70:
        return "Fair"
    elif score >= 60:
        return "Poor"
    else:
        return "Non-compliant"

def save_analysis_results(results, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    return filename