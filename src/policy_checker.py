import numpy as np
from .text_processor import TextProcessor
from .feature_engineer import FeatureEngineer

class PolicyChecker:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.feature_engineer = FeatureEngineer()
        self.policy_rules = self._load_default_rules()
        
    def _load_default_rules(self):
        return {
            'data_collection': {
                'required': True,
                'keywords': ['collect', 'gather', 'obtain'],
                'weight': 1.0
            },
            'consent': {
                'required': True,
                'keywords': ['consent', 'permission', 'agree'],
                'weight': 1.0
            },
            'third_party_sharing': {
                'required': True,
                'keywords': ['third party', 'share with', 'partner'],
                'weight': 1.0
            },
            'user_rights': {
                'required': True,
                'keywords': ['access', 'delete', 'modify', 'right'],
                'weight': 1.0
            },
            'security': {
                'required': True,
                'keywords': ['encrypt', 'secure', 'protect'],
                'weight': 1.0
            }
        }
    
    def check_policy_compliance(self, policy_text):
        results = {
            'overall_score': 0,
            'rule_violations': [],
            'warnings': [],
            'compliance_percentage': 0,
            'section_analysis': {}
        }
        
        sections = self.text_processor.extract_sections(policy_text)
        
        total_weight = sum(rule['weight'] for rule in self.policy_rules.values())
        achieved_weight = 0
        
        for rule_name, rule in self.policy_rules.items():
            rule_found = False
            for section_name, section_text in sections.items():
                section_lower = section_text.lower()
                
                keyword_count = sum(1 for keyword in rule['keywords'] if keyword in section_lower)
                
                if keyword_count > 0:
                    rule_found = True
                    results['section_analysis'][f"{rule_name}_{section_name}"] = {
                        'found': True,
                        'keyword_count': keyword_count,
                        'section': section_name
                    }
            
            if rule['required'] and not rule_found:
                results['rule_violations'].append({
                    'rule': rule_name,
                    'issue': f"Required section '{rule_name}' not found",
                    'severity': 'high'
                })
            elif rule_found:
                achieved_weight += rule['weight']
        
        results['compliance_percentage'] = (achieved_weight / total_weight) * 100 if total_weight > 0 else 0
        
        readability_score = self.text_processor.calculate_readability(policy_text)
        results['readability'] = readability_score
        
        if readability_score < 30:
            results['warnings'].append({
                'issue': 'Policy is difficult to read',
                'suggestion': 'Simplify language for better user understanding',
                'severity': 'medium'
            })
        
        features = self.feature_engineer.extract_compliance_features(policy_text)
        results['features'] = features.tolist()
        
        return results