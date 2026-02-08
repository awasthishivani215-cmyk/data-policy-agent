import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FeatureEngineer:
    def __init__(self, max_features=1000):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.count_vectorizer = CountVectorizer(max_features=500, stop_words='english')
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.lsa = TruncatedSVD(n_components=50)
        
    def extract_text_features(self, text_series):
        tfidf_features = self.tfidf_vectorizer.fit_transform(text_series)
        count_features = self.count_vectorizer.fit_transform(text_series)
        
        lsa_features = self.lsa.fit_transform(tfidf_features)
        
        return lsa_features
    
    def extract_numeric_features(self, df):
        numeric_features = []
        
        for text in df['policy_text']:
            features = []
            features.append(len(text.split()))
            features.append(len(text.split('.')))
            features.append(np.mean([len(word) for word in text.split()]))
            features.append(text.count('data'))
            features.append(text.count('privacy'))
            features.append(text.count('collection'))
            features.append(text.count('share'))
            features.append(text.count('right'))
            features.append(text.count('security'))
            numeric_features.append(features)
        
        return np.array(numeric_features)
    
    def extract_compliance_features(self, text):
        features = []
        
        keywords_sets = {
            'data_collection': ['collect', 'gather', 'obtain', 'acquire'],
            'third_party': ['third party', 'partner', 'affiliate', 'share with'],
            'user_rights': ['access', 'delete', 'modify', 'rectify', 'opt-out'],
            'security': ['encrypt', 'secure', 'protect', 'firewall', 'authentication'],
            'retention': ['retain', 'store', 'keep', 'period', 'duration']
        }
        
        text_lower = text.lower()
        
        for key, keywords in keywords_sets.items():
            present = any(keyword in text_lower for keyword in keywords)
            features.append(1 if present else 0)
        
        retention_match = re.search(r'(\d+)\s*(day|month|year)s?', text_lower)
        if retention_match:
            num = int(retention_match.group(1))
            unit = retention_match.group(2)
            if unit == 'year':
                features.append(num * 365)
            elif unit == 'month':
                features.append(num * 30)
            else:
                features.append(num)
        else:
            features.append(0)
        
        return np.array(features)
    
    def encode_categorical(self, labels):
        return self.label_encoder.fit_transform(labels)
    
    def scale_features(self, features):
        return self.scaler.fit_transform(features)