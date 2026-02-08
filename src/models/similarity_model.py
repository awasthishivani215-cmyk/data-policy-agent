from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from difflib import SequenceMatcher

class SimilarityModel:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.policy_vectors = None
        self.policy_texts = []
        
    def train(self, policy_texts):
        self.policy_texts = policy_texts
        self.policy_vectors = self.tfidf.fit_transform(policy_texts)
        
    def find_similar_policies(self, new_policy, threshold=0.8, top_n=5):
        new_vector = self.tfidf.transform([new_policy])
        similarities = cosine_similarity(new_vector, self.policy_vectors)[0]
        
        similar_indices = np.where(similarities >= threshold)[0]
        similar_policies = []
        
        for idx in similar_indices[:top_n]:
            similar_policies.append({
                'index': idx,
                'similarity': float(similarities[idx]),
                'text': self.policy_texts[idx]
            })
        
        return sorted(similar_policies, key=lambda x: x['similarity'], reverse=True)
    
    def detect_changes(self, old_policy, new_policy):
        old_sections = self._extract_sections(old_policy)
        new_sections = self._extract_sections(new_policy)
        
        changes = []
        
        all_sections = set(list(old_sections.keys()) + list(new_sections.keys()))
        
        for section_name in all_sections:
            old_content = old_sections.get(section_name, "")
            new_content = new_sections.get(section_name, "")
            
            if old_content != new_content:
                similarity = SequenceMatcher(None, old_content, new_content).ratio()
                
                if not old_content:
                    change_type = "added"
                elif not new_content:
                    change_type = "removed"
                else:
                    change_type = "modified"
                
                changes.append({
                    'section': section_name,
                    'change_type': change_type,
                    'similarity': similarity,
                    'old_length': len(old_content),
                    'new_length': len(new_content)
                })
        
        return changes
    
    def _extract_sections(self, text):
        sections = {}
        lines = text.split('\n')
        current_section = "general"
        section_content = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 0:
                if line.isupper() or (':' in line and len(line) < 100):
                    if section_content:
                        sections[current_section] = ' '.join(section_content)
                    current_section = line.lower().replace(':', '').strip()
                    section_content = []
                else:
                    section_content.append(line)
        
        if section_content:
            sections[current_section] = ' '.join(section_content)
        
        return sections
    
    def calculate_overall_similarity(self, text1, text2):
        vector1 = self.tfidf.transform([text1])
        vector2 = self.tfidf.transform([text2])
        similarity_matrix = cosine_similarity(vector1, vector2)
        return similarity_matrix[0][0] if similarity_matrix.size > 0 else 0.0