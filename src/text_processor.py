import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import textstat

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_sections(self, text):
        sections = {}
        lines = text.split('\n')
        current_section = "general"
        section_content = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 0:
                if line.isupper() or ':' in line or line.endswith(':'):
                    if section_content:
                        sections[current_section] = ' '.join(section_content)
                    current_section = line.lower().replace(':', '').strip()
                    section_content = []
                else:
                    section_content.append(line)
        
        if section_content:
            sections[current_section] = ' '.join(section_content)
        
        return sections
    
    def calculate_readability(self, text):
        return textstat.flesch_reading_ease(text)
    
    def tokenize_text(self, text):
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]
    
    def get_word_frequency(self, tokens):
        freq_dist = {}
        for token in tokens:
            freq_dist[token] = freq_dist.get(token, 0) + 1
        return freq_dist