import pandas as pd
import numpy as np
from datetime import datetime
from .text_processor import TextProcessor

class ComplaintAnalyzer:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.complaint_data = pd.DataFrame()
        
    def load_complaints(self, complaints_df):
        self.complaint_data = complaints_df
        return self
    
    def analyze_complaint(self, complaint_text):
        cleaned_text = self.text_processor.clean_text(complaint_text)
        tokens = self.text_processor.tokenize_text(cleaned_text)
        
        features = {
            'word_count': len(tokens),
            'unique_words': len(set(tokens)),
            'avg_word_length': np.mean([len(word) for word in tokens]) if tokens else 0,
            'has_urgent_words': any(word in ['immediately', 'urgent', 'emergency', 'asap'] for word in tokens),
            'has_emotional_words': any(word in ['angry', 'frustrated', 'disappointed', 'violated'] for word in tokens)
        }
        
        sentiment_score = self._calculate_sentiment_score(tokens)
        
        analysis_result = {
            'complaint_text': complaint_text[:500] + '...' if len(complaint_text) > 500 else complaint_text,
            'sentiment_score': sentiment_score,
            'features': features,
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return analysis_result
    
    def _calculate_sentiment_score(self, tokens):
        positive_words = ['good', 'excellent', 'satisfied', 'happy', 'helpful', 'resolved']
        negative_words = ['bad', 'poor', 'terrible', 'horrible', 'angry', 'frustrated', 'violated']
        
        positive_count = sum(1 for word in tokens if word in positive_words)
        negative_count = sum(1 for word in tokens if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0.5
        
        return negative_count / total_sentiment_words
    
    def generate_complaint_report(self, complaints_df=None):
        if complaints_df is None:
            complaints_df = self.complaint_data
        
        report = {
            'total_complaints': len(complaints_df),
            'category_distribution': {},
            'severity_distribution': {},
            'trends': {}
        }
        
        if 'date_received' in complaints_df.columns:
            try:
                report['date_range'] = {
                    'start': complaints_df['date_received'].min(),
                    'end': complaints_df['date_received'].max()
                }
            except:
                report['date_range'] = {'start': 'N/A', 'end': 'N/A'}
        
        if 'category' in complaints_df.columns:
            report['category_distribution'] = complaints_df['category'].value_counts().to_dict()
        
        if 'severity' in complaints_df.columns:
            report['severity_distribution'] = complaints_df['severity'].value_counts().to_dict()
        
        if 'date_received' in complaints_df.columns:
            try:
                complaints_df['month'] = pd.to_datetime(complaints_df['date_received']).dt.to_period('M')
                monthly_counts = complaints_df.groupby('month').size()
                report['trends']['monthly_counts'] = monthly_counts.astype(str).to_dict()
            except:
                report['trends']['monthly_counts'] = {}
        
        report['common_issues'] = self._find_common_issues(complaints_df)
        
        return report
    
    def _find_common_issues(self, complaints_df, top_n=10):
        if 'description' not in complaints_df.columns:
            return []
        
        all_text = ' '.join(complaints_df['description'].astype(str))
        cleaned = self.text_processor.clean_text(all_text)
        tokens = self.text_processor.tokenize_text(cleaned)
        tokens_no_stopwords = self.text_processor.remove_stopwords(tokens)
        
        freq_dist = self.text_processor.get_word_frequency(tokens_no_stopwords)
        
        sorted_freq = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)
        
        return [{'word': word, 'frequency': freq} for word, freq in sorted_freq[:top_n]]