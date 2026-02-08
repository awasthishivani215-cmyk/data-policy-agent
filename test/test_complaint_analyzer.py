import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.complaint_analyzer import ComplaintAnalyzer
import pandas as pd

class TestComplaintAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = ComplaintAnalyzer()
        
        self.sample_complaint = """
        I am very frustrated with how my data was handled.
        The company shared my personal information without consent.
        This is a serious violation of my privacy rights.
        """
        
        self.sample_df = pd.DataFrame({
            'complaint_id': [1, 2, 3],
            'description': ['Data breach', 'Unauthorized sharing', 'Poor security'],
            'category': ['security', 'privacy', 'security'],
            'date_received': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'severity': ['high', 'medium', 'low']
        })
    
    def test_analyze_complaint(self):
        result = self.analyzer.analyze_complaint(self.sample_complaint)
        
        self.assertIn('sentiment_score', result)
        self.assertIn('features', result)
        self.assertIn('complaint_text', result)
        
        self.assertIsInstance(result['sentiment_score'], float)
        self.assertGreaterEqual(result['sentiment_score'], 0)
        self.assertLessEqual(result['sentiment_score'], 1)
    
    def test_generate_complaint_report(self):
        self.analyzer.load_complaints(self.sample_df)
        report = self.analyzer.generate_complaint_report()
        
        self.assertIn('total_complaints', report)
        self.assertIn('category_distribution', report)
        self.assertIn('severity_distribution', report)
        
        self.assertEqual(report['total_complaints'], 3)
    
    def test_calculate_sentiment_score(self):
        result = self.analyzer.analyze_complaint("I am very happy with the service")
        self.assertIsInstance(result['sentiment_score'], float)
        
        result2 = self.analyzer.analyze_complaint("I am very angry and frustrated")
        self.assertIsInstance(result2['sentiment_score'], float)

if __name__ == '__main__':
    unittest.main()