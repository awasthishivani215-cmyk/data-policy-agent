import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.policy_checker import PolicyChecker

class TestPolicyChecker(unittest.TestCase):
    def setUp(self):
        self.checker = PolicyChecker()
        self.sample_policy = """
        PRIVACY POLICY
        
        DATA COLLECTION:
        We collect personal information from users.
        
        CONSENT:
        By using our service, you consent to data collection.
        
        THIRD PARTY SHARING:
        We do not share data with third parties.
        
        USER RIGHTS:
        Users have the right to access their data.
        
        SECURITY:
        We use encryption to protect user data.
        """
    
    def test_check_policy_compliance(self):
        result = self.checker.check_policy_compliance(self.sample_policy)
        
        self.assertIn('compliance_percentage', result)
        self.assertIn('rule_violations', result)
        self.assertIn('warnings', result)
        
        self.assertGreaterEqual(result['compliance_percentage'], 0)
        self.assertLessEqual(result['compliance_percentage'], 100)
    
    def test_extract_sections(self):
        from src.text_processor import TextProcessor
        processor = TextProcessor()
        sections = processor.extract_sections(self.sample_policy)
        
        self.assertIsInstance(sections, dict)
        self.assertGreater(len(sections), 0)
    
    def test_empty_policy(self):
        result = self.checker.check_policy_compliance("")
        self.assertEqual(result['compliance_percentage'], 0)

if __name__ == '__main__':
    unittest.main()