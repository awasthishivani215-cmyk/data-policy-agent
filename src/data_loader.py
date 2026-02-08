import os
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import yaml

class DataLoader:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
    def load_policy_document(self, file_path):
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self._read_pdf(file_path)
        elif file_extension == '.docx':
            return self._read_docx(file_path)
        elif file_extension == '.txt':
            return self._read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _read_pdf(self, file_path):
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text
    
    def _read_docx(self, file_path):
        doc = Document(file_path)
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        return '\n'.join(full_text)
    
    def _read_txt(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def load_complaints_data(self, csv_path):
        df = pd.read_csv(csv_path)
        required_columns = ['complaint_id', 'description', 'category', 'date_received', 'severity']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        return df
    
    def load_policies_database(self, csv_path):
        df = pd.read_csv(csv_path)
        required_columns = ['policy_id', 'policy_text', 'version', 'effective_date', 'company']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        return df