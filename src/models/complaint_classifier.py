from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

class ComplaintClassifier:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
        self.label_binarizer = MultiLabelBinarizer()
        self.severity_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        
    def prepare_complaint_features(self, complaints_text):
        tfidf_features = self.tfidf.fit_transform(complaints_text)
        return tfidf_features
    
    def train_category_classifier(self, X_train, y_train):
        y_train_binarized = self.label_binarizer.fit_transform(y_train)
        self.classifier.fit(X_train, y_train_binarized)
        
    def train_severity_classifier(self, X_train, severity_labels):
        self.severity_classifier.fit(X_train, severity_labels)
    
    def predict_categories(self, X):
        predictions = self.classifier.predict(X)
        return self.label_binarizer.inverse_transform(predictions)
    
    def predict_severity(self, X):
        return self.severity_classifier.predict(X)
    
    def predict_all(self, complaint_text):
        features = self.tfidf.transform([complaint_text])
        categories = self.predict_categories(features)
        severity = self.predict_severity(features)
        
        if categories:
            return categories[0], severity[0]
        return [], 'medium'
    
    def save_models(self, base_path):
        joblib.dump(self.tfidf, f"{base_path}_tfidf.pkl")
        joblib.dump(self.classifier, f"{base_path}_classifier.pkl")
        joblib.dump(self.label_binarizer, f"{base_path}_label_binarizer.pkl")
        joblib.dump(self.severity_classifier, f"{base_path}_severity_classifier.pkl")
    
    def load_models(self, base_path):
        self.tfidf = joblib.load(f"{base_path}_tfidf.pkl")
        self.classifier = joblib.load(f"{base_path}_classifier.pkl")
        self.label_binarizer = joblib.load(f"{base_path}_label_binarizer.pkl")
        self.severity_classifier = joblib.load(f"{base_path}_severity_classifier.pkl")