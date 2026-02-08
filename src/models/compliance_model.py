from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class ComplianceModel:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'naive_bayes': MultinomialNB()
        }
        self.model = self.models.get(model_type, RandomForestClassifier())
        self.feature_importance = None
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        return scores.mean(), scores.std()
    
    def tune_hyperparameters(self, X_train, y_train, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_
    
    def save_model(self, filepath):
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        self.model = joblib.load(filepath)
        return self