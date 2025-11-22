import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000)
        self.model = LogisticRegression(max_iter=1000)
    
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def train(self, csv_path='data.csv'):
        # Load data
        df = pd.read_csv(csv_path)
        df['text'] = df['text'].fillna('')
        
        # Clean text
        df['clean'] = df['text'].apply(self.clean_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['clean'], df['label'], test_size=0.2, random_state=42
        )
        
        # Vectorize
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_vec)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
        print("\n", classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        # Save
        joblib.dump(self.model, 'model.pkl')
        joblib.dump(self.vectorizer, 'vectorizer.pkl')
        print("\n✓ Model saved!")
    
    def predict(self, text):
        clean = self.clean_text(text)
        vec = self.vectorizer.transform([clean])
        pred = self.model.predict(vec)[0]
        prob = self.model.predict_proba(vec)[0]
        
        return {
            'prediction': 'FAKE' if pred == 1 else 'REAL',
            'confidence': float(max(prob) * 100)
        }

# Create sample data if needed
def create_sample_data():
    data = {
        'text': [
            'Scientists discover new species in rainforest with extensive research',
            'Stock market shows steady growth according to financial reports',
            'New vaccine passes clinical trials with promising results',
            'Technology company announces quarterly earnings report',
            'International summit addresses climate change policies',
            'Shocking aliens spotted downtown! You won\'t believe this!',
            'Miracle cure doctors hate! Click now before deleted!',
            'Secret government conspiracy finally revealed to public!',
            'Lose 50 pounds in one week with this weird trick!',
            'Unbelievable truth they don\'t want you to know!'
        ],
        'label': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    }
    pd.DataFrame(data).to_csv('data.csv', index=False)
    print("✓ Sample data created: data.csv")

# Usage
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'create':
        create_sample_data()
    else:
        detector = FakeNewsDetector()
        try:
            detector.train()
        except FileNotFoundError:
            print("Run: python fake_news_detector.py create")