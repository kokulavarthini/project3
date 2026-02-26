import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
    return " ".join(filtered_tokens)

def train_model():
    # Load dataset
    df = pd.read_csv('job_dataset.csv')
    
    # Preprocess
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['processed_text']).toarray()
    y = df['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Save model and vectorizer
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

def predict_fraud(text):
    if not os.path.exists('model.pkl') or not os.path.exists('tfidf.pkl'):
        train_model()
        
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
        
    processed = preprocess_text(text)
    vectorized = tfidf.transform([processed]).toarray()
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0][prediction]
    
    return "Fake" if prediction == 1 else "Real", float(probability)

if __name__ == '__main__':
    train_model()
