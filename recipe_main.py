import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

# Load dataset
data_path = "C:/Users/HP/Downloads/recipe.csv.xlsx"
data = pd.read_excel(data_path)

# Prepare dataset
data['TranslatedRecipeName'] = data['TranslatedRecipeName'].astype(str).str.lower()
data['TranslatedRecipeName'] = data['TranslatedRecipeName'].str.replace('[^a-zA-Z\s]', '', regex=True)

# Vectorize text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['TranslatedRecipeName'])
y = data['TranslatedInstructions']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model and vectorizer
model_path = "C:/Users/HP/Downloads/recipe_model.pkl"
vectorizer_path = "C:/Users/HP/Downloads/recipe_vectorizer.pkl"
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("Model and vectorizer saved successfully.")

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Test saving and loading
loaded_model = joblib.load(model_path)
loaded_vectorizer = joblib.load(vectorizer_path)

print("Model and vectorizer loaded successfully.")
