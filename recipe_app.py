import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and vectorizer
model_path = "C:/Users/HP/Downloads/recipe_model.pkl"
vectorizer_path = "C:/Users/HP/Downloads/recipe_vectorizer.pkl"
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

st.title("Recipe Recommendation System")

recipe_name = st.text_input("Enter Recipe Name:")

if st.button("Predict"):
    if recipe_name:
        # Prepare text for prediction
        X_input = vectorizer.transform([recipe_name])
        prediction = model.predict(X_input)
        
        # Display result
        st.write("Predicted Instruction:", prediction[0])
    else:
        st.write("Please enter a recipe name.")
