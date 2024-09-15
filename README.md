                     Recipe Recommendation System

Overview

The Recipe Recommendation System is a machine learning application designed to recommend recipes based on user input. It uses text data from recipes to predict and suggest instructions for cooking based on the recipe names provided. This system employs natural language processing (NLP) techniques and machine learning algorithms to deliver accurate and personalized recipe recommendations.

Features:

Recipe Recommendation: Predicts recipe instructions based on the names of recipes provided.

Text Preprocessing: Cleans and normalizes text data for better performance.

Model Training: Uses logistic regression to train a model on recipe data.

Evaluation Metrics: Assesses model performance using classification metrics.

Persistence: Saves and loads the trained model and vectorizer for future use.

Technologies Used :

Programming Language: Python

Machine Learning Libraries:
             scikit-learn: For implementing logistic regression and TF-IDF vectorization.
             pandas: For data manipulation and preprocessing.
             numpy: For numerical operations.
             
Model Persistence:
joblib: For saving and loading model artifacts.
