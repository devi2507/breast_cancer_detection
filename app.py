import streamlit as st
import numpy as np
import joblib
model=joblib.load("breast_cancer_svm.pkl")
st.title("Breast cancer prediction")
st.write("Enter the tumor characteristics to predict if it is benign or malignant")
feature_names = [
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
    "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
    "Radius Error", "Texture Error", "Perimeter Error", "Area Error", "Smoothness Error",
    "Compactness Error", "Concavity Error", "Concave Points Error", "Symmetry Error", "Fractal Dimension Error",
    "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness",
    "Worst Compactness", "Worst Concavity", "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
]

# Input fields for all 30 features
features = []
for feature in feature_names:
    value = st.number_input(feature, min_value=0.0, format="%.4f")
    features.append(value)

# Convert inputs into NumPy array
features = np.array([features]).reshape(1, -1)
features = np.array([features]).reshape(1, -1)


if st.button("Predict"):
    prediction = model.predict(features)[0]
    result = "🟢 Benign (1) 😊" if prediction == 1 else "🔴 Malignant (0) 😞"
    st.write(f"### Prediction: {result}")