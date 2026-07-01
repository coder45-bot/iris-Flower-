import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('iris_model.pkl')

st.title("Iris Flower Classifier")

# Create input sliders for the 4 features of the Iris flower
sepal_l = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_w = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_l = st.slider("Petal Length", 1.0, 7.0, 3.5)
petal_w = st.slider("Petal Width", 0.1, 2.5, 1.0)

# Make a prediction
if st.button("Predict Species"):
    features = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    prediction = model.predict(features)
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.write(f"The predicted species is: **{species[prediction[0]]}**")