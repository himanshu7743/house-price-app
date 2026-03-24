import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🏠 House Price Prediction App")

# Inputs
area = st.number_input("Area (sq ft)")
bedrooms = st.number_input("Bedrooms")
age = st.number_input("House Age")

# Prediction
if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, age]])
    prediction = model.predict(input_data)
    
    st.success(f"Estimated Price: ₹ {int(prediction[0])}")