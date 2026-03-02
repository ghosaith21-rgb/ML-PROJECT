import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("rf_model.pkl")

st.set_page_config(page_title="Wine Quality Predictor", layout="centered")

st.title("🍷 Wine Quality Prediction App")
st.write("Enter wine characteristics to predict its quality.")

st.sidebar.header("Input Features")

def user_input():
    fixed_acidity = st.sidebar.number_input("Fixed Acidity", 0.0, 20.0, 7.0)
    volatile_acidity = st.sidebar.number_input("Volatile Acidity", 0.0, 2.0, 0.5)
    citric_acid = st.sidebar.number_input("Citric Acid", 0.0, 2.0, 0.3)
    residual_sugar = st.sidebar.number_input("Residual Sugar", 0.0, 20.0, 2.0)
    chlorides = st.sidebar.number_input("Chlorides", 0.0, 1.0, 0.05)
    free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", 0.0, 100.0, 15.0)
    total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", 0.0, 300.0, 46.0)
    density = st.sidebar.number_input("Density", 0.9900, 1.0050, 0.9968)
    pH = st.sidebar.number_input("pH", 2.5, 4.5, 3.3)
    sulphates = st.sidebar.number_input("Sulphates", 0.0, 2.0, 0.6)
    alcohol = st.sidebar.number_input("Alcohol", 0.0, 20.0, 10.0)
    type_wine = st.sidebar.selectbox("Type of Wine", ("Red", "White"))

    Type_White_Wine = 1 if type_wine == "White" else 0

    data = {
    "fixed acidity": fixed_acidity,
    "volatile acidity": volatile_acidity,
    "citric acid": citric_acid,
    "residual sugar": residual_sugar,
    "chlorides": chlorides,
    "free sulfur dioxide": free_sulfur_dioxide,
    "total sulfur dioxide": total_sulfur_dioxide,
    "density": density,
    "pH": pH,
    "sulphates": sulphates,
    "alcohol": alcohol,
    "Type_White Wine": Type_White_Wine
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input()

st.subheader("Input Summary")
st.write(input_df)

if st.button("Predict Quality"):
    prediction = model.predict(input_df)
    quality_output = "Low Quality" if prediction[0] ==0 else "High Quality"
    st.success(f"Predicted Wine Quality: {quality_output}")
