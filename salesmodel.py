import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor

def load_model():
    with open("salesmodel.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def load_encoder():
    with open("encoder.pkl", "rb") as file:
        encoder = pickle.load(file)
    return encoder

# Load the trained model and encoder
model = load_model()
encoder = load_encoder()

# Set background image
def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background: url('https://source.unsplash.com/1600x900/?house,architecture') no-repeat center center fixed;
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
set_bg()

# Define the Streamlit UI
st.title("🏡 Chennai House Price Prediction")
st.write("Enter property details to predict the sales price.")

# User input fields
area = st.selectbox("Select Area", ["Anna Nagar", "Karapakkam", "Adyar", "Velachery", "Chrompet", "KK Nagar", "T Nagar"])
build_type = st.selectbox("Building Type", ["Commercial", "House", "Others"])
street = st.selectbox("Street Type", ["Paved", "Gravel", "No Access"])
utility = st.selectbox("Utility Availability", ["AllPub", "No sewage", "ELO"])
sale_cond = st.selectbox("Sale Condition", ["Normal Sale", "AbNormal", "AdjLand", "Partial"])
is_parking = st.radio("Parking Facility", ["Yes", "No"])

int_features = {
    "AREA": [area],
    "BUILDTYPE": [build_type],
    "STREET": [street],
    "UTILITY_AVAIL": [utility],
    "SALE_COND": [sale_cond],
    "PARK_FACIL": [1 if is_parking == "Yes" else 0],
    "DATE_SALE_year": [st.number_input("Sale Year", min_value=2000, max_value=2025, value=2022)],
    "DATE_SALE_month": [st.slider("Sale Month", 1, 12, 6)],
    "DATE_SALE_day": [st.slider("Sale Day", 1, 31, 15)],
    "DATE_BUILD_year": [st.number_input("Build Year", min_value=1950, max_value=2025, value=2005)],
    "DATE_BUILD_month": [st.slider("Build Month", 1, 12, 1)],
    "DATE_BUILD_day": [st.slider("Build Day", 1, 31, 1)]
}

df_input = pd.DataFrame(int_features)

# Encode categorical variables
df_input[["AREA", "BUILDTYPE", "STREET", "UTILITY_AVAIL", "SALE_COND"]] = encoder.transform(df_input[["AREA", "BUILDTYPE", "STREET", "UTILITY_AVAIL", "SALE_COND"]])

# Predict
if st.button("Predict Price 💰"):
    prediction = model.predict(df_input)[0]
    st.success(f"🏠 Estimated House Price: ₹ {prediction:,.2f}")
