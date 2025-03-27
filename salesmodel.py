import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor

def load_model():
    with open("final.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def load_encoder():
    with open("label.pkl", "rb") as file:
        encoder = pickle.load(file)
    return encoder

# Load the trained model and encoder
model = load_model()
encoder = load_encoder()

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

st.title("🏡 Chennai House Price Prediction")
st.write("Enter property details to predict the sales price.")

# User input fields
area = st.selectbox("Select Area", ["Anna Nagar", "Karapakkam", "Adyar", "Velachery", "Chrompet", "KK Nagar", "T Nagar"])
build_type = st.selectbox("Building Type", ["Commercial", "House", "Others"])
street = st.selectbox("Street Type", ["Paved", "Gravel", "No Access"])
utility = st.selectbox("Utility Availability", ["AllPub", "No sewage", "ELO"])
sale_cond = st.selectbox("Sale Condition", ["Normal Sale", "AbNormal", "AdjLand", "Partial"])
is_parking = st.radio("Parking Facility", ["Yes", "No"])

# Date-related fields
sale_year = st.number_input("Sale Year", min_value=2000, max_value=2025, value=2022)
sale_month = st.slider("Sale Month", 1, 12, 6)
sale_day = st.slider("Sale Day", 1, 31, 15)
build_year = st.number_input("Build Year", min_value=1950, max_value=2025, value=2005)

# Calculate age of building
build_age = 2005 - build_year

# Prepare input dictionary
int_features = {
    "AREA": [area],
    "BUILDTYPE": [build_type],
    "STREET": [street],
    "UTILITY_AVAIL": [utility],
    "SALE_COND": [sale_cond],
    "PARK_FACIL": [1 if is_parking == "Yes" else 0],
    "year_date_sale": [sale_year],
    "month_date_sale": [sale_month],
    "day_date_sale": [sale_day],
    "build_age": [build_age]
}

df_input = pd.DataFrame(int_features)

# Encode categorical variables with error handling
for col in ["AREA", "BUILDTYPE", "STREET", "UTILITY_AVAIL", "SALE_COND"]:
    try:
        df_input[col] = df_input[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else 0)
    except ValueError:
        df_input[col] = 0  # Assign default encoding for unseen values

# Ensure correct column order
model_features = model.get_booster().feature_names
df_input = df_input.reindex(columns=model_features, fill_value=0)

df_input = df_input.astype(float)

if st.button("Predict Price 💰"):
    prediction = model.predict(df_input)[0]
    st.success(f"🏠 Estimated House Price: ₹ {prediction:,.2f}")
