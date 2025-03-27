import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor

def load_model():
    with open("salesmodel.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Load the trained model
model = load_model()

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

# Define categorical mappings (ensure consistency with model training)
area_mapping = {"Anna Nagar": 0, "Karapakkam": 1, "Adyar": 2, "Velachery": 3, "Chrompet": 4, "KK Nagar": 5, "T Nagar": 6}
build_type_mapping = {"Commercial": 0, "House": 1, "Others": 2}
street_mapping = {"Paved": 0, "Gravel": 1, "No Access": 2}
utility_mapping = {"AllPub": 0, "No sewage": 1, "ELO": 2}
sale_cond_mapping = {"Normal Sale": 0, "AbNormal": 1, "AdjLand": 2, "Partial": 3}

# Define the Streamlit UI
st.title("🏡 Chennai House Price Prediction")
st.write("Enter property details to predict the sales price.")

# User input fields
area = st.selectbox("Select Area", list(area_mapping.keys()))
build_type = st.selectbox("Building Type", list(build_type_mapping.keys()))
street = st.selectbox("Street Type", list(street_mapping.keys()))
utility = st.selectbox("Utility Availability", list(utility_mapping.keys()))
sale_cond = st.selectbox("Sale Condition", list(sale_cond_mapping.keys()))
is_parking = st.radio("Parking Facility", ["Yes", "No"])

# Convert user input into numerical format
int_features = {
    "AREA": [area_mapping[area]],
    "BUILDTYPE": [build_type_mapping[build_type]],
    "STREET": [street_mapping[street]],
    "UTILITY_AVAIL": [utility_mapping[utility]],
    "SALE_COND": [sale_cond_mapping[sale_cond]],
    "PARK_FACIL": [1 if is_parking == "Yes" else 0],
    "DATE_SALE_year": [st.number_input("Sale Year", min_value=2000, max_value=2025, value=2022)],
    "DATE_SALE_month": [st.slider("Sale Month", 1, 12, 6)],
    "DATE_SALE_day": [st.slider("Sale Day", 1, 31, 15)],
    "DATE_BUILD_year": [st.number_input("Build Year", min_value=1950, max_value=2025, value=2005)],
    "DATE_BUILD_month": [st.slider("Build Month", 1, 12, 1)],
    "DATE_BUILD_day": [st.slider("Build Day", 1, 31, 1)]
}

df_input = pd.DataFrame(int_features)

# Ensure feature order matches training data
expected_columns = ["AREA", "BUILDTYPE", "STREET", "UTILITY_AVAIL", "SALE_COND", "PARK_FACIL", 
                    "DATE_SALE_year", "DATE_SALE_month", "DATE_SALE_day", 
                    "DATE_BUILD_year", "DATE_BUILD_month", "DATE_BUILD_day"]
df_input = df_input[expected_columns]  # Reorder columns

# Predict
if st.button("Predict Price 💰"):
    prediction = model.predict(df_input)[0]
    st.success(f"🏠 Estimated House Price: ₹ {prediction:,.2f}")
