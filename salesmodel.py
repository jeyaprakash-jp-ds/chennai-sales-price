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

# Define the Streamlit UI
st.title("🏡 Chennai House Price Prediction")
st.write("Enter property details to predict the sales price.")

# Encoding mappings (Ensure categorical data is properly mapped)
area_mapping = {"Anna Nagar": 0, "Karapakkam": 1, "Adyar": 2, "Velachery": 3, "Chrompet": 4, "KK Nagar": 5, "T Nagar": 6}
build_type_mapping = {"Commercial": 0, "House": 1, "Others": 2}
street_mapping = {"Paved": 0, "Gravel": 1, "No Access": 2}
utility_mapping = {"AllPub": 0, "No sewage": 1, "ELO": 2}
sale_cond_mapping = {"Normal Sale": 0, "AbNormal": 1, "AdjLand": 2, "Partial": 3}

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
    "BUILDT
