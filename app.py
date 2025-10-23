import streamlit as st
import pandas as pd
import joblib

# Load saved models
log_model = joblib.load("logistic_model.pkl")
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

st.title("IBD Risk Prediction")
st.write("Enter patient data below:")

# 22 feature names
feature_names = [
    "WHEAT(CHAPATI,ROTI,NAAN,DALIA,RAWA/SOOJI,SEVIYAAN)",
    "WHEAT FREE CEREALS",
    "FRUITS",
    "OTHER VEGETABLES",
    "STARCHY(POTATO,SWEET PATATO,ARBI ETC)",
    "PULSES AND LEGUMES",
    "PREDOMINANT SATURATED FATS",
    "PREDOMINANT UNSATURATED FATS",
    "TRANS FATS",
    "NUTS AND OILSEEDS",
    "LOW LACTOSE DAIRY",
    "SWEETEND BEVERAGES",
    "ULTRA PROCESSED FOODS",
    "READT TO EAT PACKAGED SNACKS",
    "SAVORY SNACKS",
    "PROCESSED FOODS",
    "INDIAN SWEET MEATS",
    "FOOD SUPPLEMENTS",
    "ERGOGENIC SUPPLEMENTS"
]

# Collect inputs
features = {}
for feature in feature_names:
    features[feature] = st.number_input(f"{feature}", value=0.0)

input_df = pd.DataFrame([features])

# Predict button
if st.button("Predict"):
    logistic_prob = log_model.predict_proba(input_df)[0][1]
    rf_prob = rf_model.predict_proba(input_df)[0][1]
    xgb_prob = xgb_model.predict_proba(input_df)[0][1]

    st.write("### Prediction Probabilities")
    st.write(f"Logistic Regression: {logistic_prob:.2f}")
    st.write(f"Random Forest: {rf_prob:.2f}")
    st.write(f"XGBoost: {xgb_prob:.2f}")
