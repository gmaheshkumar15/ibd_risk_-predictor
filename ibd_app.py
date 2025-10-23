import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained models
# -----------------------------
# NOTE: Ensure these files ('logistic_model.pkl', 'rf_model.pkl', 'xgb_model.pkl') 
# are in the same directory as your Streamlit script when you run the app.
try:
    log_model = joblib.load("logistic_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    # For safety, initialize feature_names from one of the models
    feature_names = list(log_model.feature_names_in_) 
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'logistic_model.pkl', 'rf_model.pkl', and 'xgb_model.pkl' are available.")
    # Fallback for local development if models aren't present. 
    feature_names = [f"Feature_{i}" for i in range(10)] 


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="IBD Risk Prediction", layout="wide")

# -----------------------------
# Custom CSS (RE-ATTEMPT: Using !important on the generic class selector)
# -----------------------------
st.markdown("""
    <style>
    /* Main background color for the app */
    .stApp { background-color: #FFFF99; } 

    /* Center text in the number input boxes */
    .stNumberInput>div>input { text-align: center; } 
    
    /* FIX: Target the generic Streamlit label for the input and force size/color */
    .stNumberInput label {
        font-weight: bold !important;
        font-size: 22px !important; /* FORCED LARGE SIZE */
        color: #000000 !important; /* FORCED PURE BLACK */
        text-align: center;
        width: 100%; 
        display: block; 
        margin-bottom: 5px; 
    }
    
    /* Style for the actual number input box border */
    .stNumberInput input[type="number"] {
        border: 2px solid black; 
        border-radius: 5px; 
        padding: 5px 10px; 
    }

    /* Logo styling */
    .logo-left { float: left; width: 120px; }
    .logo-right { float: right; width: 120px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Logos and Title
# -----------------------------
col_logo_left, col_title, col_logo_right = st.columns([1, 5, 1])

with col_logo_left:
    st.markdown('<img src="https://brandlogovector.com/wp-content/uploads/2022/04/IIT-Delhi-Icon-Logo.png" class="logo-left">', unsafe_allow_html=True)

with col_title:
    st.markdown(
        "<h1 style='text-align:center; font-size:48px; color:black;'>Inflammatory Bowel Disease Risk Prediction Tool</h1>",
        unsafe_allow_html=True
    )

with col_logo_right:
    st.markdown('<img src="https://tse2.mm.bing.net/th/id/OIP.fNb1hJAUj-8vwANfP3SDJgAAAA?pid=Api&P=0&h=180" class="logo-right">', unsafe_allow_html=True)

# -----------------------------
# Feature names and cleaning utility
# -----------------------------
def clean_feature_name(name):
    """Converts 'feature_name_example' to 'Feature Name Example'."""
    return name.replace("_", " ").title()

# -----------------------------
# Layout input and output
# -----------------------------
col_input, col_output = st.columns([3,2])

# -------- Left column: Input features --------
features = {}
with col_input.container():
    st.header("Enter Patient Data ")
    n = len(feature_names)
    half = n // 2

    for i in range(half):
        # Create two columns for side-by-side inputs
        c1, c2 = st.columns(2, gap="medium") 
        
        # Column 1: Feature i
        with c1:
            # The label argument attaches the clean feature name directly above the input box
            features[feature_names[i]] = st.number_input(
                label=clean_feature_name(feature_names[i]),
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                key=f"{feature_names[i]}"
            )
            
        # Column 2: Feature i + half
        with c2:
            # The label argument attaches the clean feature name directly above the input box
            features[feature_names[i+half]] = st.number_input(
                label=clean_feature_name(feature_names[i+half]),
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                key=f"{feature_names[i+half]}"
            )

# Create the input DataFrame from the gathered features
input_df = pd.DataFrame([features], columns=feature_names)

# -------- Right column: Predictions --------
with col_output.container():
    st.header("Predictions")
    # Button is placed outside the 'if predict_clicked' block so it is always visible
    predict_clicked = st.button("Predict")

    if predict_clicked:
        # --- Predict probabilities ---
        try:
            logistic_prob = log_model.predict_proba(input_df)[0][1]
            rf_prob = rf_model.predict_proba(input_df)[0][1]
            xgb_prob = xgb_model.predict_proba(input_df)[0][1]
        except Exception as e:
            st.error(f"Prediction Error: Could not run models. Check your model files and input features. Details: {e}")
            logistic_prob, rf_prob, xgb_prob = 0.5, 0.5, 0.5 # Default to medium risk on error
            
        # --- Probabilities & risk categories side by side ---
        prob_col, risk_col = st.columns(2)
        
        with prob_col:
            st.subheader("Probabilities")
            st.write(f"**Logistic Regression:** {logistic_prob:.2f}")
            st.write(f"**Random Forest:** {rf_prob:.2f}")
            st.write(f"**XGBoost:** {xgb_prob:.2f}")

        with risk_col:
            st.subheader("Risk Categories")
            
            def colored_risk_label(prob):
                """Returns HTML formatted risk label based on probability."""
                if prob < 0.33:
                    return f"<span style='color:white;background-color:green;padding:5px;border-radius:5px;'>Low</span>"
                elif prob < 0.66:
                    return f"<span style='color:black;background-color:yellow;padding:5px;border-radius:5px;'>Medium</span>"
                else:
                    return f"<span style='color:white;background-color:red;padding:5px;border-radius:5px;'>High</span>"

            st.markdown("**Logistic Regression:** " + colored_risk_label(logistic_prob), unsafe_allow_html=True)
            st.markdown("**Random Forest:** " + colored_risk_label(rf_prob), unsafe_allow_html=True)
            st.markdown("**XGBoost:** " + colored_risk_label(xgb_prob), unsafe_allow_html=True)