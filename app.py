import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest

# -----------------------------
# Load or Train Models
# -----------------------------
@st.cache_resource
def load_models():
    # Load dataset to retrain quickly (or load pre-saved models if you dumped them)
    df = pd.read_csv("card_transdata.csv")
    X = df.drop("fraud", axis=1)
    y = df["fraud"]

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    log_reg.fit(X_scaled, y)

    # Train Isolation Forest (fraud rate ≈ 0.09)
    iso_forest = IsolationForest(contamination=0.09, random_state=42)
    iso_forest.fit(X_scaled)

    return scaler, log_reg, iso_forest

scaler, log_reg, iso_forest = load_models()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Credit Card Fraud Detection App")
st.write("Enter transaction details below and check if it’s likely fraud or legit.")

# Input form
with st.form("fraud_form"):
    distance_from_home = st.number_input("Distance from home", min_value=0.0, step=0.1)
    distance_from_last_transaction = st.number_input("Distance from last transaction", min_value=0.0, step=0.1)
    ratio_to_median_purchase_price = st.number_input("Ratio to median purchase price", min_value=0.0, step=0.01)
    repeat_retailer = st.selectbox("Repeat retailer?", [0, 1])
    used_chip = st.selectbox("Used chip?", [0, 1])
    used_pin_number = st.selectbox("Used PIN number?", [0, 1])
    online_order = st.selectbox("Online order?", [0, 1])
    
    submitted = st.form_submit_button("Check Transaction")

if submitted:
    # Build dataframe for the transaction
    transaction = pd.DataFrame([{
        'distance_from_home': distance_from_home,
        'distance_from_last_transaction': distance_from_last_transaction,
        'ratio_to_median_purchase_price': ratio_to_median_purchase_price,
        'repeat_retailer': repeat_retailer,
        'used_chip': used_chip,
        'used_pin_number': used_pin_number,
        'online_order': online_order
    }])

    # Scale features
    transaction_scaled = scaler.transform(transaction)

    # Logistic Regression prediction
    lr_pred = log_reg.predict(transaction_scaled)[0]
    lr_prob = log_reg.predict_proba(transaction_scaled)[0][1]

    # Isolation Forest prediction
    iso_pred = iso_forest.predict(transaction_scaled)[0]
    iso_pred = 1 if iso_pred == -1 else 0  # map -1 → fraud, 1 → legit

    # Display results
    st.subheader("🔍 Predictions:")
    if lr_pred == 1:
        st.error(f"Logistic Regression: FRAUD DETECTED! (Probability: {lr_prob:.2f})")
    else:
        st.success(f"Logistic Regression: Legit transaction (Fraud probability: {lr_prob:.2f})")

    if iso_pred == 1:
        st.error("Isolation Forest: FRAUD DETECTED!")
    else:
        st.success("Isolation Forest: Legit transaction")
