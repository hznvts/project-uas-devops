import streamlit as st
import pandas as pd
import requests

# URL backend API (FastAPI)
import os

API_URL = os.environ.get("API_URL")  # ambil dari environment variable




# Load data untuk dropdown
df = pd.read_csv("flights.csv")
df = df[['OP_CARRIER','ORIGIN','DEST']].dropna()

# ===== Streamlit UI =====
st.title("✈️ [Update v2.0] Flight Delay Prediction (Microservice)")

carrier = st.selectbox("Select Carrier", sorted(df['OP_CARRIER'].unique()))
origin = st.selectbox("Select Origin Airport", sorted(df['ORIGIN'].unique()))
dest = st.selectbox("Select Destination Airport", sorted(df['DEST'].unique()))
distance = st.number_input("Distance (miles)", min_value=50, max_value=5000, value=500)

if st.button("Predict"):
    # Kirim request ke backend FastAPI
    payload = {
        "carrier": carrier,
        "origin": origin,
        "dest": dest,
        "distance": distance
    }
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        if result["prediction"] == 1:
            st.error("Prediksi: ✈️ Delay")
        else:
            st.success("Prediksi: ✅ On Time")
    else:
        st.error("❌ Error dari server: " + str(response.text))
