import streamlit as st
import joblib
import numpy as np

st.title("🏨 Hotel Booking Cancellation Prediction")
st.write("Predict whether a booking will be canceled or not.")

# Load model
model = joblib.load("booking_cancellation_model.pkl")

# Input fields
lead_time = st.number_input("Lead Time (days before check-in)", min_value=0)
avg_price = st.number_input("Average Room Price", min_value=0.0)
special_requests = st.number_input("Number of Special Requests", min_value=0)
week_nights = st.number_input("Number of Week Nights", min_value=0)
weekend_nights = st.number_input("Number of Weekend Nights", min_value=0)

if st.button("Predict"):
    X = np.array([[lead_time, avg_price, special_requests, week_nights, weekend_nights]])
    pred = model.predict(X)[0]

    if pred == 1:
        st.error("❌ Booking likely to be **Canceled**")
    else:
        st.success("✅ Booking likely to be **Not Canceled**")
