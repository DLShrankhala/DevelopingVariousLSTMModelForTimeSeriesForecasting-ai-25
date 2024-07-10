import streamlit as st

# Title
st.title("AQI Value Prediction")

# Function to determine status based on AQI value
def get_status(aqi_value):
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# Predict AQI Status
st.subheader("Predict AQI Status")
aqi_value = st.number_input("Enter AQI Value", min_value=0.0, max_value=500.0, step=0.1)

if st.button("Predict"):
    status = get_status(aqi_value)
    st.success(f"The AQI status is: {status}")
