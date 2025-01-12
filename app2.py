import streamlit as st
import joblib
import numpy as np
import os

# Load model and preprocessing files
try:
    model = joblib.load(os.path.join(os.getcwd(), 'yield_prediction_model.pkl'))
    scaler = joblib.load(os.path.join(os.getcwd(), 'scaler.pkl'))
    label_encoders = joblib.load(os.path.join(os.getcwd(), 'label_encoders.pkl'))
    rmse = joblib.load(os.path.join(os.getcwd(), 'rmse.pkl'))
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or files: {e}")
    st.stop()

# Title and inputs
st.title("Crop Yield Prediction")
st.write("Enter the farm and crop details below to get a yield prediction.")

# Inputs for each feature
crop_type = st.selectbox("Crop Type", label_encoders['Crop_Type'].classes_)
irrigation_type = st.selectbox("Irrigation Type", label_encoders['Irrigation_Type'].classes_)
soil_type = st.selectbox("Soil Type", label_encoders['Soil_Type'].classes_)
season = st.selectbox("Season", label_encoders['Season'].classes_)
farm_area = st.number_input("Farm Area (acres)", min_value=0.0)
fertilizer_used = st.number_input("Fertilizer Used (tons)", min_value=0.0)
pesticide_used = st.number_input("Pesticide Used (kg)", min_value=0.0)
water_usage = st.number_input("Water Usage (cubic meters)", min_value=0.0)

# Predict yield
if st.button("Predict Yield"):
    try:
        # Encode and scale inputs
        inputs = np.array([
            label_encoders['Crop_Type'].transform([crop_type])[0],
            label_encoders['Irrigation_Type'].transform([irrigation_type])[0],
            label_encoders['Soil_Type'].transform([soil_type])[0],
            label_encoders['Season'].transform([season])[0],
            farm_area, fertilizer_used, pesticide_used, water_usage
        ]).reshape(1, -1)

        inputs[:, 4:] = scaler.transform(inputs[:, 4:])
        
        # Make prediction
        prediction = model.predict(inputs)
        st.write(f"Predicted Crop Yield: {prediction[0]:.2f} tons")
        st.write(f"Error (RMSE): {rmse}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
