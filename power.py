import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="⚡ PowerPulse Energy Predictor",
    page_icon="⚡",
    layout="centered"
)

# Load model, imputer, and feature names
with open("model.pkl", "rb") as f:
    model, imputer, feature_names = pickle.load(f)

# Sidebar content
with st.sidebar:
    st.markdown("<h1 style='color:white;'>⚡ PowerPulse</h1>", unsafe_allow_html=True)
    st.markdown("""
        <p>Welcome to the <b>PowerPulse Energy Predictor</b>!  
        Enter your household parameters and get a reliable prediction of the global active power consumption.</p>
    """, unsafe_allow_html=True)

# Title and instructions
st.title("⚡ PowerPulse: Energy Usage Prediction")
st.write("Enter your parameters below to predict Global Active Power.")

# Input form
cols = st.columns(2)
user_inputs = {}
for i, feature in enumerate(feature_names):
    with cols[i % 2]:
        user_inputs[feature] = st.number_input(
            label=feature.replace('_', ' ').title(),
            min_value=0.0,
            step=0.01,
            format="%.3f",
            value=0.0
        )

# Prediction logic
if st.button("🔮 Predict Global Active Power"):
    input_df = pd.DataFrame([user_inputs])[feature_names]
    input_imputed = imputer.transform(input_df)
    prediction = model.predict(input_imputed)[0]

    st.markdown(
        f'<div style="margin-top: 20px; padding: 20px; background: linear-gradient(135deg, #f6d365, #fda085); border-radius: 15px; font-size: 24px; font-weight: bold; text-align: center;">⚡ Predicted Global Active Power: {prediction:.4f} kW</div>',
        unsafe_allow_html=True
    )

# Footer
st.markdown(
    "<div style='text-align:center; margin-top:3rem; font-style: italic; color:#777;'>Powered by Streamlit & Scikit-learn</div>",
    unsafe_allow_html=True
)
