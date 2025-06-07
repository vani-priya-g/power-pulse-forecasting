import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(
    page_title="⚡ PowerPulse Energy Predictor",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #212121;
    }
    .card {
        background: #ffffffdd;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
        border: 3px solid #3b82f6;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #f7971e, #ffd200);
        color: #000;
        font-weight: 700;
        padding: 10px 25px;
        border-radius: 30px;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #ffd200, #f7971e);
        color: #222;
        cursor: pointer;
    }
    .css-1d391kg {
        background: linear-gradient(180deg, #6a11cb 0%, #2575fc 100%);
        color: white;
    }
    input[type="number"] {
        border: 2px solid #3b82f6 !important;
        border-radius: 8px;
        padding: 7px;
        font-size: 16px;
        font-weight: 600;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        border-radius: 20px;
        padding: 30px;
        font-size: 28px;
        font-weight: 700;
        color: #442c00;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar content
with st.sidebar:
    st.markdown("<h1 style='color:white;'>⚡ PowerPulse</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <p>Welcome to the <b>PowerPulse Energy Predictor</b>!  
        Enter your household parameters and get a reliable prediction of the global active power consumption.</p>  
        <br>
        """,
        unsafe_allow_html=True,
    )

# --- Main Title and Description ---
st.title("⚡ PowerPulse: Energy Usage Prediction")
st.write(
    "Welcome to the energy prediction tool. Input your parameters below and get an accurate forecast of Global Active Power."
)

# --- Instruction Card ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.write("Fill in the parameters below and click **Predict** to see the estimated Global Active Power.")
st.markdown("</div>", unsafe_allow_html=True)

# --- Load Model ---
with open("model.pkl", "rb") as f:
    model, imputer, feature_names = pickle.load(f)

# --- Input Form ---
st.markdown('<div class="card">', unsafe_allow_html=True)
cols = st.columns(2)
user_inputs = {}
for i, feature in enumerate(feature_names):
    with cols[i % 2]:
        user_inputs[feature] = st.number_input(
            label=f"{feature.replace('_', ' ').title()}",
            min_value=0.0,
            step=0.01,
            format="%.3f",
            value=0.0,
            key=feature,
        )
st.markdown("</div>", unsafe_allow_html=True)

# --- Prediction ---
if st.button("🔮 Predict Global Active Power"):
    input_array = np.array([[user_inputs[feat] for feat in feature_names]])
    input_imputed = imputer.transform(input_array)
    prediction = model.predict(input_imputed)[0]

    st.markdown(
        f'<div class="prediction-box">⚡ Predicted Global Active Power: {prediction:.4f} kW</div>',
        unsafe_allow_html=True,
    )

# --- Footer (optional) ---
st.markdown(
    """
    <div style="text-align:center; margin-top:3rem; font-style: italic; color:#555;">
    Powered by Streamlit & Scikit-learn
    </div>
    """,
    unsafe_allow_html=True,
)
 