import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🚀",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    background-color: #2563eb;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    color: white;
}

.prediction-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #111827;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
    color: #22c55e;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model_loaded = False

# ---------------- HEADER ----------------
st.title("🚀 Machine Learning Prediction App")
st.markdown("### Smart Prediction System using Streamlit")

st.write(
    "Deploy your machine learning model easily using Streamlit."
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ App Info")
st.sidebar.info(
    "Enter feature values and get instant predictions from your ML model."
)

# ---------------- INPUT SECTION ----------------
st.subheader("📊 Enter Input Features")

col1, col2 = st.columns(2)

with col1:
    feature1 = st.number_input("Feature 1", value=0.0)
    feature2 = st.number_input("Feature 2", value=0.0)
    feature3 = st.number_input("Feature 3", value=0.0)

with col2:
    feature4 = st.number_input("Feature 4", value=0.0)
    feature5 = st.number_input("Feature 5", value=0.0)
    feature6 = st.number_input("Feature 6", value=0.0)

# ---------------- DATAFRAME ----------------
input_data = pd.DataFrame(
    [[feature1, feature2, feature3,
      feature4, feature5, feature6]],
    columns=[
        "feature1",
        "feature2",
        "feature3",
        "feature4",
        "feature5",
        "feature6"
    ]
)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict"):

    if model_loaded:

        try:
            prediction = model.predict(input_data)

            st.markdown(
                f"""
                <div class="prediction-box">
                    Prediction Result: {prediction[0]}
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
