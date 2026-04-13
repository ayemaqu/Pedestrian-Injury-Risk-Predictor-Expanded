# app.py — run with: streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === PAGE SETUP ===
st.set_page_config(page_title="Pedestrian Injury Model", page_icon="🚦", layout="centered")
st.title("🚦 Pedestrian Injury Risk Predictor")
st.write(
    "This tool uses a trained prediction model to estimate the likelihood that a crash results "
    "in a pedestrian injury. It’s tuned for safety and uses a low threshold (0.07) so we  "
    "don’t miss potential injury cases."
)

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    return joblib.load("calibrated_model.pkl")

model = load_model()

THRESHOLD = 0.07   # safety-optimized threshold

# === USER INPUTS (SIDEBAR) ===
st.sidebar.header("Crash Details")

cf1 = st.sidebar.selectbox(
    "🔴 Primary Cause",
    [
        "Driver Inattention/Distraction",
        "Failure to Yield Right-of-Way",
        "Following Too Closely",
        "Unsafe Speed",
        "Driver Inexperience",
        "Turning Improperly",
        "Alcohol Involvement",
        "Fell Asleep",
        "Traffic Control Disregarded",
        "Aggressive Driving/Road Rage"
    ]
)

hour = st.sidebar.slider("🕥 Hour of Day (0–23)", 0, 23, 12)

veh_group = st.sidebar.selectbox(
    "🚗 Vehicle Type",
    ["Sedan", "SUV", "Truck", "Bus", "Motorcycle", "Van", "Taxi", "Bike"]
)

boro = st.sidebar.selectbox(
    "📍 Borough",
    ["Brooklyn", "Queens", "Manhattan", "Bronx", "Staten Island"]
)

go = st.sidebar.button("Predict")


# === DISPLAY INPUTS ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Inputs")

    # The model still needs these exact column names:
    raw_input = pd.DataFrame([{
        "cf1_clean": cf1,
        "hour": hour,
        "veh_group": veh_group,
        "BoroName": boro
    }])

    # Pretty display labels for stakeholders
    display_df = raw_input.rename(columns={
        "cf1_clean": "Primary Cause",
        "hour": "Hour of Day",
        "veh_group": "Vehicle Type",
        "BoroName": "Borough"
    })

    display_vertical = display_df.T
    display_vertical.columns = [""]  # remove the column name

    st.dataframe(display_vertical, use_container_width=True, hide_index=False)

# === PREDICT ===
with col2:
    st.subheader("Prediction")

    if go:
        proba = model.predict_proba(raw_input)[0][1]
        pred_class = int(proba > THRESHOLD)

        st.write(f"**Predicted injury probability:** {proba:.3f}")

        if pred_class == 1:
            st.error("⚠️ **High risk of pedestrian injury**")
        else:
            st.success("Low risk of injury")

        st.caption(
            f"Threshold = {THRESHOLD}. If probability > {THRESHOLD}, the model flags an injury risk."
        )
    else:
        st.info("Enter details on the left and click **Predict**.")


# === EXPLANATION SECTION ===
st.divider()
st.subheader("Model Background")

st.write("""

This model was trained on past NYPD crash data to learn the conditions that are most commonly linked to pedestrian 
injuries. It looks at factors like vehicle type, driver behavior, time of day, and borough.
         
Because missing an injury risk is more serious than raising a false alert, the model uses a 
low safety threshold (0.07). This helps the tool catch more situations where an injury might occur. 
The output should be used as a screening signal to support decision-making, not as a final determination.
""")
