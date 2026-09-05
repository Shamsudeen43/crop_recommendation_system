import streamlit as st
import pandas as pd
import joblib


# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="Shamsudeen Crop Recommendation",
    page_icon="🌱",
    layout="centered"
)


# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

model = joblib.load("crop_recommender.joblib")


# ------------------------------------------------
# APPLICATION TITLE
# ------------------------------------------------

st.title("🌱 Shamsudeen Agro Solutions")

st.header(
    "AI Crop Recommendation System"
)

st.write(
    """
    Enter the soil and environmental conditions
    below to receive the most suitable crop
    recommendation.
    """
)


# ------------------------------------------------
# INPUT FORM
# ------------------------------------------------

with st.form("crop_recommendation_form"):

    st.subheader("Soil Information")

    col1, col2 = st.columns(2)

    with col1:

        N = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            max_value=200.0,
            value=50.0
        )

        P = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            max_value=150.0,
            value=40.0
        )

        K = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            max_value=150.0,
            value=40.0
        )

        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            step=0.01
        )

    with col2:

        soil = st.selectbox(
            "Soil Texture",
            [
                "sandy loam",
                "loamy",
                "clay",
                "loamy clay"
            ]
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=100.0
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=25.0
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )


    submit = st.form_submit_button(
        "🌱 Recommend Crop"
    )


# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

if submit:

    input_data = pd.DataFrame([
        {
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
            "soil": soil
        }
    ])


prediction = model.predict(input_data)[0]

crop_name = str(prediction).replace("_", " ").title()

st.success(
    f"🌱 Primary Recommended Crop: **{crop_name}**"
)

if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(input_data)[0]

    if hasattr(model, "classes_"):
        classes = model.classes_

    elif hasattr(model, "named_steps"):
        classes = None

        for step_name, step_model in model.named_steps.items():
            if hasattr(step_model, "classes_"):
                classes = step_model.classes_
                break

    else:
        classes = None

    if classes is not None:

        recommendations = sorted(
            zip(classes, probabilities),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        st.subheader("🏆 Top 3 Crop Recommendations")

        for crop, probability in recommendations:

            crop_name = str(crop).replace("_", " ").title()

            st.write(
                f"**{crop_name}** — {probability:.1%}"
            )

            st.progress(float(probability))

st.caption(
    "Shamsudeen
    Agro Solutions — "
    "AI decision-support prototype"
)
