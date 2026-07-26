

import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Crop Yield Category Estimator", page_icon="🌾")

# --- Load model bundle (model + encoders + feature order) from Task 3 ---
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_model()
model = bundle["model"]
encoders = bundle["encoders"]
feature_cols = bundle["feature_cols"]
target_encoder = encoders["yield_category"]

CONFIDENCE_THRESHOLD = 0.5

st.title("🌾 Crop Yield Category Estimator")
st.write(
    "Enter a farm plot's details to estimate whether it is heading for a "
    "**Poor**, **Average**, or **Good** yield this season."
)

# --- Input form ---
with st.form("plot_form"):
    col1, col2 = st.columns(2)

    with col1:
        crop = st.selectbox("Crop", options=list(encoders["crop"].classes_))
        area_acres = st.number_input("Area (acres)", min_value=0.1, max_value=50.0, value=3.0, step=0.1)
        irrigation_type = st.selectbox("Irrigation type", options=list(encoders["irrigation_type"].classes_))

    with col2:
        fertiliser_used = st.selectbox("Fertiliser used?", options=list(encoders["fertiliser_used"].classes_))
        rainfall_mm = st.number_input("Rainfall this season (mm)", min_value=0.0, max_value=3000.0, value=750.0, step=10.0)
        previous_yield = st.selectbox("Previous season's yield", options=list(encoders["previous_yield"].classes_))

    submitted = st.form_submit_button("Predict Yield Category")

if submitted:
    record = {
        "crop": crop,
        "area_acres": area_acres,
        "irrigation_type": irrigation_type,
        "fertiliser_used": fertiliser_used,
        "rainfall_mm": rainfall_mm,
        "previous_yield": previous_yield,
    }

    row = pd.DataFrame([record])
    for col in ["crop", "irrigation_type", "fertiliser_used", "previous_yield"]:
        le = encoders[col]
        row[col] = le.transform([record[col]])

    X = row[feature_cols]
    probs = model.predict_proba(X)[0]
    pred_idx = probs.argmax()
    confidence = probs[pred_idx]
    predicted_label = target_encoder.classes_[pred_idx]

    st.subheader("Result")

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            f"⚠️ **Unsure** (confidence {confidence:.0%}, below the "
            f"{CONFIDENCE_THRESHOLD:.0%} threshold). This plot's details "
            f"don't closely match patterns the model has learned. "
            f"**Refer this case to an extension officer** rather than "
            f"acting on this prediction."
        )
    else:
        if predicted_label == "Poor":
            st.error(f"🔴 Predicted category: **{predicted_label}** (confidence: {confidence:.0%})")
        elif predicted_label == "Average":
            st.info(f"🟡 Predicted category: **{predicted_label}** (confidence: {confidence:.0%})")
        else:
            st.success(f"🟢 Predicted category: **{predicted_label}** (confidence: {confidence:.0%})")

    st.write("**Full probability breakdown:**")
    prob_df = pd.DataFrame({
        "Category": target_encoder.classes_,
        "Probability": [f"{p:.1%}" for p in probs]
    })
    st.table(prob_df)

st.caption(
    "Model: RandomForestClassifier (scikit-learn) · Trained on 80 synthetic "
    "farm plot records · Confidence threshold: 50%"
)
