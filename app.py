"""
Streamlit UI - Vitamin & Mineral Deficiency Disease Predictor

Loads three separate files (no subfolder, all in the same directory as
this script):
    - best_model.pkl        -> trained classifier
    - encoder.pkl            -> LabelEncoder (converts 0-4 back to disease names)
    - feature_columns.pkl    -> exact column order the model was trained on

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -------------------------------------------------------------------
# Page setup
# -------------------------------------------------------------------
st.set_page_config(page_title="Deficiency Disease Predictor", page_icon="🩺", layout="wide")

# Cap the content width on large screens (laptops/desktops) so it doesn't
# stretch edge-to-edge, while still shrinking naturally on mobile.
st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 Vitamin & Mineral Deficiency Predictor")
st.write("Fill in your details below and the model will predict which deficiency you may have.")

# -------------------------------------------------------------------
# Step 1: Load the three saved files
# -------------------------------------------------------------------
model = joblib.load("best_model.pkl")
label_encoder = joblib.load("encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Short, plain-language descriptions for hover tooltips
DISEASE_INFO = {
    "Anemia": "A condition where the blood lacks enough healthy red blood cells or hemoglobin, often caused by low iron or Vitamin B12. Common signs: fatigue, pale skin, weakness.",
    "Healthy": "No clear deficiency pattern was detected based on the values entered.",
    "Night_Blindness": "Difficulty seeing in low light or darkness, most commonly caused by a Vitamin A deficiency.",
    "Rickets_Osteomalacia": "A condition where bones become soft or weak due to a lack of Vitamin D or calcium. In children it's called Rickets, in adults Osteomalacia.",
    "Scurvy": "A condition caused by severe Vitamin C deficiency, leading to bleeding gums, joint pain, and slow wound healing.",
}


# -------------------------------------------------------------------
# Step 2: Basic information (collapsible section)
# -------------------------------------------------------------------
with st.expander("📋 Basic Information", expanded=True):
    st.markdown(
        "<div style='background-color:#4a90d9; padding:8px 14px; border-radius:8px; margin-bottom:10px;'>"
        "<span style='color:white; font-weight:600;'>📋 Basic Information</span></div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=110, value=35)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("**BMI (Body Mass Index)** — calculated from your height and weight, shows whether your weight is healthy for your height.")

    col1, col2 = st.columns(2)
    with col1:
        height_cm = st.number_input("Height (cm)", min_value=100, max_value=220, value=170)
    with col2:
        weight_kg = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)

    bmi = weight_kg / ((height_cm / 100) ** 2)
    bmi = round(bmi)

    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    st.info(f"Your BMI: **{bmi}** ({bmi_category})")


# -------------------------------------------------------------------
# Step 3: Lifestyle & diet (collapsible section)
# -------------------------------------------------------------------
with st.expander("🥗 Lifestyle & Diet"):
    st.markdown(
        "<div style='background-color:#5cb85c; padding:8px 14px; border-radius:8px; margin-bottom:10px;'>"
        "<span style='color:white; font-weight:600;'>🥗 Lifestyle & Diet</span></div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        smoking_status = st.selectbox("Smoking status", ["Never", "Former", "Current"])
    with col2:
        alcohol_consumption = st.selectbox("Alcohol consumption", ["Not_Specified", "Moderate", "Heavy"])

    col1, col2 = st.columns(2)
    with col1:
        exercise_level = st.selectbox("Exercise level", ["Sedentary", "Light", "Moderate", "Active"])
    with col2:
        diet_type = st.selectbox("Diet type", ["Omnivore", "Vegetarian", "Vegan", "Pescatarian"])

    col1, col2 = st.columns(2)
    with col1:
        sun_exposure = st.selectbox(
            "Sun exposure", ["Low", "Moderate", "High"],
            help="How much direct sunlight you get on your skin regularly. Sunlight helps your body make Vitamin D.",
        )
    with col2:
        income_level = st.selectbox("Income level", ["Low", "Middle", "High"])

    latitude_region = st.selectbox(
        "Latitude region", ["Low", "Mid", "High"],
        help="Roughly how far your location is from the equator. Regions farther from the equator (higher latitude) get less sunlight.",
    )


# -------------------------------------------------------------------
# Step 4: Symptoms (collapsible section)
# -------------------------------------------------------------------
with st.expander("🩹 Symptoms (check all that apply)"):
    st.markdown(
        "<div style='background-color:#e07a5f; padding:8px 14px; border-radius:8px; margin-bottom:10px;'>"
        "<span style='color:white; font-weight:600;'>🩹 Symptoms</span></div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        has_night_blindness = st.checkbox("Night blindness", help="Difficulty seeing in low light or at night.")
        has_bleeding_gums = st.checkbox("Bleeding gums", help="Gums bleed easily, e.g. while brushing teeth.")
        has_muscle_weakness = st.checkbox("Muscle weakness", help="Reduced strength in muscles, difficulty with physical tasks.")
        has_memory_problems = st.checkbox("Memory problems", help="Difficulty remembering things or concentrating.")
    with col2:
        has_fatigue = st.checkbox("Fatigue", help="Feeling constantly tired or low on energy.")
        has_bone_pain = st.checkbox("Bone pain", help="Aching or tenderness in the bones.")
        has_numbness_tingling = st.checkbox("Numbness / tingling", help="A pins-and-needles or numb sensation, often in hands or feet.")
        has_pale_skin = st.checkbox("Pale skin", help="Skin looking paler than usual, often linked to low iron/hemoglobin.")

    symptoms_count = sum([
        has_night_blindness, has_fatigue, has_bleeding_gums, has_bone_pain,
        has_muscle_weakness, has_numbness_tingling, has_memory_problems, has_pale_skin,
    ])


# -------------------------------------------------------------------
# Step 5: Lab values (collapsible section)
# -------------------------------------------------------------------
with st.expander("🧪 Lab Values (based on your last blood test, if known)"):
    st.markdown(
        "<div style='background-color:#8e7cc3; padding:8px 14px; border-radius:8px; margin-bottom:10px;'>"
        "<span style='color:white; font-weight:600;'>🧪 Lab Values</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("If you don't have your lab report, just leave these as 'Normal'.")

    LAB_OPTIONS = ["Low", "Normal", "High"]

    RDA_MAP = {"Low": 30, "Normal": 100, "High": 150}
    VITAMIN_C_MAP = {"Low": 25, "Normal": 100, "High": 150}
    VITAMIN_B12_MAP = {"Low": 45, "Normal": 95, "High": 150}

    col1, col2 = st.columns(2)
    with col1:
        vitamin_a_choice = st.selectbox("Vitamin A level", LAB_OPTIONS, index=1,
            help="Vitamin A supports eyesight and the immune system. Low levels can cause night blindness.")
        vitamin_d_choice = st.selectbox("Vitamin D level", LAB_OPTIONS, index=1,
            help="Vitamin D helps your body absorb calcium for strong bones. Low levels can cause rickets or bone pain.")
        vitamin_b12_choice = st.selectbox("Vitamin B12 level", LAB_OPTIONS, index=1,
            help="Vitamin B12 is needed for healthy nerves and red blood cells. Low levels can cause fatigue and numbness.")
        calcium_choice = st.selectbox("Calcium level", LAB_OPTIONS, index=1,
            help="Calcium is needed for strong bones and teeth.")
    with col2:
        vitamin_c_choice = st.selectbox("Vitamin C level", LAB_OPTIONS, index=1,
            help="Vitamin C supports wound healing and healthy gums. Low levels can cause scurvy.")
        vitamin_e_choice = st.selectbox("Vitamin E level", LAB_OPTIONS, index=1,
            help="Vitamin E is an antioxidant that protects your body's cells.")
        folate_choice = st.selectbox("Folate level", LAB_OPTIONS, index=1,
            help="Folate (Vitamin B9) is needed for cell growth and red blood cell production.")
        iron_choice = st.selectbox("Iron level", LAB_OPTIONS, index=1,
            help="Iron is needed to make hemoglobin, which carries oxygen in your blood. Low levels can cause anemia.")

    col1, col2 = st.columns(2)
    with col1:
        hemoglobin_choice = st.selectbox("Hemoglobin level", LAB_OPTIONS, index=1,
            help="Hemoglobin is the protein in red blood cells that carries oxygen. Low levels indicate anemia.")
        serum_vitamin_b12_choice = st.selectbox("Serum Vitamin B12 (blood test)", LAB_OPTIONS, index=1,
            help="The direct blood measurement of Vitamin B12, usually from a lab report (in pg/mL).")
    with col2:
        serum_vitamin_d_choice = st.selectbox("Serum Vitamin D (blood test)", LAB_OPTIONS, index=1,
            help="The direct blood measurement of Vitamin D, usually from a lab report (in ng/mL).")
        serum_folate_choice = st.selectbox("Serum Folate (blood test)", LAB_OPTIONS, index=1,
            help="The direct blood measurement of Folate, usually from a lab report (in ng/mL).")

    HEMOGLOBIN_MAP = {"Low": 9.0, "Normal": 13.8, "High": 16.5}
    SERUM_VIT_D_MAP = {"Low": 9.5, "Normal": 30.0, "High": 60.0}
    SERUM_B12_MAP = {"Low": 180.0, "Normal": 350.0, "High": 700.0}
    SERUM_FOLATE_MAP = {"Low": 9.0, "Normal": 13.0, "High": 20.0}


# -------------------------------------------------------------------
# Step 6: Build the input row and predict
# -------------------------------------------------------------------
if st.button("Predict", type="primary", use_container_width=True):

    row = {col: 0 for col in feature_columns}

    row["age"] = age
    row["bmi"] = bmi
    row["vitamin_a_percent_rda"] = RDA_MAP[vitamin_a_choice]
    row["vitamin_c_percent_rda"] = VITAMIN_C_MAP[vitamin_c_choice]
    row["vitamin_d_percent_rda"] = RDA_MAP[vitamin_d_choice]
    row["vitamin_e_percent_rda"] = RDA_MAP[vitamin_e_choice]
    row["vitamin_b12_percent_rda"] = VITAMIN_B12_MAP[vitamin_b12_choice]
    row["folate_percent_rda"] = RDA_MAP[folate_choice]
    row["calcium_percent_rda"] = RDA_MAP[calcium_choice]
    row["iron_percent_rda"] = RDA_MAP[iron_choice]
    row["hemoglobin_g_dl"] = HEMOGLOBIN_MAP[hemoglobin_choice]
    row["serum_vitamin_d_ng_ml"] = SERUM_VIT_D_MAP[serum_vitamin_d_choice]
    row["serum_vitamin_b12_pg_ml"] = SERUM_B12_MAP[serum_vitamin_b12_choice]
    row["serum_folate_ng_ml"] = SERUM_FOLATE_MAP[serum_folate_choice]
    row["symptoms_count"] = symptoms_count

    row["has_night_blindness"] = int(has_night_blindness)
    row["has_fatigue"] = int(has_fatigue)
    row["has_bleeding_gums"] = int(has_bleeding_gums)
    row["has_bone_pain"] = int(has_bone_pain)
    row["has_muscle_weakness"] = int(has_muscle_weakness)
    row["has_numbness_tingling"] = int(has_numbness_tingling)
    row["has_memory_problems"] = int(has_memory_problems)
    row["has_pale_skin"] = int(has_pale_skin)

    row[f"gender_{gender}"] = 1
    row[f"smoking_status_{smoking_status}"] = 1
    row[f"alcohol_consumption_{alcohol_consumption}"] = 1
    row[f"exercise_level_{exercise_level}"] = 1
    row[f"diet_type_{diet_type}"] = 1
    row[f"sun_exposure_{sun_exposure}"] = 1
    row[f"income_level_{income_level}"] = 1
    row[f"latitude_region_{latitude_region}"] = 1

    input_df = pd.DataFrame([row])[feature_columns]

    # -----------------------------------------------------------------
    # Step 7: Predict
    # -----------------------------------------------------------------
    probabilities = model.predict_proba(input_df)[0]

    predicted_index = probabilities.argmax()
    predicted_disease = label_encoder.classes_[predicted_index]
    confidence = probabilities[predicted_index]

    # -----------------------------------------------------------------
    # Step 8: Show the result
    # -----------------------------------------------------------------
    st.subheader("Result")

    disease_description = DISEASE_INFO.get(predicted_disease, "")

    if predicted_disease == "Healthy":
        st.success(f"Prediction: **{predicted_disease}** (confidence: {confidence:.0%})")
    else:
        st.warning(f"Prediction: **{predicted_disease}** (confidence: {confidence:.0%})")

    # Small hover-info icon next to the predicted disease name, explaining what it is
    st.markdown(
        f"<span title=\"{disease_description}\" style='cursor:help; font-size:14px;'>"
        f"ℹ️ Hover here to learn what <b>{predicted_disease}</b> means</span>",
        unsafe_allow_html=True,
    )

    result_df = pd.DataFrame({
        "Disease": label_encoder.classes_,
        "Probability": probabilities,
    }).sort_values("Probability", ascending=False).reset_index(drop=True)

    display_names = {
        "Healthy": "No Deficiency Pattern",
        "Anemia": "Anemia",
        "Scurvy": "Scurvy",
        "Rickets_Osteomalacia": "Rickets / Osteomalacia",
        "Night_Blindness": "Night Blindness",
    }
    result_df["Display"] = result_df["Disease"].map(display_names)
    result_df["Percent"] = (result_df["Probability"] * 100).round(1)

    # Attach the plain-language description to each row, so hovering over
    # any bar in the chart shows what that disease actually means
    result_df["Description"] = result_df["Disease"].map(DISEASE_INFO)

    # Each disease gets its own distinct color (not a gradient)
    category_colors = {
        "Anemia": "#e63946",
        "No Deficiency Pattern": "#2a9d8f",
        "Night Blindness": "#f4a261",
        "Rickets / Osteomalacia": "#8e7cc3",
        "Scurvy": "#e07a5f",
    }

    fig = px.bar(
        result_df,
        x="Percent",
        y="Display",
        orientation="h",
        text="Percent",
        color="Display",
        color_discrete_map=category_colors,
        labels={"Percent": "Confidence (%)", "Display": ""},
        title="Prediction confidence across all possibilities (hover over a bar for details)",
        custom_data=["Description"],
    )
    fig.update_traces(
        texttemplate="%{text}%",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Confidence: %{x}%<br><br>%{customdata[0]}<extra></extra>",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Note: 'No Deficiency Pattern' is just one of the 5 categories the model "
        "chooses from - it is NOT an overall health score. A low percentage here "
        "simply means your entered symptoms/values matched a deficiency pattern "
        "more closely than a healthy pattern; it doesn't mean you are '3% healthy'."
    )
    st.caption(
        "This means your symptoms and values also show some resemblance to the "
        "other conditions listed above, but the top one is the model's best guess."
    )
    st.caption(
        "⚠️ This is a machine learning module trained on a sample dataset. "
        "It is not a real medical diagnosis - please consult a doctor for any health concerns."
    )