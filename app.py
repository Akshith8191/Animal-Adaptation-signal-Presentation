
import streamlit as st
import pandas as pd
import joblib
import os

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Wildlife Adaptation Strategy Prediction", layout="wide")
st.title("🦝 Wildlife Adaptation Strategy Prediction")
# Sidebar navigation
st.sidebar.markdown("""
<style>
    .big-link {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
        text-decoration: underline;
        margin-bottom: 15px;
        display: block;
        cursor: pointer;
    }
    .big-link:hover {
        color: #d62728;
    }
</style>

<a href="#introduction" class="big-link">Introduction</a>
<a href="#project-overview" class="big-link">Project Overview</a>
<a href="#prediction" class="big-link">Prediction</a>
""", unsafe_allow_html=True)


# ============================================================
# 1️⃣ INTRODUCTION
# ============================================================
st.markdown('<h2 id="introduction">📜 Introduction</h2>', unsafe_allow_html=True)
st.write("""
Urban environments present unique challenges for wildlife, requiring animals to adapt their behaviors 
to survive alongside human activity. Understanding these adaptation strategies is crucial for developing 
measures that promote coexistence and minimize human-wildlife conflict.
This project uses machine learning to predict the adaptation strategy an animal is most likely to adopt 
based on environmental and behavioral factors. By analyzing data on species, time of observation, 
habitat type, noise levels, human density, food and shelter availability, and unusual behaviors, 
the system aims to classify the strategy into one of four categories — Innovation, Exploitation, 
Avoidance, or Habituation.
The insights gained from this analysis can inform better wildlife management, 
urban planning, and conservation efforts.
""")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("https://img.freepik.com/premium-photo/save-earth-animal-day-concept-illustration-generative-ai_658559-3687.jpg?semt=ais_hybrid&w=740", width=400)

# ============================================================
# 2️⃣ PROJECT OVERVIEW (Objective + Dataset + Model Results)
# ============================================================
st.markdown('<h2 id="project-overview">🔍 Project Overview</h2>', unsafe_allow_html=True)

# Objective
st.header("🎯 Objective")
st.write("""
The objective of this project is to build a **machine learning-based prediction system** 
that determines the likely **adaptation strategy** an animal adopts when living in 
human-influenced environments. The system considers various environmental and behavioral 
factors, including species type, time of observation, location type, noise levels, human 
density, food source availability, and shelter quality. By analyzing these parameters, 
the model predicts whether the adaptation signal corresponds to **Innovation**, 
**Exploitation**, **Avoidance**, or **Habituation**. This approach helps wildlife experts 
make better decisions, improves our understanding of how animals live in cities, and 
supports planning that allows people and wildlife to share the environment.
""")

# Dataset Overview
st.header("📊 Dataset Overview")
st.write("""
This dataset contains **1,000 simulated observations** of urban wildlife behavior collected across various 
environments and times of the day.

The purpose is to study how different animal species adapt to human-dominated landscapes.
""")

column_data = {
    "Column Name": [
        "Animal_ID", "Species", "Observation_Time", "Location_Type", "Noise_Level_dB",
        "Human_Density", "Food_Source_Score", "Shelter_Quality_Score",
        "Behavior_Anomaly_Score", "Estimated_Daily_Distance_km", "Adaptation_Signal (Target)"
    ],
    "Description": [
        "Unique identifier for each observed animal.",
        "Species of the animal (Raccoon, Pigeon, Fox, Squirrel).",
        "Time of day when the observation was made (Morning, Afternoon, Evening, Night).",
        "Type of urban area where the animal was observed (Park, Residential, Commercial, Industrial).",
        "Ambient noise level at the observation site, measured in decibels (dB).",
        "Estimated human presence per 100 square meters.",
        "Score (1–10) indicating ease of access to food.",
        "Score (1–10) indicating the quality of nearby shelter.",
        "Score (0–1) showing how unusual the observed behavior was.",
        "Estimated kilometers the animal travels daily.",
        "Observed adaptation strategy (Exploitation, Avoidance, Habituation, Innovation)."
    ]
}
df_columns = pd.DataFrame(column_data)
st.dataframe(df_columns)

# Target variable explanation
st.subheader("🎯 Target Variable – Adaptation_Signal")
st.write("""
**Adaptation_Signal** represents the behavioral strategy an animal uses to cope with urban environments.
It has four possible classes:

- **Exploitation** – The animal actively uses human resources (e.g., food waste, man-made shelters).
- **Avoidance** – The animal deliberately stays away from human activity and infrastructure.
- **Habituation** – The animal becomes accustomed to human presence, showing reduced fear or avoidance.
- **Innovation** – The animal develops new behaviors to solve problems in an urban setting (e.g., opening trash bins, crossing roads strategically).
""")

# Model Evaluation Results
st.header("📈 Model Evaluation Results")
eval_data = {
    "Model": ["KNN", "Naive Bayes", "Decision Tree", "Random Forest", "SVM", "Logistic Regression", "XGBoost"],
    "Train F1": [0.42, 0.34, 1.00, 1.00, 0.54, 0.29, 1.00],
    "Test F1": [0.24, 0.26, 0.24, 0.26, 0.26, 0.22, 0.27],
    "CrossVal F1": [0.25, 0.24, 0.24, 0.25, 0.24, 0.26, 0.27],
    "Fit Status": ["Underfit", "Underfit", "Overfit", "Overfit", "Overfit", "Underfit", "Overfit"]
}
df_eval = pd.DataFrame(eval_data)
st.dataframe(df_eval)

st.subheader("📌 Explanation of Results")
st.write("""
**Overfitting Models (Decision Tree, Random Forest, SVM, XGBoost):**

These models show nearly perfect performance on the training data (Train F1 scores close to 1.00 for most, 0.54 for SVM), which means they have essentially memorized the training examples. However, their performance drops sharply on unseen test data (Test F1 scores around 0.24–0.27), and their cross-validation scores remain low as well. This large gap indicates that the models have learned noise and very specific patterns in the training set that do not generalize well to new data, a phenomenon known as **overfitting**.

---

**Underfitting Models (KNN, Naive Bayes, Logistic Regression):**

These models have low F1 scores on both training and testing datasets (Train F1 scores between 0.29 and 0.42; Test F1 scores between 0.22 and 0.26), suggesting they are too simple or not flexible enough to capture the underlying complexity of the data. Their consistently low cross-validation scores further confirm that these models do not learn sufficient information from the data to make accurate predictions. This problem is referred to as **underfitting**.

---

**Cross-Validation Scores Across Models:**

All models have similarly low cross-validation F1 scores (between 0.24 and 0.27), indicating that none generalize particularly well when evaluated on different subsets of the dataset. This likely points to issues such as limited or imbalanced data, noisy or irrelevant features, or the need for better feature engineering and data preprocessing.
""")

st.subheader("🏆 Best Model Choice")
st.write("""
While **XGBoost** achieved the highest Test F1 (0.27) and CrossVal F1 (0.27),  
the difference compared to other models is small, and it still shows signs of overfitting.

Overall, XGBoost is the best-performing model among those tested.
""")

# ============================================================
# 3️⃣ PREDICTION
# ============================================================
st.markdown('<h2 id="prediction">🐾 Prediction</h2>', unsafe_allow_html=True)
st.write("---")

base_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(base_dir, "input.csv")
model_path = os.path.join(base_dir, "model.pkl")

Data = pd.read_csv(csv_path)
model = joblib.load(model_path)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("https://img.freepik.com/premium-photo/picture-animals-animals-around-planet-with-animals-animals_1276068-24006.jpg?semt=ais_hybrid&w=740")

st.write("Machine Learning Applied to the Selected Dataset:")
st.dataframe(Data.head())
st.write("---")

st.subheader(":orange[🦊 Provide Animal Information to Discover Adaptation Signals:]")

def get_min_max(species, column):
    filtered = Data[Data["Species"] == species][column]
    return float(filtered.min()), float(filtered.max())

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    species = st.selectbox("Select Species:", Data.Species.unique())
with col2:
    observation_time = st.selectbox("Select Observation Time:", Data.Observation_Time.unique())
with col3:
    location_type = st.selectbox("Select Location Type:", Data.Location_Type.unique())

col4, col5, col6 = st.columns([1, 1, 1])
with col4:
    ndb_min, ndb_max = get_min_max(species, "Noise_Level_dB")
    noise_level = st.number_input("Enter Noise Level in DB:", ndb_min, ndb_max)
    st.caption(f"Range: {ndb_min} - {ndb_max}")
with col5:
    hd_min, hd_max = get_min_max(species, "Human_Density")
    human_density = st.number_input("Enter Human Density:", hd_min, hd_max)
    st.caption(f"Range: {hd_min} - {hd_max}")
with col6:
    fss_min, fss_max = get_min_max(species, "Food_Source_Score")
    food_source_score = st.number_input("Enter Food Source Score:", int(fss_min), int(fss_max))
    st.caption(f"Range: {int(fss_min)} - {int(fss_max)}")

col7, col8, col9 = st.columns([1, 1, 1])
with col7:
    sqs_min, sqs_max = get_min_max(species, "Shelter_Quality_Score")
    shelter_quality_score = st.number_input("Enter Shelter Quality Score:", int(sqs_min), int(sqs_max))
    st.caption(f"Range: {int(sqs_min)} - {int(sqs_max)}")
with col8:
    bas_min, bas_max = get_min_max(species, "Behavior_Anomaly_Score")
    behavior_anomaly_score = st.number_input("Enter Behaviour Anomaly Score:", bas_min, bas_max)
    st.caption(f"Range: {bas_min} - {bas_max}")
with col9:
    edd_min, edd_max = get_min_max(species, "Estimated_Daily_Distance_km")
    estimated_daily_distance = st.number_input("Enter Estimated Daily Distance in Km:", edd_min, edd_max)
    st.caption(f"Range: {edd_min} - {edd_max}")

if st.button("Predict"):
    user_data = pd.DataFrame([[species, observation_time, location_type, noise_level, human_density,
                               food_source_score, shelter_quality_score, behavior_anomaly_score, estimated_daily_distance]],
                             columns=Data.columns)

    st.write("Given Data:")
    st.dataframe(user_data)

    # Encode categorical values
    user_data.replace({
        "Fox": 0, "Squirrel": 1, "Raccoon": 2, "Pigeon": 3,
        "Morning": 0, "Night": 1, "Afternoon": 2, "Evening": 3,
        "Residential": 0, "Industrial": 1, "Park": 2, "Commercial": 3
    }, inplace=True)

    probs = model.predict_proba(user_data)[0]
    adaptation_map = {0: "Innovation", 1: "Exploitation", 2: "Avoidance", 3: "Habituation"}

    classprobs_named = {adaptation_map[int(k)]: round(float(probs[i]), 2) for i, k in enumerate(model.classes_)}
    st.write("Predicted Probabilities:")
    st.write(classprobs_named)

    out = model.predict(user_data)[0]
    result = adaptation_map.get(out, str(out))

    st.subheader(f":green[Predicted Adaptation Signal: :blue[{result}]]")
    st.balloons()
