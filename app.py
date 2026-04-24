import streamlit as st
import pandas as pd
import pickle

# --------------------------
# 1. Load Model & Initialize
# --------------------------
model = None
try:
    with open('final_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Model load error: {e}")

st.set_page_config(page_title="Outcome Prediction", layout="wide")

# --------------------------
# 2. Mappings (From your Jupyter Notebook)
# --------------------------
# Note: Keeping the trailing spaces exactly as seen in your screenshot
animal_type_map = {'Dog': 1, 'Cat': 2, 'Other': 3, 'Bird': 4, 'Livestock': 5}

sex_map = {
    'Intact Male': 0,
    'Intact Female ': 1,  # Space included per screenshot
    'Neutered Male': 2,
    'Spayed Female': 3,
    'Unknown': 4
}

intake_type_map = {
    'Stray ': 0,           # Space included per screenshot
    'Owner Surrender ': 1, # Space included per screenshot
    'Public Assist ': 2,   # Space included per screenshot
    'Wildlife ': 3,        # Space included per screenshot
    'Euthanasia Request': 4
}

intake_condition_map = {
    'Normal ': 0,          # Space included per screenshot
    'Injured ': 1,         # Space included per screenshot
    'Sick': 2,
    'Aged ': 3,            # Space included per screenshot
    'Feral ': 2,           # Mapped to 2 in your screenshot
    'Pregnant ': 3,        # Mapped to 3 in your screenshot
    'Other ': 4            # Space included per screenshot
}

# --------------------------
# 3. Custom CSS
# --------------------------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f172a, #020617); color: #e2e8f0; }
    .block-container { padding-top: 2rem !important; max-width: 95% !important; }
    .title-text { font-size: 32px; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #5eead4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .glass-card { background: rgba(30, 41, 59, 0.5); border: 1px solid #334155; border-radius: 15px; padding: 25px; }
    .result-container { display: flex; flex-direction: column; justify-content: center; align-items: center; height: 300px; border: 2px solid #38bdf8; border-radius: 20px; background: rgba(56, 189, 248, 0.05); text-align: center; }
    .prediction-val { font-size: 48px; font-weight: 900; color: #5eead4; }
    .stButton>button { background: linear-gradient(90deg, #2563eb, #0ea5e9); color: white; border-radius: 10px; padding: 15px; font-weight: bold; width: 100%; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-text">🐾 Animal Outcome Dashboard</div>', unsafe_allow_html=True)

# --------------------------
# 4. MAIN LAYOUT
# --------------------------
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Animal Profile")
    
    c1, c2 = st.columns(2)
    with c1:
        # These strings MUST match the keys in your maps above exactly
        animal_type = st.selectbox("Animal Type", list(animal_type_map.keys()))
        sex = st.selectbox("Sex upon Intake", list(sex_map.keys()))
        age = st.number_input("Age in Days", min_value=0, value=365)
    
    with c2:
        intake_type = st.selectbox("Intake Type", list(intake_type_map.keys()))
        intake_condition = st.selectbox("Intake Condition", list(intake_condition_map.keys()))
    
    predict_btn = st.button("⚡ ANALYZE DATA")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.subheader("Prediction Result")
    if predict_btn:
        if model is not None:
            # CREATE THE NUMERICAL DATAFRAME
            try:
                sample = pd.DataFrame([{
                    'animal_type': animal_type_map[animal_type],
                    'sex_upon_intake': sex_map[sex],
                    'age_in_days': age,
                    'intake_type': intake_type_map[intake_type],
                    'intake_condition': intake_condition_map[intake_condition]
                }])

                # Reorder columns if your model expects a specific order
                # sample = sample[['animal_type', 'intake_type', ...]] 

                prediction = model.predict(sample)[0]
                
                # If prediction is a number (e.g., 0 or 1), you might need an outcome_map here
                # prediction = outcome_map.get(prediction, prediction)

                try:
                    prob = model.predict_proba(sample).max() * 100
                except:
                    prob = 100.0

                st.markdown(f"""
                    <div class="result-container">
                        <p style="color: #38bdf8; font-weight: 600; margin-bottom: 0;">ESTIMATED OUTCOME</p>
                        <div class="prediction-val">{prediction}</div>
                        <p style="color: #94a3b8; font-size: 14px; margin-top: 10px;">Confidence Score: {prob:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.info("Model not loaded. Please check the 'final_model.pkl' file.")
    else:
        st.markdown('<div class="result-container" style="border-style: dashed; opacity: 0.5;"><p>Click "Analyze Data" to view prediction</p></div>', unsafe_allow_html=True)
