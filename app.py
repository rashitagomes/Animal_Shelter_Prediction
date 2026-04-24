import pickle
import streamlit as st
import os

# ✅ ALWAYS define first
model = None

model_path = "final_model.pkl"

if not os.path.exists(model_path):
    st.warning("⚠ Model file not found")
else:
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")

    
st.set_page_config(page_title="Outcome Prediction", layout="wide")

# --------------------------
# Custom CSS (FIT-TO-SCREEN & NO RED)
# --------------------------
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #020617);
        color: #e2e8f0;
    }
    
    /* Force content to stay compact */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
        max-width: 95% !important;
    }

    /* Titles */
    .title-text {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 5px;
        background: linear-gradient(90deg, #38bdf8, #5eead4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Side-by-side Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid #334155;
        border-radius: 15px;
        padding: 25px;
        height: 100%;
    }

    /* Inputs */
    .stSelectbox, .stNumberInput {
        margin-bottom: 10px;
    }

    /* Vibrant Result Box */
    .result-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 300px;
        border: 2px solid #38bdf8;
        border-radius: 20px;
        background: rgba(56, 189, 248, 0.05);
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.2);
        text-align: center;
    }

    .prediction-val {
        font-size: 48px;
        font-weight: 900;
        color: #5eead4;
        text-shadow: 0 0 15px rgba(94, 234, 212, 0.5);
    }

    /* Button - Blue/Cyan only */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #0ea5e9);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 15px;
        font-weight: bold;
        width: 100%;
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 15px rgba(14, 165, 233, 0.6);
        border: none;
        color: white;
    }

    /* Hide redundant Streamlit UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --------------------------
# HEADER
# --------------------------
st.markdown('<div class="title-text">🐾 Animal Outcome Dashboard</div>', unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; margin-top: -10px;'>Predicting future transitions for shelter animals.</p>", unsafe_allow_html=True)

# --------------------------
# MAIN LAYOUT (Single Screen)
# --------------------------
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Animal Profile")
    
    # Nested columns for input compactness
    c1, c2 = st.columns(2)
    with c1:
        animal_type = st.selectbox("Animal Type", ["Dog", "Cat", "Other", "Bird"])
        sex = st.selectbox("Sex upon Intake", ["Neutered Male", "Spayed Female", "Intact Male", "Intact Female", "Unknown"])
        age = st.number_input("Age in Days", min_value=0, value=365)
    
    with c2:
        intake_type = st.selectbox("Intake Type", ["Stray", "Owner Surrender", "Public Assist", "Wildlife", "Euthanasia Request"])
        intake_condition = st.selectbox("Intake Condition", ["Normal", "Injured", "Sick", "Aged", "Feral", "Other"])
    
    predict_btn = st.button("⚡ ANALYZE DATA")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.subheader("Prediction Result")
    if predict_btn:
        if model is not None:
            sample = pd.DataFrame([{
                'animal_type': animal_type,
                'sex_upon_intake': sex,
                'age_in_days': age,
                'intake_type': intake_type,
                'intake_condition': intake_condition
            }])

            prediction = model.predict(sample)[0]
            
            try:
                prob = model.predict_proba(sample).max() * 100
            except:
                prob = 0

            st.markdown(f"""
                <div class="result-container">
                    <p style="color: #38bdf8; font-weight: 600; letter-spacing: 1px; margin-bottom: 0;">ESTIMATED OUTCOME</p>
                    <div class="prediction-val">{prediction}</div>
                    <p style="color: #94a3b8; font-size: 14px; margin-top: 10px;">Confidence Score: {prob:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Subtle notice instead of red error
            st.info("Please ensure 'final_model.pkl' is uploaded to the root directory.")
    else:
        st.markdown("""
            <div class="result-container" style="border-style: dashed; opacity: 0.5;">
                <p style="color: #94a3b8;">Click 'Analyze Data' to view prediction</p>
            </div>
        """, unsafe_allow_html=True)

# Footer/Padding cleanup
st.markdown("<br>", unsafe_allow_html=True)
