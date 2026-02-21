import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ----------------------
# Load Model & Encoders
# ----------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    gender_encoder = pickle.load(f)

with open("onehot1.pkl", "rb") as f:
    employment_encoder = pickle.load(f)

with open("onehot.pkl", "rb") as f:
    education_encoder = pickle.load(f)

with open("target1.pkl", "rb") as f:
    grade_encoder = pickle.load(f)

with open("target2.pkl", "rb") as f:
    purpose_encoder = pickle.load(f)

with open("target3.pkl", "rb") as f:
    marital_encoder = pickle.load(f)

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Loan Prediction", page_icon="💰", layout="wide")
st.title("💰 Loan Paid Back Prediction")

st.markdown("Predict if a loan will be paid back successfully. Fill the details below:")

# ----------------------
# Inputs in two columns
# ----------------------
col1, col2 = st.columns(2)

with col1:
    annual_income = st.number_input("💵 Annual Income")
    debt_ratio = st.number_input("📊 Debt To Income Ratio")
    credit_score = st.number_input("📝 Credit Score")
    loan_amount = st.number_input("💰 Loan Amount")

with col2:
    interest_rate = st.number_input("📈 Interest Rate")
    gender = st.selectbox("🚻 Gender", ["Male", "Female"])
    marital_status = st.selectbox("💍 Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    education_level = st.selectbox("🎓 Education Level", [
        "High School", "Bachelor", "Master", "PhD", "Associate", "Other"
    ])

# ----------------------
# More inputs in expander
# ----------------------
with st.expander("More Loan Details 🔽"):
    employment_status = st.selectbox("💼 Employment Status", [
        "Employed", "Unemployed", "Self-employed", "Retired", "Other"
    ])
    loan_purpose = st.selectbox("🏦 Loan Purpose", [
        "Debt consolidation", "Other", "Car", "Home", "Education",
        "Business", "Medical", "Vacation"
    ])
    grade_subgrade = st.selectbox("🏷 Grade Subgrade", [
        "A1","A2","A3","A4","A5",
        "B1","B2","B3","B4","B5",
        "C1","C2","C3","C4","C5",
        "D1","D2","D3","D4","D5",
        "E1","E2","E3","E4","E5",
        "F1","F2","F3","F4","F5"
    ])

# ----------------------
# Prepare Input DataFrame
# ----------------------
input_df = pd.DataFrame({
    "annual_income": [annual_income],
    "debt_to_income_ratio": [debt_ratio],
    "credit_score": [credit_score],
    "loan_amount": [loan_amount],
    "interest_rate": [interest_rate],
    "gender": [gender],
    "marital_status": [marital_status],
    "education_level": [education_level],
    "employment_status": [employment_status],
    "loan_purpose": [loan_purpose],
    "grade_subgrade": [grade_subgrade]
})

# ----------------------
# Apply Encoders
# ----------------------
# Label Encoding
input_df["gender"] = gender_encoder.transform(input_df["gender"])

# Target Encoding
input_df["grade_subgrade"] = grade_encoder.transform(input_df["grade_subgrade"])
input_df["loan_purpose"] = purpose_encoder.transform(input_df["loan_purpose"])
input_df["marital_status"] = marital_encoder.transform(input_df["marital_status"])

# OneHot Encoding
emp_encoded = employment_encoder.transform(input_df[["employment_status"]])
emp_encoded_df = pd.DataFrame(emp_encoded, columns=employment_encoder.get_feature_names_out(['employment_status']))

edu_encoded = education_encoder.transform(input_df[["education_level"]])
edu_encoded_df = pd.DataFrame(edu_encoded, columns=education_encoder.get_feature_names_out(['education_level']))

# Drop original categorical columns and concat encoded
input_df = input_df.drop(["employment_status", "education_level"], axis=1)
input_df = pd.concat([input_df, emp_encoded_df, edu_encoded_df], axis=1)

# ----------------------
# Prediction Button
# ----------------------
if st.button("Predict 💡"):
    prediction_proba = model.predict_proba(input_df)[0][1] * 100  # probability in %
    
    st.markdown(f"### Prediction Probability: {prediction_proba:.2f}%")
    
    # Progress bar
    st.progress(int(prediction_proba))
    
    # Risk Level
    if prediction_proba > 75:
        st.success("✅ Loan Very Likely to be Paid Back")
    elif prediction_proba > 40:
        st.info("⚠️ Loan Might Be Paid Back")
    else:
        st.error("❌ Loan Unlikely to be Paid Back")
