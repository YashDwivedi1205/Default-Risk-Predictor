import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved object
model = pickle.load(open("default_risk_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
model_columns = pickle.load(open("model_columns.pkl","rb"))


st.title("🏦Bank Loan Default Risk Predictor")
st.write("Enter applicant details to predict loan approval probability.")

# Input Form
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents",["0","1","2","3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value= 0)
coapplicant_income = st.number_input("Coapplicant Income", min_value= 0)
loan_amount = st.number_input("Loan Amount", min_value = 0)
loan_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# # Prediction

    
if st.button("Predict Risk"):
    # 1. Create DataFrame
    input_data = pd.DataFrame([[gender, married, dependents, education, self_employed, 
                                 applicant_income, coapplicant_income, loan_amount, 
                                 loan_term, credit_history, property_area]],
                               columns=['Gender', 'Married', 'Dependents', 'Education', 
                                        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 
                                        'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'])
        
    # 2. Encoding
    input_encoded = pd.get_dummies(input_data)
    
    # 3. Align features with the 20 columns the scaler expects
    # This will drop any extra columns and add missing ones as 0
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # 4. Final Verification check
    if input_final.shape[1] == 20:
        input_scaled = scaler.transform(input_final)
        prediction = model.predict(input_scaled)
        
        if prediction[0] == 1:
            st.success("✅ The loan is likely to be APPROVED")
        else:
            st.error("❌ The loan is likely to be REJECTED")
    else:
        st.error(f"Mismatch: Your app produced {input_final.shape[1]} columns, but we need 20.")