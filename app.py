import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# --- 1. Load the trained model and encoders ---
@st.cache_resource
def load_models():
    """
    Load the Random Forest Classifier and Label Encoders from pickle files.
    """
    model, encoders = None, None
    try:
        with open('customer_churn_model.pkl', 'rb') as file:
            model_dict = pickle.load(file)
            model = model_dict['model']  # Extract the actual model
    except FileNotFoundError:
        st.error("Model file 'customer_churn_model.pkl' not found. Ensure it is in the same directory.")
    
    try:
        with open('encoders.pkl', 'rb') as file:
            encoders = pickle.load(file)
    except FileNotFoundError:
        st.error("Encoders file 'encoders.pkl' not found. Ensure it is in the same directory.")
        
    return model, encoders

rfc_model, label_encoders = load_models()

# Define the expected features based on standard Telco churn dataset
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# --- 2. Set up the Streamlit page layout ---
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("Customer Churn Prediction App")
st.write("""
This application predicts whether a customer is likely to churn. 
Fill out the customer information in the sidebar and click **Predict Customer Churn** to evaluate their churn probability!
""")

# --- 3. Create an input form ---
st.sidebar.header("Customer Information Input")

def get_user_input():
    """
    Collect user inputs from Streamlit sidebar widgets to match the 19 features required.
    """
    data = {}
    
    # Collect numerical inputs using number_input
    st.sidebar.subheader("Numerical Features")
    data['tenure'] = st.sidebar.number_input('Tenure (months)', min_value=0, max_value=100, value=12)
    data['MonthlyCharges'] = st.sidebar.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=50.0)
    data['TotalCharges'] = st.sidebar.number_input('Total Charges ($)', min_value=0.0, max_value=10000.0, value=600.0)
    
    # Collect categorical inputs using selectbox
    st.sidebar.subheader("Categorical Features")
    for feature in categorical_features:
        # Dynamically retrieve options from loaded encoders if available
        if label_encoders and feature in label_encoders:
            options = label_encoders[feature].classes_
        else:
            # Fallback values if the encoders haven't been loaded correctly
            options = [0, 1] if feature == 'SeniorCitizen' else ['Yes', 'No']
            
        data[feature] = st.sidebar.selectbox(f"{feature}", options)

    # Return as DataFrame with correct column order
    correct_order = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
                     'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                     'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
    return pd.DataFrame(data, index=[0])[correct_order]

input_df = get_user_input()

st.subheader("Selected Customer Profile")
st.dataframe(input_df)

# --- 4. Preprocess user input & 5. Make a prediction ---
if st.button("Predict Customer Churn"):
    if rfc_model and label_encoders:
        # Preprocess input data
        processed_df = input_df.copy()
        
        # Apply the exact LabelEncoder transformations from the training phase
        for feature in categorical_features:
            if feature == 'SeniorCitizen':
                # SeniorCitizen is already numeric (0 or 1), no encoding needed
                continue
            if feature in label_encoders:
                # Transform single value using the pre-fit label encoder
                processed_df[feature] = label_encoders[feature].transform(processed_df[feature])
            else:
                st.warning(f"No encoder found for {feature}, skipping text encoding.")
        
        # Determine Prediction
        try:
            prediction = rfc_model.predict(processed_df)
            prediction_proba = rfc_model.predict_proba(processed_df)
            
            # Predict returns an array, mapping [1] / [0] conventionally mapped to Churn presence
            churn_status = "Churn" if prediction[0] == 1 else "No Churn"
            probability_of_churn = prediction_proba[0][1]
            
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error(f"**Prediction: {churn_status}**")
            else:
                st.success(f"**Prediction: {churn_status}**")
                
            st.write(f"Customer's Probability of Churning: **{probability_of_churn * 100:.2f}%**")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("Cannot perform prediction because the model or encoders failed to load.")


