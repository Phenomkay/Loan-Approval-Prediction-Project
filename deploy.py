import streamlit as st
import pickle
import numpy as np
import base64
from PIL import Image

# Load the saved model, encoder, and scaler
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Set the title and background image
def set_background(png_file):
    with open(png_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string.decode()}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function and pass the path of your background image
set_background("loan photo.png")  # Replace with the path to your background image

# Title and introduction
st.title('Loan Approval Prediction App')
st.subheader('Determine if a loan will be approved or rejected')

# Function to predict loan approval
def predict_loan_approval(input_data):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction

# Input fields for user to enter loan details
no_of_dependents = st.number_input('Number of Dependents', min_value=0, max_value=10, step=1)
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
income_annum = st.number_input('Applicant\'s Annual Income', min_value=1000)
loan_amount = st.number_input('Loan Amount', min_value=1)
loan_term = st.number_input('Loan Term (in months)', min_value=1, step=1)
cibil_score = st.number_input('CIBIL Score', min_value=300, max_value=900)
residential_assets_value = st.number_input('Residential Assets Value')
commercial_assets_value = st.number_input('Commercial Assets Value')
luxury_assets_value = st.number_input('Luxury Assets Value')
bank_asset_value = st.number_input('Bank Asset Value')

# Create a dictionary to map user input for encoding
edu_map = {'Graduate': 1, 'Not Graduate': 0}
self_emp_map = {'Yes': 1, 'No': 0}

# Button to predict
if st.button('Predict Loan Status'):
    # Preprocess inputs
    input_data = np.array([[no_of_dependents, edu_map[education], self_emp_map[self_employed], 
                            income_annum, loan_amount, loan_term, cibil_score, 
                            residential_assets_value, commercial_assets_value, 
                            luxury_assets_value, bank_asset_value]])

    # Call the prediction function
    prediction = predict_loan_approval(input_data)

    # Display result
    if prediction == 1:
        st.success('Loan Approved!')
    else:
        st.error('Loan Rejected!')

# Footer
st.markdown('---')
st.markdown('Developed by [Caleb Osagie](https://github.com/Phenomkay)')