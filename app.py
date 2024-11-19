import pickle
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import base64
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the FastAPI app
app = FastAPI()

# Load the model, encoder, and scaler
try:
    model = pickle.load(open('model.pkl', 'rb'))
    encoder = pickle.load(open('encoder.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except Exception as e:
    logging.error("Error loading model, encoder or scaler: %s", e)
    raise

# Input data model for FastAPI
class LoanRequest(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float

# Load background image for HTML
def set_background_image():
    image_path = "loan_photo.png"  # Change to the path of your image
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
        return f"background-image: url('data:image/png;base64,{encoded_image}'); background-size: cover;"
    return ""

# Prediction logic
def predict_loan_approval(input_data):
    try:
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        return prediction
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        raise

@app.post("/predict/")
async def predict_loan(request: Request):
    try:
        body = await request.json()
        logging.debug("Received JSON body: %s", body)

        # Map education and self_employed to numeric values
        edu_map = {'Graduate': 1, 'Not Graduate': 0}
        self_emp_map = {'Yes': 1, 'No': 0}

        # Prepare the input data for prediction
        input_data = np.array([[body['no_of_dependents'], edu_map[body['education']], self_emp_map[body['self_employed']], 
                                body['income_annum'], body['loan_amount'], body['loan_term'], body['cibil_score'], 
                                body['residential_assets_value'], body['commercial_assets_value'], 
                                body['luxury_assets_value'], body['bank_asset_value']]])

        # Get prediction
        prediction = predict_loan_approval(input_data)

        # Return prediction response
        if prediction == 1:
            return {"status": "Loan Approved!"}
        else:
            return {"status": "Loan Rejected!"}
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logging.error("Error loading index.html: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
