"""
LoanSense AI - Intelligent Loan Risk Assessment Application
============================================================
This is a Flask web application for predicting loan default risk
using machine learning. The application provides a professional
interface for loan risk assessment.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR.parent / 'frontend'

# Initialize Flask app
app = Flask(__name__, template_folder=str(TEMPLATES_DIR))

# Load the trained model and preprocessors

try:
    # Prefer the smaller model for serverless deployments; fallback keeps local compatibility.
    model_path = BASE_DIR / 'loan_model.pkl'
    if not model_path.exists():
        model_path = BASE_DIR / 'model.pkl'

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(BASE_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load the post-encoding feature names (24 features) used during training
    features_path = BASE_DIR / 'features.pkl'
    with open(features_path, 'rb') as f:
        model_features = pickle.load(f)

    print(f"✅ Model loaded from {model_path.name} expecting {len(model_features)} features!")
except FileNotFoundError as e:
    print(f"⚠️ Model files not found: {e}. Please run the notebook first.")
    model = None
    scaler = None
    model_features = None

# Define categorical options for the form
EDUCATION_OPTIONS = ["High School", "Bachelor's", "Master's", "PhD"]
EMPLOYMENT_OPTIONS = ["Full-time", "Part-time", "Self-employed", "Unemployed"]
MARITAL_OPTIONS = ["Single", "Married", "Divorced"]
LOAN_PURPOSE_OPTIONS = ["Home", "Auto", "Education", "Business", "Other"]
YES_NO_OPTIONS = ["Yes", "No"]


# =============================================================================
# TASK 9: Backend Routes
# =============================================================================

@app.route('/')
def home():
    """Home page with the prediction form"""
    return render_template('index.html',
                         education_options=EDUCATION_OPTIONS,
                         employment_options=EMPLOYMENT_OPTIONS,
                         marital_options=MARITAL_OPTIONS,
                         loan_purpose_options=LOAN_PURPOSE_OPTIONS,
                         yes_no_options=YES_NO_OPTIONS)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        age = int(request.form['age'])
        income = float(request.form['income'])
        loan_amount = float(request.form['loan_amount'])
        credit_score = int(request.form['credit_score'])
        months_employed = int(request.form['months_employed'])
        num_credit_lines = int(request.form['num_credit_lines'])
        interest_rate = float(request.form['interest_rate'])
        loan_term = int(request.form['loan_term'])
        dti_ratio = float(request.form['dti_ratio'])
        education = request.form['education']
        employment_type = request.form['employment_type']
        marital_status = request.form['marital_status']
        has_mortgage = request.form['has_mortgage']
        has_dependents = request.form['has_dependents']
        loan_purpose = request.form['loan_purpose']
        has_cosigner = request.form['has_cosigner']
        
        # Build raw input DataFrame
        input_data = pd.DataFrame({
            'Age': [age],
            'Income': [income],
            'LoanAmount': [loan_amount],
            'CreditScore': [credit_score],
            'MonthsEmployed': [months_employed],
            'NumCreditLines': [num_credit_lines],
            'InterestRate': [interest_rate],
            'LoanTerm': [loan_term],
            'DTIRatio': [dti_ratio],
            'HasMortgage': [has_mortgage],
            'HasDependents': [has_dependents],
            'HasCoSigner': [has_cosigner],
            'Education': [education],
            'EmploymentType': [employment_type],
            'MaritalStatus': [marital_status],
            'LoanPurpose': [loan_purpose]
        })

        # Step 1: Binary label encode Yes/No columns (matches training)
        binary_cols = ['HasMortgage', 'HasDependents', 'HasCoSigner']
        for col in binary_cols:
            input_data[col] = (input_data[col] == 'Yes').astype(int)

        # Step 2: One-hot encode multi-class columns with drop_first=True (matches training)
        multi_cat_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']
        input_encoded = pd.get_dummies(input_data, columns=multi_cat_cols, drop_first=True)

        # Step 3: Scale numerical features
        numerical_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                          'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
        input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])

        # Step 4: Align columns to exactly match training feature set (fills missing dummies with 0)
        input_final = input_encoded.reindex(columns=model_features, fill_value=0)

        # Make prediction
        prediction = model.predict(input_final)[0]
        probability_array = model.predict_proba(input_final)[0]

        probability = probability_array[0] if prediction == 0 else probability_array[1]

        return render_template('result.html', prediction=int(prediction), probability=float(probability))
        
    except Exception as e:
        return render_template('error.html', error_message=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (returns JSON)"""
    try:
        data = request.get_json()
        
        # Build raw input DataFrame
        input_data = pd.DataFrame([data])

        # Binary label encode Yes/No columns
        binary_cols = ['HasMortgage', 'HasDependents', 'HasCoSigner']
        for col in binary_cols:
            if col in input_data.columns:
                input_data[col] = (input_data[col] == 'Yes').astype(int)

        # One-hot encode multi-class columns with drop_first=True
        multi_cat_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']
        input_encoded = pd.get_dummies(input_data, columns=multi_cat_cols, drop_first=True)

        # Scale numerical features
        numerical_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                          'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
        input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])

        # Align to training feature set
        input_final = input_encoded.reindex(columns=model_features, fill_value=0)

        # Make prediction
        prediction = model.predict(input_final)[0]
        probability = model.predict_proba(input_final)[0]

        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'prediction_label': 'Default' if prediction == 1 else 'No Default',
            'probability_default': float(probability[1]),
            'probability_no_default': float(probability[0])
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/about')
def about():
    """About page with model information"""
    return render_template('about.html')


@app.route('/stats')
def stats():
    """Statistics page with model metrics"""
    return render_template('stats.html')


# =============================================================================
# Run the Flask Application
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏦 LOAN DEFAULT PREDICTION WEB APPLICATION")
    print("="*60)
    print("Starting Flask server...")
    print("Open http://127.0.0.1:5000 in your browser")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
