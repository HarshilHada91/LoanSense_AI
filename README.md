# LoanSense AI

An end-to-end machine learning web application that predicts loan default risk from applicant and loan attributes.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3-000000?style=flat-square&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Model Inputs](#model-inputs)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Overview

LoanSense AI helps evaluate whether a loan application is likely to default. It combines a trained classification model with a Flask-based UI and a JSON API.

The project includes:

- A browser interface for manual prediction.
- A programmatic API endpoint for integration.
- A notebook used for model development.

## Key Features

- Predict loan default risk from 16 input features.
- Return class prediction plus probability scores.
- Offer both web-form and API workflows.
- Include deployment-ready configuration for Gunicorn.
- Handle invalid requests with structured error responses.

## Tech Stack

- Backend: Flask
- ML/Data: scikit-learn, pandas, numpy
- Serving: Gunicorn
- Frontend: HTML templates (Jinja2)

## Project Structure

```text
loan_ml_project/
|-- backend/
|   |-- app.py
|   |-- requirements.txt
|   |-- Procfile
|   |-- runtime.txt
|   |-- api/
|   |   `-- index.py
|   |-- Loan_Default_ML_Project.ipynb
|   |-- Loan_default.csv
|   |-- loan_model.pkl
|   |-- model.pkl
|   |-- scaler.pkl
|   |-- label_encoders.pkl
|   |-- feature_names.pkl
|   `-- features.pkl
|-- frontend/
|   `-- templates/
|       |-- base.html
|       |-- index.html
|       |-- result.html
|       |-- stats.html
|       |-- about.html
|       `-- error.html
|-- render.yaml
`-- README.md
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/HarshilHada91/LoanSense_AI.git
cd LoanSense_AI
```

### 2. Create and Activate a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Run the Application

Development mode:

```bash
python backend/app.py
```

Production-like local run:

```bash
cd backend && gunicorn app:app
```

Open the app at:

- <http://127.0.0.1:5000>

## Usage

### Web App

1. Open the home page.
2. Fill out applicant and loan details.
3. Submit the form.
4. Review predicted risk and confidence probabilities.

### API

Endpoint:

```http
POST /api/predict
Content-Type: application/json
```

Example request:

```json
{
  "Age": 35,
  "Income": 75000,
  "LoanAmount": 25000,
  "CreditScore": 720,
  "MonthsEmployed": 48,
  "NumCreditLines": 4,
  "InterestRate": 7.5,
  "LoanTerm": 60,
  "DTIRatio": 0.35,
  "Education": "Bachelor's",
  "EmploymentType": "Full-time",
  "MaritalStatus": "Married",
  "HasMortgage": "Yes",
  "HasDependents": "No",
  "LoanPurpose": "Home",
  "HasCoSigner": "No"
}
```

Example response:

```json
{
  "success": true,
  "prediction": 0,
  "prediction_label": "No Default",
  "probability_default": 0.12,
  "probability_no_default": 0.88
}
```

## API Reference

### POST /api/predict

Input:

- JSON object with all required numerical and categorical fields.

Output:

- `success`: boolean
- `prediction`: 0 or 1
- `prediction_label`: `No Default` or `Default`
- `probability_default`: float
- `probability_no_default`: float

Error output (HTTP 400):

- `success`: false
- `error`: error message

## Model Inputs

Numerical features:

- Age
- Income
- LoanAmount
- CreditScore
- MonthsEmployed
- NumCreditLines
- InterestRate
- LoanTerm
- DTIRatio

Categorical features:

- Education
- EmploymentType
- MaritalStatus
- HasMortgage
- HasDependents
- LoanPurpose
- HasCoSigner

## Deployment

This repository already includes Render-ready configuration:

- `render.yaml` in project root
- `backend/Procfile`: `web: gunicorn app:app`
- `backend/runtime.txt`: Python runtime version

Suitable platforms:

- Render
- Railway
- Heroku-compatible runtimes

### Render (Recommended)

Option 1: Blueprint deploy (recommended)

1. Push this repository to GitHub.
2. In Render, select New + and choose Blueprint.
3. Select your repository and deploy.
4. Render reads `render.yaml` and configures the service automatically.

Option 2: Manual Web Service setup

1. Create a new Web Service from your repository.
2. Set Root Directory to `backend`.
3. Set Build Command to `pip install -r requirements.txt`.
4. Set Start Command to `gunicorn app:app`.
5. Deploy.

## Troubleshooting

- Model files missing at startup:
  Ensure `loan_model.pkl` (or `model.pkl`), `scaler.pkl`, `label_encoders.pkl`, and `feature_names.pkl` are present in `backend/`.
- Package installation issues:
  Upgrade pip using `python -m pip install --upgrade pip`.
- Render deployment fails to start:
  Confirm Root Directory is `backend` and Start Command is `gunicorn app:app`.
- Large model files on GitHub:
  This project uses Git LFS for large artifacts.

## Roadmap

- Add model versioning and metadata endpoint.
- Add input schema validation for API requests.
- Add unit tests for prediction pipeline and routes.
- Add CI workflow for linting and tests.

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request.

## License

This project is licensed under the MIT License.

If you use this work in academic or portfolio settings, please provide attribution.
