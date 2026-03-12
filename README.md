<div align="center">

# LoanSense AI

### Intelligent Loan Risk Assessment Platform

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)](https://getbootstrap.com/)

**Predict loan default risk with 88.6% accuracy using advanced machine learning algorithms**

[Live Demo](#) · [Features](#-features) · [Installation](#-installation) · [Usage](#-usage) · [API](#-api-reference)

</div>

---

## Overview

LoanSense AI is an intelligent loan risk assessment platform that leverages machine learning to predict the likelihood of loan defaults. Built with a Random Forest classifier trained on 255,000+ real loan records, this system provides financial institutions with data-driven insights for lending decisions.

### Why LoanSense AI?

- **High Accuracy**: 88.6% prediction accuracy with optimized hyperparameters
- **Real-time Analysis**: Instant risk assessment in under 1 second
- **16 Features**: Comprehensive analysis using credit, income, and loan parameters
- **Modern Interface**: Clean, professional UI with responsive design
- **API Ready**: RESTful JSON API for seamless integration

---

## Features

| Feature                    | Description                                     |
| -------------------------- | ----------------------------------------------- |
| **Risk Prediction**        | Instant loan default probability assessment     |
| **Confidence Scoring**     | Model confidence percentage for each prediction |
| **Multi-Feature Analysis** | Evaluates 16 financial and personal indicators  |
| **Interactive Dashboard**  | Modern web interface for easy data entry        |
| **Statistics View**        | Detailed model performance metrics and insights |
| **REST API**               | JSON endpoint for programmatic access           |

---

## Tech Stack

| Component        | Technology                         |
| ---------------- | ---------------------------------- |
| **Backend**      | Python 3.11, Flask                 |
| **ML Framework** | Scikit-learn, Pandas, NumPy        |
| **Frontend**     | HTML5, CSS3, Bootstrap 5           |
| **Model**        | Random Forest Classifier           |
| **Deployment**   | Gunicorn, Render/Heroku compatible |

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/HarshilHada91/LoanSense_AI-.git
cd LoanSense_AI-

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`

---

## Usage

### Web Interface

1. Navigate to the home page
2. Fill in the loan application details:
   - Personal information (age, education, marital status)
   - Employment details (type, income, duration)
   - Loan specifics (amount, purpose, term, interest rate)
   - Credit information (score, credit lines, DTI ratio)
3. Click **"Analyze Risk"** to get the prediction
4. View the risk assessment with confidence score

### Input Parameters

| Parameter       | Type    | Range/Options                                   |
| --------------- | ------- | ----------------------------------------------- |
| Age             | Integer | 18 - 100                                        |
| Income          | Float   | Annual income in USD                            |
| Loan Amount     | Float   | Requested amount in USD                         |
| Credit Score    | Integer | 300 - 850                                       |
| Months Employed | Integer | 0+                                              |
| Interest Rate   | Float   | 0% - 30%                                        |
| Loan Term       | Integer | 6 - 360 months                                  |
| DTI Ratio       | Float   | 0 - 1                                           |
| Education       | Select  | High School, Bachelor's, Master's, PhD          |
| Employment Type | Select  | Full-time, Part-time, Self-employed, Unemployed |
| Loan Purpose    | Select  | Home, Auto, Education, Business, Other          |

---

## API Reference

### Predict Endpoint

```http
POST /api/predict
Content-Type: application/json
```

**Request Body:**

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

**Response:**

```json
{
  "prediction": 0,
  "probability": 0.87,
  "risk_level": "Low",
  "message": "Low default risk - Consider approval"
}
```

---

## Model Performance

| Metric        | Score |
| ------------- | ----- |
| **Accuracy**  | 88.6% |
| **Precision** | 65.2% |
| **Recall**    | 54.8% |
| **F1-Score**  | 59.5% |
| **ROC-AUC**   | 84.3% |

### Models Evaluated

| Model               | Accuracy    |
| ------------------- | ----------- |
| Random Forest       | **88.6%** ✓ |
| XGBoost             | 86.2%       |
| Gradient Boosting   | 85.1%       |
| Logistic Regression | 81.4%       |
| Decision Tree       | 78.9%       |

---

## Project Structure

```
LoanSense_AI/
├── app.py                    # Flask application
├── requirements.txt          # Python dependencies
├── Procfile                  # Deployment configuration
├── runtime.txt               # Python version
│
├── templates/                # HTML templates
│   ├── base.html            # Base template
│   ├── index.html           # Home page (prediction form)
│   ├── result.html          # Prediction results
│   ├── about.html           # About page
│   ├── stats.html           # Statistics dashboard
│   └── error.html           # Error page
│
├── model.pkl                 # Trained Random Forest model
├── scaler.pkl                # Feature scaler
├── label_encoders.pkl        # Categorical encoders
├── feature_names.pkl         # Feature names
│
├── Loan_default.csv          # Training dataset
└── Loan_Default_ML_Project.ipynb  # Jupyter notebook
```

---

## Deployment

### Render (Recommended)

1. Connect your GitHub repository to Render
2. Configure build settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
3. Deploy

### Heroku

```bash
heroku login
heroku create loansense-ai
git push heroku main
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with by Harshil Hada**

[Report Bug](https://github.com/HarshilHada91/LoanSense_AI-/issues) · [Request Feature](https://github.com/HarshilHada91/LoanSense_AI-/issues)

</div>
#   L o a n S e n s e _ A I -  
 #   L o a n S e n s e _ A I -  
 