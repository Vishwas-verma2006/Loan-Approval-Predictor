# Loan Approval Predictor

A Streamlit machine learning application that predicts whether a loan application is likely to be approved based on applicant, financial, credit, and property information.

## Run Locally

```bash
pip install streamlit pandas scikit-learn joblib xgboost
streamlit run loan_approval_pro.py
```

The application loads the trained model from `loan_approval_model.joblib`.

## Project Files

- `loan_approval_pro.py` — Streamlit web application for loan predictions
- `loan_approval_model.joblib` — Saved trained machine learning model
- `practise_task1.ipynb` — Data preprocessing, model training, evaluation, and experimentation

## Model Information

- **Dataset:** Bank loan approval dataset
- **Algorithm:** XGBoost Classifier
- **Estimators:** 70
- **Learning rate:** 0.1
- **Maximum depth:** 3
- **Random state:** 42
- **Problem type:** Binary classification
- **Target:** Loan approved or not approved

The data preparation process fills missing numerical values with the mean and missing categorical values with the most frequent value. Categorical columns are converted into numerical features using one-hot encoding.

## Input Features

The application uses the following information:

- Gender
- Marital status
- Number of dependents
- Education
- Self-employment status
- Applicant income
- Coapplicant income
- Loan amount
- Loan repayment term
- Credit history
- Property area

## Prediction

The application provides:

- Loan approval or rejection
- Estimated probability of loan approval

A loan is predicted as approved when its approval probability is at least **60%**. Otherwise, it is predicted as not approved.

The notebook also includes model comparisons, evaluation metrics, hyperparameter experiments, cross-validation, confusion matrices, and probability-threshold testing.

> This project is intended for educational purposes. Its predictions should not be used as the sole basis for real financial or lending decisions.
