# Loan-Approval-Predictor
The application uses the trained model stored in loan_approval_model.joblib.

## Project Files
loan_approval_pro.py — Streamlit prediction application
loan_approval_model.joblib — Saved trained machine learning model
practise_task1.ipynb — Data preprocessing, model training, evaluation, and experimentation

## Model Information
Algorithm: XGBoost Classifier
Estimators: 70
Learning rate: 0.1
Maximum depth: 3
Random state: 42
Problem type: Binary classification
Target: Loan approved or not approved

The model was trained using a bank loan approval dataset. Missing numerical values were filled using the mean, while missing categorical values were filled using the most frequent value. Categorical features were converted into numerical values using one-hot encoding.

## Input Features
The application accepts the following information:

Gender
Marital status
Number of dependents
Education
Self-employment status
Applicant income
Coapplicant income
Loan amount
Loan repayment term
Credit history
Property area
The input data is converted into the same encoded feature format used during model training before making a prediction.

## Prediction
The application provides:

Loan approval or rejection
Estimated probability of loan approval
The Streamlit application uses a 60% probability threshold. If the approval probability is 60% or higher, the loan is predicted as approved; otherwise, it is predicted as not approved.

The notebook also includes model evaluation, hyperparameter experiments, cross-validation, confusion matrices, and probability threshold testing.

This project is intended for educational purposes. Its predictions should not be used as the sole basis for real financial or lending decisions.
