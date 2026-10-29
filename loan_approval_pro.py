import streamlit as st 
import pandas as pd 
import sklearn 
import joblib 

# loading the model 
with open('D:/CODING/ML using Python/Venv/Code/projects/loan_approval_model.joblib', 'rb') as f:
    model = joblib.load(f)

st.title("Loan Approval Prediction")
st.subheader('''_This is a simple web application for Machine learning model. which can predict the loan approval based on the input features._''')
st.write('''_This project will help predict wheather the loan will be approved or not based on the user data provided._''')

def user_input():
    st.write("**Please enter the following details:-**")
    with st.container(border=True):
        st.write("**Personal Details:-**")
        st.session_state.gender = st.selectbox('Gender', options=['Male', 'Female']) 
        st.session_state.married = st.selectbox("Married", options=['Yes', 'No'])
        st.session_state.dependents = st.selectbox("How many dependents members do you have?", options={'0': 0, '1': 0, '2': 0, '3+': 0})
        st.session_state.education = st.selectbox("Education", options=['Graduate', 'Not Graduate'])
        st.session_state.self = st.selectbox("Are you a self employeed ?", options=['Yes', 'No'])

    st.session_state.gender = 1 if st.session_state.gender == 'Male' else 0
    st.session_state.married = 1 if st.session_state.married == 'Yes' else 0
    st.session_state.education = 1 if st.session_state.education == 'Graduate' else 0
    st.session_state.self = 1 if st.session_state.self == 'Yes' else 0

    with st.container(border=True):
        st.write("**Finacial Details:-**")
        app_income = st.number_input("Applicant Income", min_value=0, step=100, key='applicant_income')
        coapp_income = st.number_input("Family members Income", min_value=0, step=100, key='family_income')
        loan_amt = st.number_input('Loan amount', min_value=0, step=100, key='laon_amount')
        loan_term =st.number_input('How many months do you need to pay the loan back?', min_value=0, step=1, key='laon_months')
        st.session_state.credit = st.selectbox("Do you have a credit history?", options=['Yes', 'No'], key='credit_history')
        st.session_state.area = st.selectbox("Your property area", options=['Urban', 'Semiurban', 'Rural'])

    st.session_state.credit = 1 if st.session_state.credit == 'Yes' else 0
    # for dependents 
    dep_dic = {'0': 0, '1': 0, '2': 0, '3+': 0}
    if st.session_state.dependents is not None:
        if st.session_state.dependents == '1':
            dep_dic['1'] = 1
        elif st.session_state.dependents == '2':
            dep_dic['2'] = 1
        elif st.session_state.dependents == '3+':
            dep_dic['3+'] = 1  
        st.session_state.dependents = dep_dic

    # for area
    dic = {'Urban': 0, 'Semiurban': 0, 'Rural': 0} 
    if st.session_state.area is not None:
        if st.session_state.area == 'Urban':
            dic['Urban'] = 1
        elif st.session_state.area == 'Semiurban':
            dic['Semiurban'] = 1 
        else:
            dic['Urban'] = 0
            dic['Semiurban'] = 0
        st.session_state.area = dic

    features = {
        'ApplicantIncome':[app_income],
        'CoapplicantIncome': [coapp_income],
        'LoanAmount': [loan_amt],
        'Loan_Amount_Term':[loan_term],
        'Credit_History':[st.session_state.credit],
        'Gender_Male' : [st.session_state.gender],
        'Married_Yes': [st.session_state.married],
        'Dependents_1': [st.session_state.dependents['1']],
        'Dependents_2': [st.session_state.dependents['2']],
        'Dependents_3+': [st.session_state.dependents['3+']],
        'Education_Not Graduate': [st.session_state.education],
        'Self_Employed_Yes':[st.session_state.self],
        'Property_Area_Semiurban':[st.session_state.area['Semiurban']],
        'Property_Area_Urban':[st.session_state.area['Urban']]
    }

    for i in features:
        if features[i] is None:
            st.warning(f"Please fill the {i} field.")

    return pd.DataFrame(features, index=[0]) # this will return a dataframe with the features and their values in a single row

input_df = user_input()

if 'prediction' not in st.session_state:
    st.session_state.prediction = None

if 'probablity' not in st.session_state:
    st.session_state.probablity = None

# button for prediction 
if st.button('**Predict**', help='Click to predict the loan approval'):
    if input_df['LoanAmount'].values[0] == 0 or input_df['Loan_Amount_Term'].values[0] == 0:
        st.warning("Please fill all the values properly before predicting the loan approval.")
    else:
        #st.session_state.prediction = model.predict(input_df)
        threshold = 0.6 # we used threshold value from default 50% to 60% to reduce the false positive rate and increase the true positive rate
        pro = model.predict_proba(input_df)[0][1] 
        st.session_state.prediction = 1 if pro >= threshold else 0

if st.session_state.prediction is not None:
    st.write("**Prediction:-**")
    st.write(f"**❌ Loan not approved**" if st.session_state.prediction == 0 else "**✅ Loan approved**")

# button for probability 
if st.button("**Probablity**", help='Click to see the probablity of loan approval'):
    st.session_state.probablity = model.predict_proba(input_df)


if st.session_state.probablity is not None:
    st.write(f"**Probablity of loan approved:- {st.session_state.probablity[0][1]*100:.2f}%**")
    st.write("The probablity must be greater than 60% to approve the loan.")
    #st.write(f"**❌ Loan not approved:- {st.session_state.probablity[0][0]*100:.2f}%**")
    #st.write(f"**✅ Loan approved:- {st.session_state.probablity[0][1]*100:.2f}%**")

if st.button("**Clear**", help='Click to clear the previous prediction and prbablity'):
    st.session_state.prediction = None
    st.session_state.probablity = None
    st.rerun()
