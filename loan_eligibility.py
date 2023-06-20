import streamlit as st
import numpy as np
import pandas as pd

train = pd.read_csv(r"C:\REAL TIME PROJECTS\COLLEGE PROJECT\LOAN ELLIGIBILITY PRIDECTION\loan-train.csv")
test = pd.read_csv(r"C:\REAL TIME PROJECTS\COLLEGE PROJECT\LOAN ELLIGIBILITY PRIDECTION\loan-test.csv")

train = train.drop('Loan_ID',axis=1)

train = train.drop('Gender',axis=1)

train.Loan_Status = train.Loan_Status.replace({"Y": 1, "N" : 0})
# loan_test.Loan_Status = loan_test.Loan_Status.replace({"Y": 1, "N" : 0})

train.Married = train.Married.replace({"Yes": 1, "No" : 0})
test.Married = test.Married.replace({"Yes": 1, "No" : 0})

train.Self_Employed = train.Self_Employed.replace({"Yes": 1, "No" : 0})
test.Self_Employed = test.Self_Employed.replace({"Yes": 1, "No" : 0})

train['Credit_History'].fillna(train['Credit_History'].mode(), inplace=True) # Mode
test['Credit_History'].fillna(test['Credit_History'].mode(), inplace=True) # Mode

train['Self_Employed'].fillna(train['Self_Employed'].mean(), inplace=True) # Mode
test['Self_Employed'].fillna(test['Self_Employed'].mean(), inplace=True) # Mode

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mean(), inplace=True) # Mode
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mean(), inplace=True) # Mode

train['LoanAmount'].fillna(train['LoanAmount'].mean(), inplace=True) # Mean
test['LoanAmount'].fillna(test['LoanAmount'].mean(), inplace=True) # Mean

train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)

train['Married'].fillna(train['Married'].mode()[0], inplace=True)
test['Married'].fillna(test['Married'].mode()[0], inplace=True)

train['Credit_History'].fillna(train['Credit_History'].mean(), inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mean(), inplace=True)

from sklearn.preprocessing import LabelEncoder
feature_col = ['Property_Area','Education', 'Dependents']
le = LabelEncoder()
for col in feature_col:
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])

st.title("LOAN ELGIBILITY CHECH")

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

x_train = train.drop('Loan_Status',axis=1).values
y_train = train['Loan_Status'].values

def user_report():
  Married = st.sidebar.slider('Married',0,1)
  Dependents = st.sidebar.slider('Dependents',0,1)
  Education = st.sidebar.slider('Education',0,1)
  Self_Employed = st.sidebar.slider('Self_Employed',0,1)
  ApplicantIncome = st.sidebar.number_input('ApplicantIncome')
  CoapplicantIncome = st.sidebar.number_input('CoapplicantIncome')
  LoanAmount = st.sidebar.number_input('LoanAmount')
  Loan_Amount_Term = st.sidebar.number_input('Loan_Amount_Term')
  Credit_History = st.sidebar.slider('Credit_History',0,1)
  Property_Area = st.sidebar.slider('Property_Area',0,2)

  user_report = {

      'Married':Married,
      'Dependents':Dependents,
      'Education':Education,
      'Self_Employed':Self_Employed,
      'ApplicantIncome':ApplicantIncome,
      'CoapplicantIncome':CoapplicantIncome,
      'LoanAmount':LoanAmount,
      'Loan_Amount_Term':Loan_Amount_Term,
      'Credit_History':Credit_History,
      'Property_Area':Property_Area,
  }
  report_data = pd.DataFrame(user_report, index=[0])
  return report_data

user_data = user_report()

lr = LogisticRegression()
lr.fit(x_train, y_train)

test = test.drop('Loan_ID',axis=1)
test = test.drop('Gender',axis=1)

y_pred = lr.predict(test)

st.subheader("ACCURACY : ")
st.write(str((lr.score(x_train, y_train))*100)+'%')

user_result = lr.predict(user_data)

st.subheader("LOAN ELIGIBILITY STATUS")

output = ''

if user_result[0] == 0:
  output = 'NOT ELIGIBLE'
else:
  output = 'ELIGIBLE'

st.write(output)

