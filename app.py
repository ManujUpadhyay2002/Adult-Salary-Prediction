import streamlit as st
import pandas as pd
import pipe 

st.title('Income Analysis')

dict_data = {'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5, '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9,
             'Some-college': 10, 'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14, 'Prof-school': 15, 'Doctorate': 16}

age = st.text_input('Your Age', )

workclass = st.selectbox(
    'Select Your Work Class',
    ('Private', 'Local-gov', 'other', 'Self-emp-not-inc', 'Federal-gov',
     'State-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked'))

education = st.selectbox(
    'Select Your Education',
    ('Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate'))

maritalstatus = st.selectbox(
    'Select Your maritalstatus ',
    ('Never-married', 'Married-civ-spouse', 'Widowed', 'Divorced',
     'Separated', 'Married-spouse-absent', 'Married-AF-spouse'))

occupation = st.selectbox(
    'Select Your occupation',
    ('Machine-op-inspct', 'Farming-fishing', 'Protective-serv', 'Other',
     'Other-service', 'Prof-specialty', 'Craft-repair', 'Adm-clerical',
     'Exec-managerial', 'Tech-support', 'Sales', 'Priv-house-serv',
     'Transport-moving', 'Handlers-cleaners', 'Armed-Forces'))

relationship = st.selectbox(
    'Select Your relationship',
    ('Own-child', 'Husband', 'Not-in-family', 'Unmarried', 'Wife',
     'Other-relative'))

race = st.selectbox(
    'Select Your race',
    ('Black', 'White', 'Asian-Pac-Islander', 'Other',
     'Amer-Indian-Eskimo'))

gender = st.selectbox(
    'Select Your gender',
    ('Male', 'Female'))

h_p_w = st.text_input('Hours per week you work', )

data = pd.DataFrame([age,workclass,dict_data[education],maritalstatus,occupation,relationship,race,gender,h_p_w]).transpose()

if st.button('Predict'):
    # 1. predict
    result = pipe.predict(data)
    # 2. display
    if result == 1:
        st.header("Salary Greater-than 50,000")
    else:
        st.header("Salary Less-than 50,000")
