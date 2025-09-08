

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "tensorflow==2.20.0"])

import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import pandas as pd
import numpy as np
import streamlit as st

model = tf.keras.models.load_model('model.h5')

with open('onehot_encoder_geo.pickle', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('scaler.pickle', 'rb') as file:
    scaler = pickle.load(file)
with open('label_encoder_gender.pickle', 'rb') as file:
    label_encoder_gender = pickle.load(file)

# Get categories for geography from the encoder
categories_geo = onehot_encoder_geo.categories_[0]

st.title('Customer Churn Prediction')

## User Input
geography = st.selectbox('Geography', categories_geo)
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10, 5)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare Input Data
input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_credit_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}
input_df = pd.DataFrame([input_data])

geo_value = st.selectbox("Select Geography", ["France", "Germany", "Spain"])
gender_value = st.selectbox("Select Gender", ["Male", "Female"])

# Suppose geo_value = "France" (from user input)
geo_encoded = onehot_encoder_geo.transform([[geo_value]])   # 👈 make sure input is 2D

# If encoder output is sparse, convert to array
geo_encoded_df = pd.DataFrame(
    geo_encoded.toarray(),
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)
input_full = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

input_scaled = scaler.transform(input_full)
prediction = model.predict(input_scaled)
churn = prediction[0][0] > 0.5

if churn:
    st.write('Customer is likely to churn.')
else:
    st.write('Customer is not likely to churn.')
st.write(f'Churn probability: {prediction[0][0]:.2f}')