
# app.py
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder


# Set page title
st.title("Crop Recommendation System")

st.header("Enter Soil and Environmental Details")

# Input fields for each parameter

N = st.slider("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0)
P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=400.0, value=100.0)


with open(r"C:\Users\user\Desktop\pyprac\logistic\project1\crop-prediction\modelsave.pkl",'rb') as f:
    model = pickle.load(f)
    

with open(r"C:\Users\user\Desktop\pyprac\logistic\project1\crop-prediction\standarscaler.pkl",'rb') as f:
    sc = pickle.load(f)
    

with open(r"C:\Users\user\Desktop\pyprac\logistic\project1\crop-prediction\LabelEncoder.pkl",'rb') as f:
    le = pickle.load(f)

df = pd.DataFrame({"N":[N],"P":[P],"K":[K],"temperature":[K],"humidity":[humidity],"ph":[ph],"rainfall":[rainfall]})

scaled_df = sc.transform(df)

y_pred = model.predict(scaled_df)
st.write(le.inverse_transform(y_pred)[0])

