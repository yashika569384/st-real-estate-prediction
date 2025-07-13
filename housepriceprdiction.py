import streamlit as st
import pickle
import numpy as np
import pandas as pd
# Load external CSS file
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stApp {
        background-color:light pink;
        padding: 20px;
        border-radius: 10px;
    }
    h1 {
        color: red;
        text-align: center;
    }
    .st-emotion-cache-ocsh0s {
    border-radius:50px;
    background-color:blue;}
    .st-emotion-cache-89jlt8 p{
    font-size:20px;
    
    }
    
    </style>
    """,
    unsafe_allow_html=True
)


st.title(":rainbow[ Real Estate Price Prediction]🏡")
st.header("Predict price")


with open('banglore_home_prices_model1.pickle','rb') as f:
  data=pickle.load(f)
model=data['model']
locations=data['locations']


location=st.selectbox("select location",locations)

bath = st.slider("🛁 Number of Bathrooms", 1, 5, 1)
bhk=st.slider("Number of BHK" ,1,5,1)
sqft=st.number_input("Enter Square Feet Area",min_value=300, max_value=10000)


if st.button("predict price"):
  x=np.zeros(len(locations)+3)
  x[0]=sqft
  x[1]=bath
  x[2]=bhk

  loc_index=locations.index(location)  if location in locations else -1
  if loc_index>=0:
    x[loc_index+3]=1

  predicted_price=model.predict([x])[0]

  st.write(f"🏡 Estimated Price: ₹{predicted_price:,.2f} Lakh")


