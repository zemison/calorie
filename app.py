import streamlit as st
import numpy as np
import pandas as pd
import pickle

# laod model
rfr = pickle.load(open('rfr.pkl','rb'))
x_train = pd.read_csv('X_train.csv')

def pred(Gender,Age,Height,Weight,Duration,Heart_rate,Body_temp):
    features = np.array([[Gender,Age,Height,Weight,Duration,Heart_rate,Body_temp]])
    prediction = rfr.predict(features).reshape(1,-1)
    return prediction[0]


# web app
# Gender Age Height Weight Duration Heart_Rate Body_Temp
st.title("Calories Burn Prediction")

Gender = st.selectbox('Gender', x_train['Gender'])
Age = st.selectbox('Age', x_train['Age'])
Height = st.selectbox('Height', x_train['Height'])
Weight = st.selectbox('Weight', x_train['Weight'])
Duration = st.selectbox('Duration (minutes)', x_train['Duration'])
Heart_rate = st.selectbox('Heart Rate (bpm)', x_train['Heart_Rate'])
Body_temp = st.selectbox('Body Temperature', x_train['Body_Temp'])

result = pred(Gender,Age,Height,Weight,Duration,Heart_rate,Body_temp)

if st.button('predict'):
    if result:
        st.write("You have consumed this calories :",result)
