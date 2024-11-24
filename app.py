import streamlit as st
import pandas as pd

# Load your CSV files
df_calories = pd.read_csv('calories.csv')
df_exercise = pd.read_csv('exercise (3).csv')

st.title("Calorie Tracker")

# Show the calorie data
st.write("Calories Data:")
st.write(df_calories)

# Show exercise data
st.write("Exercise Data:")
st.write(df_exercise)
