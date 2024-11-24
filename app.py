import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Set page configuration
st.set_page_config(page_title="Calories Prediction", layout="wide")

# Title of the web app
st.title("Calories Prediction Using XGBoost")

# Load and display the dataset
@st.cache
def load_data():
    calories_data = pd.read_csv('path_to_your_calories_data.csv')  # Replace with your actual file path
    return calories_data

# Load data
calories_data = load_data()

# Display first 5 rows of data
st.subheader("Data Preview")
st.write(calories_data.head())

# Show the shape and info of the dataset
st.subheader("Data Info")
st.write(f"Shape of the dataset: {calories_data.shape}")
st.write(f"Data Types: {calories_data.dtypes}")

# Show some basic statistics
st.subheader("Data Description")
st.write(calories_data.describe())

# Visualizing Correlation Heatmap
st.subheader("Correlation Heatmap")
plt.figure(figsize=(10, 6))
sns.heatmap(calories_data.corr(), annot=True, cmap="coolwarm", fmt='.2f')
st.pyplot()

# Visualizing distribution of Calories
st.subheader("Calories Distribution")
plt.figure(figsize=(10, 6))
sns.histplot(calories_data['Calories'], kde=True)
st.pyplot()

# Preprocessing the data
st.subheader("Data Preprocessing")
calories_data['Gender'] = calories_data['Gender'].map({'male': 1, 'female': 0})
X = calories_data.drop(columns=['User_ID', 'Calories'], axis=1)
Y = calories_data['Calories']

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Display the shape of train and test sets
st.write(f"Shape of X_train: {X_train.shape}")
st.write(f"Shape of X_test: {X_test.shape}")

# Train the model
model = XGBRegressor()

# Train the model using training data
model.fit(X_train, y_train)

# Predict the output for test data
y_pred = model.predict(X_test)

# Display model performance
st.subheader("Model Performance")
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
st.write(f"Mean Squared Error: {mse}")
st.write(f"R-Squared Score: {r2}")

# Show predicted vs actual graph
st.subheader("Predicted vs Actual")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Actual vs Predicted Calories")
st.pyplot()

# Save the model
import pickle
pickle.dump(model, open('calories_model.pkl', 'wb'))

st.write("Model has been saved successfully!")
