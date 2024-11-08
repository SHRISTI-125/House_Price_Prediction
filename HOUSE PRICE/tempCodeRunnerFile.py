import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Housing.csv')

df['guestroom'] = df['guestroom'].map({'yes': 1, 'no': 0})
df['furnishingstatus'] = df['furnishingstatus'].map({'furnished': 1, 'semi-furnished': 0})

X = df[['area', 'bedrooms', 'guestroom', 'stories', 'parking', 'furnishingstatus']]
y = df[['price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

lr = LinearRegression()
lr.fit(X_train, y_train)

st.title("House Price Prediction")

area_input = st.sidebar.number_input("Enter area in square feet:", value=1000)
bedrooms = st.sidebar.number_input("Number of bedrooms:", min_value=1, max_value=10, step=1)
guestroom = st.sidebar.selectbox("Guest room available?", ("Yes", "No"))
stories = st.sidebar.number_input("Number of stories:", min_value=1, max_value=4, step=1)
parking_count = st.sidebar.number_input("Parking spaces:", min_value=0, max_value=3, step=1)
furnishingstatus = st.sidebar.selectbox("Furnished status:", ("Furnished", "Semi-Furnished"))

guest_room_value = 1 if guestroom == "Yes" else 0
furnishingstatus_value = 1 if furnishingstatus == "Furnished" else 0

predicting_price = lr.predict([[area_input, bedrooms, guest_room_value, stories, parking_count, furnishingstatus_value]])
st.write(f"Predicted price for the given inputs: {predicting_price[0][0]:.2f}")

fig, ax = plt.subplots()
ax.scatter(df['area'], df['price'], color='pink', label='Actual data')
ax.plot(X_test['area'], lr.predict(X_test), color='purple', label='Regression line')
ax.set_xlabel('Area')
ax.set_ylabel('Price')
ax.legend()

st.pyplot(fig)