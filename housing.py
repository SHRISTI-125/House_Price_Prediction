import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Housing.csv')
df = df[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']]

X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
A_train, A_test, B_train, B_test = train_test_split(df[['area']], y, test_size=0.2, random_state=2)

lr = LinearRegression()
lr.fit(X_train, y_train)

#st.title(Fore.GREEN + "House Price Prediction")
st.markdown(f'<h1 style="color:#ADD8E6;">House Price Prediction</h1>', unsafe_allow_html=True)

# Sidebar 
area_input = st.sidebar.number_input("Enter area in square feet:", value=1000)
bedrooms_input = st.sidebar.number_input("Number of bedrooms:", min_value=1, max_value=8, step=1)
bathrooms_input = st.sidebar.number_input("Number of bathrooms:", min_value=1, max_value=3, step=1)
stories_input = st.sidebar.number_input("Number of stories:", min_value=1, max_value=4, step=1)
parking_input = st.sidebar.number_input("Number of parking spaces:", min_value=0, max_value=3, step=1)

predicted_price = lr.predict([[area_input, bedrooms_input, bathrooms_input, stories_input, parking_input]])
st.write(f"Price Predicted")
st.write(f'<h3 style="color:#32CD32;">Here is the Total Price you have to pay: <h2 style="color:pink;"> ₹ {predicted_price[0]:.2f} </h2></h3>', unsafe_allow_html=True)
print("\n\n")

# Plotting the data
fig, ax = plt.subplots()
plt.style.use(['dark_background'])
plt.suptitle('House Price Prediction showing Best Fit line', color='w')
lr.fit(A_train, B_train)
ax.scatter(df['area'],df['price'],color='blue', label='Price')
ax.plot(A_test,lr.predict(A_test),color='yellow', label='Best fit line (using area only)')
ax.set_xlabel('Area')
ax.set_ylabel('Price')
ax.legend()
# Display 
st.pyplot(fig)

st.write("These prediction is based on training data.")


