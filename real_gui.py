import streamlit as st
import pandas as pd
import pickle

# Load the trained model from the pickle file
with open("E:\\COLLEGE WORK\\3rd yr\\ML CA2\\model_pickle", 'rb') as file:
    reg = pickle.load(file)

# Function to predict the real estate price
def predict_price(inputs):
    input_df = pd.DataFrame([inputs], columns=['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude'])
    prediction = reg.predict(input_df)[0]
    return prediction

# Streamlit app
def main():
    st.title('Real Estate Price Prediction')
    
    # Input form
    st.header('Enter Parameters')
    transaction_date = st.number_input('Transaction Date')
    house_age = st.number_input('House Age')
    nearest_mrt_distance = st.number_input('Distance to the nearest MRT Station')
    num_convenience_stores = st.number_input('Number of Convenience Stores')
    latitude = st.number_input('Latitude')
    longitude = st.number_input('Longitude')
    
    # Make prediction
    if st.button('Predict'):
        inputs = [transaction_date, house_age, nearest_mrt_distance, num_convenience_stores, latitude, longitude]
        prediction = predict_price(inputs)
        st.success(f'Predicted Price: {prediction:.2f} units')
    
if __name__ == '__main__':
    main()
