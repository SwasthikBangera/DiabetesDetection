#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 08:57:07 2023

@author: yennamac
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('/Users/yennamac/Downloads/diabetes_model.sav', 'rb'))

# Creating function for prediction

def diabetes_prediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    
    # Title of the app
    st.title('Diabetes prediction web app')
    
    # Getting input data from user
    Pregnancies = st.text_input("No. of Pregnancies : ")
    Glucose = st.text_input("Gluscose level : ")
    BloodPressure = st.text_input("Blood Pressure level : ")
    SkinThickness = st.text_input("Skin Thickness : ")
    Insulin = st.text_input("Insulin value : ")
    BMI = st.text_input("BMI Value : ")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function : ")
    Age = st.text_input("Age of patient : ")
    
    #Code for prediction
    diagnosis = ''
    
    # Creating button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)


if __name__ == '__main__':
    main()
