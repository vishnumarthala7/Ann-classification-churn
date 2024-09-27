import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle
import streamlit as st

import tensorflow as tf


#load the model
model=tf.keras.models.load_model('model.h5')
with open("label_encoder_gender.pkl","rb") as file:
    label_encoder_gender=pickle.load(file)
with open("onehot_encoder_geo.pkl","rb") as file:
    one_hot_encoder=pickle.load(file)
with open("scaler.pkl","rb") as file:
    scaler=pickle.load(file)

# streamlit.app
st.title('customer churn prediction')

#user input
geography=st.selectbox("Geography",one_hot_encoder.categories_[0])
gender=st.selectbox("Gender",label_encoder_gender.classes_)
balance=st.number_input("Balance")
age=st.slider("Age",18,92)
credit_score=st.number_input("Credit Score")
estimated_salary=st.number_input("Estimated salary")
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider("Number of products",1,4)
has_cr_card=st.selectbox("Has credit card",[0,1])
is_active_member=st.selectbox("Is active Member",[0,1])
 
input_data=pd.DataFrame({
    "CreditScore":[credit_score],
    
    "Gender":[label_encoder_gender.transform([gender])[0]],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[num_of_products],
    "HasCrCard":[has_cr_card],
    "IsActiveMember":[is_active_member],
    "EstimatedSalary":[estimated_salary]

})
geo_encoder=one_hot_encoder.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoder,columns=one_hot_encoder.get_feature_names_out(["Geography"]))


input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
input_data_scaled=scaler.transform(input_data)

preduction=model.predict(input_data_scaled)
pred=preduction[0][0]

if pred>0.5:
    st.write("the custmor is likely to churn")
else:
    st.write("the custmor not likely churn")