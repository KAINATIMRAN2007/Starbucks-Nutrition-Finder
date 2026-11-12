
import streamlit as st
import pandas as pd
import pickle
import json

st.set_page_config(page_title="Starbucks Nutrition Finder", page_icon="☕")

with open("starbucks_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("starbucks_metadata.json", "r") as f:
    metadata = json.load(f)

st.title("Starbucks Nutrition Finder")
st.write("Predict the calorie category of your Starbucks drink.")

st.subheader("Enter Nutrition Information")

values = {}

for feature in metadata["feature_names"]:
    values[feature] = st.number_input(
        feature,
        min_value=0.0,
        value=0.0
    )

if st.button("Predict Calorie Category"):
    input_data = pd.DataFrame([values])
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Category: {prediction}")
