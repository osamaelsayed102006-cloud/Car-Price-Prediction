import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗"
)

model = joblib.load("models/final_model.pkl")

df = pd.read_csv("data/car_ads_details_kaggle.csv")

st.title("🚗 Car Price Prediction")
st.write("Select the car details:")

brand = st.selectbox(
    "Brand",
    sorted(df["Brand"].dropna().unique())
)

brand_data = df[
    df["Brand"] == brand
]

available_models = brand_data[
    "Model"
].dropna().unique()

model_name = st.selectbox(
    "Model",
    sorted(available_models)
)

car_data = brand_data[
    brand_data["Model"] == model_name
]

available_engine_capacities = car_data[
    "Engine Capacity (CC)"
].dropna().unique()

engine_capacity = st.selectbox(
    "Engine Capacity (CC)",
    sorted(available_engine_capacities)
)

car_data = car_data[
    car_data["Engine Capacity (CC)"] == engine_capacity
]

available_body_types = car_data[
    "Body Type"
].dropna().unique()

body_type = st.selectbox(
    "Body Type",
    sorted(available_body_types)
)

car_data = car_data[
    car_data["Body Type"] == body_type
]

available_fuel_types = car_data[
    "Fuel Type"
].dropna().unique()

fuel_type = st.selectbox(
    "Fuel Type",
    sorted(available_fuel_types)
)

car_data = car_data[
    car_data["Fuel Type"] == fuel_type
]

available_transmissions = car_data[
    "Transmission Type"
].dropna().unique()

transmission_type = st.selectbox(
    "Transmission Type",
    sorted(available_transmissions)
)

kilometers = st.number_input(
    "Kilometers",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

year = st.number_input(
    "Year",
    min_value=1990,
    max_value=2026,
    value=2020,
    step=1
)

st.divider()

if st.button("🚀 Predict Price"):

    input_data = pd.DataFrame({
        "Brand": [brand],
        "Model": [model_name],
        "Kilometers": [kilometers],
        "Year": [year],
        "Fuel Type": [fuel_type],
        "Transmission Type": [transmission_type],
        "Engine Capacity (CC)": [engine_capacity],
        "Body Type": [body_type]
    })

    prediction = model.predict(input_data)

    predicted_price = prediction[0]

    st.subheader("🚗 Car Details")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Brand:** {brand}")
        st.write(f"**Model:** {model_name}")
        st.write(f"**Engine:** {engine_capacity} CC")
        st.write(f"**Body Type:** {body_type}")

    with col2:
        st.write(f"**Fuel:** {fuel_type}")
        st.write(f"**Transmission:** {transmission_type}")
        st.write(f"**Kilometers:** {kilometers:,.0f} KM")
        st.write(f"**Year:** {year}")

    st.divider()

    st.subheader("💰 Predicted Price")

    st.success(
        f"{predicted_price:,.0f} EGP"
    )