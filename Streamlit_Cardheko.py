# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import base64

# Function to set a background image for the app
def set_background_image(image_path):
    """
    Sets a background image for the Streamlit app using CSS.

    Parameters:
    - image_path (str): The file path of the background image.
    """
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    background_style = f"""
    <style>
    .stApp {{
        background-image: url('data:image/jpg;base64,{encoded_image}');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Set a background image for the app (replace 'cars.jpg' with your image path)
set_background_image('cars.jpg')

# Load the pre-trained model and model columns
@st.cache_resource
def load_model_and_columns():
    """
    Load the trained machine learning model and column names from disk.

    Returns:(replace 'best_model.pkl' and 'model_columns1.pkl' with your pickle files which is created during model prediction)
    - model (object): The pre-trained model.
    - model_columns (list): List of column names used for model prediction.
    """
    model = joblib.load('best_model.pkl')
    model_columns = joblib.load('model_columns1.pkl')
    return model, model_columns

# Load available car models from dataset(replace '../data/df_capped.xlsx' with your excel path)
@st.cache_data
def load_car_models():
    """
    Load car models from the dataset.

    Returns:
    - car_models_filtered (list): List of car model names.
    """
    try:
        df = pd.read_excel('../data/df_capped.xlsx')
        car_model_columns = [col for col in df.columns if col.startswith('model_')]
        car_models = [col.replace('model_', '') for col in car_model_columns]
        car_models_filtered = [model for model in car_models if "year" not in model.lower()]
        return car_models_filtered
    except FileNotFoundError:
        st.error("Dataset not found! Make sure the file path is correct.")
        return []

# Preprocess the input data for prediction
def preprocess_input(data, selected_car_model):
    """
    Preprocess the input data to match the format expected by the model.

    Parameters:
    - data (DataFrame): User input data.
    - selected_car_model (str): Selected car model by the user.

    Returns:
    - data (DataFrame): Preprocessed data ready for prediction.
    """
    categorical_columns = ['fuel_type', 'body_type', 'transmission_type', 'insurance_validity', 'city']
    data = pd.get_dummies(data, columns=categorical_columns)

    car_model_column = f'model_{selected_car_model}'
    data[car_model_column] = 1

    missing_cols = set(model_columns) - set(data.columns)
    for col in missing_cols:
        data[col] = 0

    data = data[model_columns]
    return data

# Initialize the model and column names
model, model_columns = load_model_and_columns()
car_models = load_car_models()

# App title and input UI
st.markdown("<h1 class='pulse-effect'>Car Dheko: Used Car Price Prediction</h1>", unsafe_allow_html=True)

# Create sliders and input boxes for numerical input
def create_slider_and_textbox(label, min_val, max_val, default_val, step, help_text):
    """
    Create a synchronized slider and text box for user input.

    Parameters:
    - label (str): Label for the input.
    - min_val (int): Minimum value for the input.
    - max_val (int): Maximum value for the input.
    - default_val (int): Default value for the input.
    - step (int): Step size for the slider.
    - help_text (str): Help text for the input.

    Returns:
    - int: Final user-selected value.
    """
    if f"{label}_slider" not in st.session_state:
        st.session_state[f"{label}_slider"] = default_val
    if f"{label}_text" not in st.session_state:
        st.session_state[f"{label}_text"] = str(default_val)

    def slider_changed():
        st.session_state[f"{label}_text"] = str(st.session_state[f"{label}_slider"])

    def text_changed():
        try:
            val = int(st.session_state[f"{label}_text"])
            if min_val <= val <= max_val:
                st.session_state[f"{label}_slider"] = val
            else:
                st.error(f"Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            st.error(f"Please enter a valid number for {label.lower()}.")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.slider(
            f"{label} (Slider)", 
            min_value=min_val, 
            max_value=max_val, 
            value=st.session_state[f"{label}_slider"], 
            step=step,
            help=help_text,
            key=f"{label}_slider",
            on_change=slider_changed
        )

    with col2:
        text_input = st.text_input(
            f"{label} (Text)", 
            value=st.session_state[f"{label}_text"],
            help=f"Alternatively, enter the {label.lower()} directly",
            key=f"{label}_text",
            on_change=text_changed
        )

    if st.session_state[f"{label}_text"] != text_input:
        st.experimental_rerun()

    return st.session_state[f"{label}_slider"]

# Collect user input for car details
kilometers_driven = create_slider_and_textbox("Kilometers Driven", 0, 300000, 10000, 1000, "Enter the number of kilometers the car has driven.")
owner_number = create_slider_and_textbox("Owner Number", 1, 4, 1, 1, "Select how many owners the car has had.")
model_year = create_slider_and_textbox("Model Year", 1990, 2023, 2020, 1, "Enter the model year of the car.")
seats = create_slider_and_textbox("Seats", 2, 7, 5, 1, "Enter the number of seats in the car.")
engine_displacement = create_slider_and_textbox("Engine Displacement (in cc)", 500, 5000, 1500, 50, "Enter the engine displacement in cubic centimeters (cc).")

# Dropdowns for categorical input
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'Electric'], help="Select the type of fuel the car uses.")
body_type = st.selectbox("Body Type", ['SUV', 'Sedan', 'Hatchback', 'Convertible', 'Coupe'], help="Select the body type of the car.")
transmission_type = st.selectbox("Transmission Type", ['Manual', 'Automatic'], help="Select the type of transmission.")
insurance_validity = st.selectbox("Insurance Validity", ['Yes', 'No'], help="Select whether the car's insurance is valid.")
city = st.selectbox("City", ['Bangalore', 'Delhi', 'Kolkata', 'Hyderabad', 'Jaipur', 'Chennai'], help="Select the city where the car is being sold.")
car_model = st.selectbox("Car Model", car_models, help="Select the model of the car.")

# Button to predict the price
if st.button("Predict Price"):
    input_data = {
        'kilometers_driven': kilometers_driven,
        'owner_number': owner_number,
        'model_year': model_year,
        'seats': seats,
        'engine_displacement': engine_displacement,
        'fuel_type': fuel_type,
        'body_type': body_type,
        'transmission_type': transmission_type,
        'insurance_validity': insurance_validity,
        'city': city
    }

    input_df = pd.DataFrame([input_data])
    processed_input = preprocess_input(input_df, car_model)

    if set(model_columns) == set(processed_input.columns):
        prediction = model.predict(processed_input)[0]
        st.markdown(f"<h2 class='fade-in-grow'>🎯 The predicted price for the car is ₹{np.round(prediction, 2)}</h2>", unsafe_allow_html=True)

# Sidebar instructions
st.sidebar.title("📜 Instructions")
st.sidebar.markdown("""
<div class="sidebar-instructions">
<ol>
    <li>Input the car details (kilometers driven, number of owners, etc.).</li>
    <li>Choose the appropriate categorical options (fuel type, body type, etc.).</li>
    <li>Click 'Predict Price' to get the estimated price of the car.</li>
    <li>Ensure that all input values are accurate for the best prediction results.</li>
</ol>
</div>
""", unsafe_allow_html=True)
