# Import necessary libraries
import pandas as pd
import ast
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import streamlit as st
import joblib

# List of file paths for the datasets from various cities(Replace the path of the datasets accordingly)
file_paths = [
    '../data/bangalore_cars.xlsx',
    '../data/delhi_cars.xlsx',
    '../data/kolkata_cars.xlsx',
    '../data/hyderabad_cars.xlsx',
    '../data/jaipur_cars.xlsx',
    '../data/chennai_cars.xlsx'
]

# Function to clean specific columns as per user's request
def clean_column_data(column_data, start_str, end_str):
    """
    Cleans column data by removing specific start and end strings.
    
    Parameters:
    column_data (str): The data to be cleaned.
    start_str (str): The starting substring to remove.
    end_str (str): The ending substring to remove.
    
    Returns:
    Cleaned data, either as a string or evaluated data (list or dictionary).
    """
    try:
        cleaned_data = column_data.replace(start_str, "").replace(end_str, "")
        return ast.literal_eval(cleaned_data) if cleaned_data.startswith("[{") or cleaned_data.startswith("{") else cleaned_data
    except Exception as e:
        return column_data  # Return original data if there's an issue

# Safely evaluate the data in columns containing dictionary-like structures
def safe_literal_eval(val):
    """
    Safely evaluates string data that resembles a dictionary or list.
    
    Parameters:
    val (str): The string to evaluate.
    
    Returns:
    Evaluated value (list, dict, or original string).
    """
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val  # Return the original value if evaluation fails

# Extract relevant details from 'new_car_overview' column
def extract_overview(overview):
    """
    Extracts key-value pairs from the 'new_car_overview' list of dictionaries.
    
    Parameters:
    overview (list): A list of dictionaries with car overview data.
    
    Returns:
    Dictionary with extracted key-value pairs.
    """
    overview_dict = {}
    if isinstance(overview, list):
        for item in overview:
            overview_dict[item['key']] = item.get('value', '')
    return overview_dict

# Extract relevant features from 'new_car_feature' column
def extract_features(feature):
    """
    Extracts key-value pairs from the 'new_car_feature' list of dictionaries.
    
    Parameters:
    feature (list): A list of dictionaries with car feature data.
    
    Returns:
    Dictionary with extracted key-value pairs.
    """
    feature_dict = {}
    if isinstance(feature, list):
        for item in feature:
            feature_dict[item['value']] = item.get('value', '')
    return feature_dict

# Extract relevant specifications from 'new_car_specs' column
def extract_specs(specs):
    """
    Extracts key-value pairs from the 'new_car_specs' list of dictionaries.
    
    Parameters:
    specs (list): A list of dictionaries with car specification data.
    
    Returns:
    Dictionary with extracted key-value pairs.
    """
    specs_dict = {}
    if isinstance(specs, list):
        for item in specs:
            specs_dict[item['key']] = item.get('value', '')
    return specs_dict

# Initialize an empty list to store all processed dataframes from different files
all_dataframes = []

# Loop through each file path to process the data
for file_path in file_paths:
    # Load the Excel file into a pandas DataFrame
    data = pd.read_excel(file_path)

    # Clean specific columns (overview, feature, specs) using defined functions
    data['new_car_overview'] = data['new_car_overview'].apply(lambda x: clean_column_data(str(x), "{'heading': 'Car overview', 'top':", ", 'bottomData': None}"))
    data['new_car_feature'] = data['new_car_feature'].apply(lambda x: clean_column_data(str(x), "{'heading': 'Features', 'top':", " 'commonIcon': 'https://stimg.cardekho.com/pwa/img/vdpN/tickG.svg'}"))
    data['new_car_specs'] = data['new_car_specs'].apply(lambda x: clean_column_data(str(x), "{'heading': 'Specifications', 'top':", ", 'commonIcon': ''}"))

    # Safely evaluate string literals into Python data structures
    data['new_car_detail'] = data['new_car_detail'].apply(safe_literal_eval)
    data['new_car_overview'] = data['new_car_overview'].apply(safe_literal_eval)
    data['new_car_feature'] = data['new_car_feature'].apply(safe_literal_eval)
    data['new_car_specs'] = data['new_car_specs'].apply(safe_literal_eval)

    # Normalize the 'new_car_detail' column and extract detailed information
    df_details = pd.json_normalize(data['new_car_detail'])

    # Extract overview, features, and specs into separate DataFrames
    df_overview = pd.DataFrame(data['new_car_overview'].apply(extract_overview).tolist())
    df_features = pd.DataFrame(data['new_car_feature'].apply(extract_features).tolist())
    df_specs = pd.DataFrame(data['new_car_specs'].apply(extract_specs).tolist())

    # Combine the extracted data with the original DataFrame
    df_final = pd.concat([df_details, df_overview, df_features, df_specs, data['car_links']], axis=1)

    # Add a column for the source city based on the file name
    city_name = os.path.basename(file_path).split('_')[0].capitalize()
    df_final['City'] = city_name

    # Append the structured DataFrame to the list of all data
    all_dataframes.append(df_final)

# Concatenate all processed DataFrames into one final DataFrame
final_data = pd.concat(all_dataframes, ignore_index=True)

# List of columns to drop based on business logic or irrelevance
columns_to_drop = [
    'it', 'owner', 'oem', 'centralVariantId', 'variantName', 
    'priceActual', 'priceSaving', 'priceFixedText', 
    'trendingText.imgUrl', 'trendingText.heading', 
    'trendingText.desc', 'RTO', 'Ownership', 
    'Year of Manufacture', 'car_links','Transmission'
]

# Drop the unnecessary columns from the DataFrame
final_cleaned_data = final_data.drop(columns=columns_to_drop, errors='ignore')

# Renaming columns for better clarity and consistency
final_cleaned_data = final_cleaned_data.rename(columns={
    'ft': 'fuel_type',
    'bt': 'body_type',
    'km': 'kilometers_driven',
    'transmission': 'transmission_type',
    'ownerNo': 'owner_number',
    'owner': 'ownership_status',
    'oem': 'manufacturer',
    'modelYear': 'model_year',
    'Kms Driven': 'kilometers_driven_clean',
    'Engine Displacement': 'engine_displacement',
    'Year of Manufacture': 'year_of_manufacture'
})

# Save the cleaned and structured DataFrame to an Excel file
output_excel_path = '../data/structured_cars_data_combined.xlsx'  # Adjust the file path if necessary
final_cleaned_data.to_excel(output_excel_path, index=False)

# Remove commas in 'kilometers_driven' and convert to numeric
final_cleaned_data['kilometers_driven'] = final_cleaned_data['kilometers_driven'].str.replace(',', '').astype(int)

# Function to clean the 'price' column by removing currency symbols and converting to numeric
def clean_price(price):
    """
    Cleans price values by removing currency symbols and converting to numeric.
    
    Parameters:
    price (str): The price value as a string.
    
    Returns:
    Numeric value of the price.
    """
    price = price.replace('₹', '').replace(',', '').strip()
    if 'Lakh' in price:
        return float(price.replace('Lakh', '').strip()) * 100000  # Convert Lakh to numeric
    if 'Crore' in price:
        return float(price.replace('Crore', '').strip()) * 10000000  # Convert Crore to numeric
    return float(price)

# Apply the price cleaning function
final_cleaned_data['price'] = final_cleaned_data['price'].apply(clean_price)

# Remove unnecessary columns
final_cleaned_data = final_cleaned_data.drop(columns=['Registration Year', 'Fuel Type', 'kilometers_driven_clean'])

# Clean 'Seats' and 'engine_displacement' columns by extracting numbers and handling NaN values
final_cleaned_data['Seats'] = final_cleaned_data['Seats'].str.extract('(\d+)').fillna(0).astype(int)
final_cleaned_data['engine_displacement'] = final_cleaned_data['engine_displacement'].str.extract('(\d+)').fillna(0).astype(int)

# Handle missing values for numerical and categorical columns
missing_values = final_cleaned_data.isnull().sum()

# Fill missing values in numerical columns with the mean
num_cols = ['kilometers_driven', 'owner_number', 'price', 'Seats', 'engine_displacement']
for col in num_cols:
    final_cleaned_data[col].fillna(final_cleaned_data[col].mean(), inplace=True)

# Fill missing values in categorical columns with the mode
cat_cols = ['fuel_type', 'body_type', 'transmission_type', 'model', 'Insurance Validity', 'City']
for col in cat_cols:
    if final_cleaned_data[col].isnull().sum() > 0:
        final_cleaned_data[col].fillna(final_cleaned_data[col].mode()[0], inplace=True)

# Verify if there are any missing values left
missing_values_after = final_cleaned_data.isnull().sum()

# One-hot encoding for nominal categorical variables
nominal_columns = ['fuel_type', 'body_type', 'transmission_type', 'Insurance Validity', 'City', 'model']
final_cleaned_data = pd.get_dummies(final_cleaned_data, columns=nominal_columns)

# Label encoding for ordinal categorical variables
ordinal_columns = ['owner_number']
label_encoder = LabelEncoder()
for col in ordinal_columns:
    final_cleaned_data[col] = label_encoder.fit_transform(final_cleaned_data[col])

# Descriptive statistics for the cleaned data
descriptive_stats = final_cleaned_data.describe()
modes = final_cleaned_data.mode().iloc[0]
medians = final_cleaned_data.median()
descriptive_stats.loc['median'] = medians
descriptive_stats.loc['mode'] = modes

# Save descriptive statistics to Excel
output_descexcel_path = '../data/descriptive_statistics.xlsx'  # Adjust the file path if necessary
descriptive_stats.to_excel(output_descexcel_path, index=True)

# Plotting various visualizations for data analysis
sns.set(style="whitegrid")

# Scatter plot: Kilometers Driven vs. Price
plt.figure(figsize=(8,6))
sns.scatterplot(x='kilometers_driven', y='price', data=final_cleaned_data)
plt.title('Scatter Plot: Kilometers Driven vs. Price')
plt.show()

# Histogram: Distribution of Model Year
plt.figure(figsize=(8,6))
sns.histplot(final_cleaned_data['model_year'], bins=20, kde=True)
plt.title('Histogram: Distribution of Model Year')
plt.show()

# Box Plot: Price distribution by Owner Number
plt.figure(figsize=(8,6))
sns.boxplot(x='owner_number', y='price', data=final_cleaned_data)
plt.title('Box Plot: Price vs. Owner Number')
plt.show()

# Correlation Heatmap for numerical features
plt.figure(figsize=(10,8))
correlation_matrix = final_cleaned_data[['kilometers_driven', 'price', 'model_year', 'Seats', 'engine_displacement']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Splitting the dataset into training and testing sets (80% train, 20% test)
X = final_cleaned_data.drop(columns='price')
y = final_cleaned_data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function for hyperparameter tuning using GridSearchCV or RandomizedSearchCV
def hyperparameter_tuning(model, param_grid, X_train, y_train, search_type='grid'):
    """
    Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
    
    Parameters:
    model: The machine learning model to tune.
    param_grid (dict): Hyperparameters to tune.
    X_train (DataFrame): Training features.
    y_train (Series): Training target.
    search_type (str): Search type ('grid' for GridSearchCV, 'random' for RandomizedSearchCV).
    
    Returns:
    The best model found through hyperparameter tuning.
    """
    if search_type == 'grid':
        search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    elif search_type == 'random':
        search = RandomizedSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, n_iter=20, random_state=42)
    
    search.fit(X_train, y_train)
    print(f"Best Params for {model.__class__.__name__} using {search_type} search: {search.best_params_}")
    return search.best_estimator_

# Define various regression models and their hyperparameters
models = {
    "Linear Regression": LinearRegression(),
    "Lasso (L1)": Lasso(max_iter=10000, random_state=42),
    "Ridge (L2)": Ridge(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# Hyperparameter grids for models that support tuning
param_grids = {
    "Random Forest": {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    "Gradient Boosting": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    },
    "Lasso (L1)": {
        'alpha': [0.01, 0.1, 1, 10, 100]
    },
    "Ridge (L2)": {
        'alpha': [0.01, 0.1, 1, 10, 100]
    }
}

# Hyperparameter tuning, model training, and evaluation
best_model_name = None
best_model = None
best_test_mse = float('inf')
best_test_r2 = float('-inf')
best_test_mae = float('inf')

# Iterate through models, perform tuning, and evaluate performance
for model_name, model in models.items():
    # Perform hyperparameter tuning if applicable
    if model_name in param_grids:
        param_grid = param_grids[model_name]
        model = hyperparameter_tuning(model, param_grid, X_train, y_train, search_type='grid')
    
    # Train the model on the training set
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_test_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"Model: {model_name}")
    print(f"Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}, Test R²: {test_r2:.4f}")
    print('-' * 50)
    
    # Track the best model based on test metrics
    if test_mse < best_test_mse and test_mae < best_test_mae and test_r2 > best_test_r2:
        best_model_name = model_name
        best_model = model
        best_test_mse = test_mse
        best_test_mae = test_mae
        best_test_r2 = test_r2

# Output the best model's details
print(f"Best Model: {best_model_name}")
print(f"Best Model Test MSE: {best_test_mse:.4f}, Best Model Test MAE: {best_test_mae:.4f}, Best Model Test R²: {best_test_r2:.4f}")

# Save the best model using joblib for later use in predictions
joblib.dump(best_model, 'best_model.pkl')
print("Best model saved as 'best_model.pkl'.")

# Save the column names used in the model to ensure consistency during future predictions
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Model columns saved as 'model_columns.pkl'.")
