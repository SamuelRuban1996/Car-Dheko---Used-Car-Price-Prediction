
# Car Dheko - Used Car Price Prediction

## Overview
This project aims to predict the prices of used cars based on data collected from six major Indian cities: Bangalore, Delhi, Kolkata, Hyderabad, Jaipur, and Chennai. The dataset contains various car attributes, including kilometers driven, model year, engine displacement, and more. The primary goal is to build a predictive model that can estimate the price of used cars based on these attributes.

## Project Workflow
1. **Data Collection**: Datasets were collected from six cities.
2. **Data Cleaning & Preprocessing**: 
   - Cleaned `new_car_overview`, `new_car_feature`, and `new_car_specs` columns.
   - Filled missing values using the mean for numerical columns and the mode for categorical columns.
   - One-hot encoding for nominal categorical variables and label encoding for ordinal variables.
   - Outliers were capped using the 5th and 95th percentiles.
3. **Feature Engineering**: 
   - Extracted relevant features such as kilometers driven, model year, seats, and engine displacement.
   - Scaled numerical features using Min-Max Scaling.
4. **Model Building**: 
   - Tested multiple regression models: Linear Regression, Lasso, Ridge, Decision Tree, Random Forest, Gradient Boosting.
   - Performed hyperparameter tuning using GridSearchCV for selected models.
5. **Model Evaluation**: 
   - Models were evaluated based on Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.
   - The best model, Gradient Boosting, was saved for future use.

## Best Model
The Gradient Boosting model achieved the best performance with the following metrics:
- Test MSE: 0.0000
- Test MAE: 0.0021
- Test R²: 0.9601

## pkl Files
-The model and column data have been saved as `best_model.pkl` and `model_columns.pkl` respectively.
-Please refer 'pklfiles.zip' for extraction of `best_model.pkl` and `model_columns.pkl` files

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/car-dheko-price-prediction.git
   cd car-dheko-price-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Requirements

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
streamlit
```

## Streamlit Application Guide

### Overview
The Streamlit application provides an interactive interface for users to input car details and receive predicted prices.

### How to Use
1. **Enter Car Details**: Input the following details:
   - Kilometers Driven
   - Owner Number
   - Model Year
   - Seats
   - Engine Displacement
   - Fuel Type, Body Type, Transmission Type, Insurance Validity, City, Car Model
2. **Predict Price**: Click the "Predict Price" button to receive the estimated price.

### Running the App
```bash
streamlit run app.py
```

## Important Documents
1. [Project Documentation](docs/Project_Documentation.docx)
2. [Visualizations and Analysis Report](docs/Visualizations_and_Analysis_Report.docx)
3. [Final Predictive Model User Guide](docs/Final_Predictive_Model_User_Guide.docx)
4. [Justification of Model Selection](docs/Justification_of_Model_Selection.docx)

---

### Visualizations
1. Scatter Plot: Kilometers Driven vs Price
2. Histogram: Model Year Distribution
3. Box Plot: Price vs Number of Owners
4. Correlation Heatmap of Numerical Features

These visualizations help in understanding the dataset and the relationships between various features.
