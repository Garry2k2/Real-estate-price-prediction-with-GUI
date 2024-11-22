# Real Estate Price Prediction 🏡

## Overview
This project is a machine learning-based application designed to predict real estate prices per unit area based on various factors. Using a dataset that includes information on house age, proximity to MRT stations, convenience stores, and geographical location (latitude and longitude), the model accurately estimates house prices. The application leverages linear regression to generate predictions and visualize the results.

## Motivation
The goal of this project was to explore how various factors influence real estate prices, providing insights for prospective buyers, sellers, and investors. With a clear understanding of the dataset, we aimed to create a predictive model that simplifies the analysis of housing data and forecasts accurate property values.

## Dataset
The project utilizes a real estate dataset, which includes multiple features related to housing prices. Each entry provides the following attributes:
- House age
- Distance to the nearest MRT station
- Number of nearby convenience stores
- Latitude and longitude coordinates
- House price per unit area

## Technical Aspect
The project is divided into several parts:

1. **Data Preparation**: Cleaning and preprocessing the dataset to remove noise, outliers, and irrelevant information. Key features were selected based on their impact on housing prices.
2. **Model Training**: A linear regression model was built using Scikit-learn to predict real estate prices. Feature engineering included standardizing and normalizing relevant attributes.
3. **Data Visualization**: Tools like Seaborn and Matplotlib were used to visualize data distributions, relationships, and model predictions.
4. **Performance Analysis**: Evaluation of the model using key metrics to assess prediction accuracy.

## Installation
The project requires Python 3.7+. To set up the environment, follow these steps:

1. Clone the repository.
2. Navigate to the project directory.
3. Install the required packages by running:

```bash
pip install -r requirements.txt
```

## Run
To execute the application locally, follow these instructions:

1. **Data Preprocessing**: Load and clean the dataset using the provided preprocessing scripts.
2. **Model Training**: Train the linear regression model using the dataset.
3. **Execution**: Use the trained model to predict real estate prices by running:


## Directory Tree
```
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── model
│   │   └── real_estate_model.pkl
│   ├── static
│   └── templates
├── config
│   ├── __init__.py
├── data
│   └── Real estate.csv
├── requirements.txt
├── README.md
└── LICENSE
```

## Model Evaluation
The model's performance is evaluated using several metrics:

- **Mean Absolute Error (MAE)**: Measures the average magnitude of prediction errors.
- **Mean Squared Error (MSE)**: Computes the average squared difference between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: Measures the standard deviation of prediction errors.

These metrics provide a clear assessment of how well the model predicts housing prices based on the given dataset.

## Performance Analysis
The analysis of the prediction model involved:

1. **Visual Inspection**: Data visualization to understand the relationship between key variables and real estate prices.
2. **Residual Analysis**: Examining prediction errors to identify any bias or patterns.
3. **Feature Impact**: Studying the influence of each feature (e.g., proximity to MRT stations, age of house) on the predicted price.
4. **Further Improvements**: Suggestions for enhancing prediction accuracy, such as testing additional algorithms, tuning hyperparameters, or expanding the dataset.

## Technologies Used
- **Python**: Programming Language
- **Pandas & Numpy**: Data Manipulation
- **Scikit-learn**: Machine Learning Framework
- **Matplotlib & Seaborn**: Data Visualization
- **Pickle**: Model Serialization
- **Flask**: Web Application Framework (Optional for Deployment)
- **Heroku**: Deployment Platform (Optional)

## Credits
- **Real Estate Dataset**: The dataset used for training and evaluation.
- **Scikit-learn**: Framework for building and evaluating the machine learning model.
- **Pandas, Numpy, Matplotlib & Seaborn**: Libraries used for data analysis and visualization.
