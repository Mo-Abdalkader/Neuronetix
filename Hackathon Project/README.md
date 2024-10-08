# Telecom Customer Churn Prediction

## Overview

This project focuses on predicting customer churn in the telecom industry using machine learning techniques. Churn refers to customers who leave a company's services, which is a critical problem for many telecom providers. By accurately predicting churn, companies can implement strategies to retain customers and minimize revenue losses. This notebook covers data preprocessing, exploratory data analysis (EDA), feature engineering, and the implementation of several classification algorithms to predict churn.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Libraries Used](#libraries-used)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Results](#results)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used for this project is derived from a telecom company's customer data. It includes various attributes of customers such as demographics, account information, and service details. The goal is to predict the likelihood of customer churn based on these attributes.

The target variable is `Churn`, which indicates whether the customer has churned (`Yes`) or not (`No`).

## Features

The dataset contains the following features:

| Feature               | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `customerID`           | Unique identifier for each customer                                          |
| `gender`               | Gender of the customer (Male, Female)                                        |
| `SeniorCitizen`        | Whether the customer is a senior citizen (1 = Yes, 0 = No)                   |
| `Partner`              | Whether the customer has a partner (Yes, No)                                 |
| `Dependents`           | Whether the customer has dependents (Yes, No)                                |
| `tenure`               | Number of months the customer has stayed with the company                    |
| `PhoneService`         | Whether the customer has phone service (Yes, No)                             |
| `MultipleLines`        | Whether the customer has multiple phone lines (Yes, No, No phone service)    |
| `InternetService`      | Customer’s internet service provider (DSL, Fiber optic, No)                  |
| `OnlineSecurity`       | Whether the customer has online security (Yes, No)                           |
| `OnlineBackup`         | Whether the customer has online backup (Yes, No)                             |
| `DeviceProtection`     | Whether the customer has device protection (Yes, No)                         |
| `TechSupport`          | Whether the customer has tech support (Yes, No)                              |
| `StreamingTV`          | Whether the customer streams TV (Yes, No)                                    |
| `StreamingMovies`      | Whether the customer streams movies (Yes, No)                                |
| `Contract`             | The type of contract the customer has (Month-to-month, One year, Two year)   |
| `PaperlessBilling`     | Whether the customer uses paperless billing (Yes, No)                        |
| `PaymentMethod`        | The payment method used by the customer (Electronic check, Mailed check, etc.) |
| `MonthlyCharges`       | The monthly charges billed to the customer                                   |
| `TotalCharges`         | The total charges incurred by the customer over their tenure                 |
| `Churn`                | Whether the customer has churned (`Yes`, `No`)                               |

## Libraries Used

The following Python libraries were utilized for data analysis, visualization, preprocessing, and machine learning:

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **matplotlib & seaborn**: For data visualization.
- **plotly**: For interactive visualizations.
- **scikit-learn**: For machine learning models, preprocessing, and metrics.
- **xgboost**: For the implementation of gradient boosting models.
- **warnings**: To suppress warnings during model training.

## Data Preprocessing

To ensure the dataset is clean and ready for machine learning models, the following preprocessing steps were carried out:

1. **Handling Missing Values**:
   - Missing values in features such as `TotalCharges` were imputed using techniques like `SimpleImputer`, `KNNImputer`, and `IterativeImputer`.

2. **Encoding Categorical Variables**:
   - Categorical features such as `gender`, `InternetService`, and `Contract` were encoded using `LabelEncoder` and `OneHotEncoder` for machine learning compatibility.

3. **Scaling Numerical Features**:
   - Features like `tenure`, `MonthlyCharges`, and `TotalCharges` were standardized using `StandardScaler` and `MinMaxScaler` to ensure they are on a similar scale for model training.

4. **Train-Test Split**:
   - The dataset was split into training and test sets using `train_test_split` from scikit-learn.

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) was performed to gain insights into the data and uncover relationships between features and the target variable `Churn`. Some key observations from EDA include:

- **Distribution Analysis**:
  - Histograms, box plots, and count plots were used to visualize the distribution of numerical and categorical features.
  
- **Correlation Analysis**:
  - A heatmap was used to understand the correlations between different features and their impact on customer churn.
  
- **Churn Proportion**:
  - The proportion of churned customers was analyzed to identify class imbalances in the dataset.

## Modeling

Various machine learning models were trained and evaluated to predict customer churn. These include:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machines (SVM)**
- **Decision Trees**
- **Random Forest**
- **AdaBoost**
- **Gradient Boosting (XGBoost)**
- **Naive Bayes**

Each model was tuned using techniques such as grid search and cross-validation, and the performance was evaluated based on metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

## Results

The final results from model training and testing indicated that certain models, such as **Random Forest** and **XGBoost**, performed better at predicting customer churn. Detailed classification reports, confusion matrices, and ROC curves were generated to assess model performance.

Key evaluation metrics include:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**
