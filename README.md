# Telco Customer Analysis
## Project Overview

1. [Insight with Power BI](#1-insight)
2. [Preprocessing](#2-preprocessing-data)
3. [Model Performance](#3-model-performance-and-summary)

## Motivation
Gain insights about customer by visualization customers profile, contracts, and services using power BI. And then perform an SVM to predict the churn based on customer's data.

The features are:

- `customerID` : Unique identifier for each customer
- `gender` : Indicates the customer's gender (male or female)
- `SeniorCitizen` : Whether the customer is a senior citizen (1 for Yes, 0 for No)
- `Partner` : Indicates if the customer has a partner (Yes or No)
- `Dependents` : Specifies if the customer has dependents (Yes or No)
- `tenure` : Number of months the customer has been with the company
- `PhoneService` : Indicates if the customer has a phone service (Yes or No)
- `MultipleLines` : Whether the customer has multiple lines (Yes, No, or No phone service)
- `InternetService` : Type of internet service (DSL, Fiber optic, or No)
- `OnlineSecurity` : If the customer has online security (Yes, No, or No internet service)
- `OnlineBackup` : Whether the customer has online backup (Yes, No, or No internet service)
- `DeviceProtection` : If the customer has device protection (Yes, No, or No internet service)
- `TechSupport` : Whether the customer has tech support (Yes, No, or No internet service)
- `StreamingTV` : If the customer has streaming TV (Yes, No, or No internet service)
- `StreamingMovies` : Whether the customer has streaming movies (Yes, No, or No internet service)
- `Contract` : The customer's contract term (Month-to-month, One year, Two year)
- `PaperlessBilling` : If the customer has paperless billing (Yes or No)
- `PaymentMethod` : Customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
- `MonthlyCharges` : The amount charged to the customer monthly
- `TotalCharges` : The total amount charged to the customer
- `Churn` : Whether the customer churned or not (Yes or No)

**Dataset:** Telco Customer Data

**Link:** [https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset/data](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## How to run

1. **Download the Repository**: Clone or download the repository from GitHub to your local machine.

2. **Install Requirements**: Run `pip install -r requirements.txt` to install the required Python packages listed in `requirements.txt`.

3. **Run the Main Script**: Execute the main script of the project by running `python main.py`.

## Files

1. **Images/**: Directory for image outputs.
2. **.gitattributes**: Git configuration file for repository settings.
3. **README.md**: Documentation file for the project overview and instructions.
4. **Telco-Customer-Data.csv**: Dataset file containing telco customer data.
5. **data_exploration.py**: Python script for initial data analysis.
6. **insights.pbix**: Power BI report file for data visualization.
7. **insights.pdf**: PDF export of the Power BI report.
8. **main.py**: Python script implementing SVM for churn prediction.
9. **requirements.txt**: List of Python dependencies for the project.

## 1. Insight

### A. Customer Profile
![1](https://github.com/eliasright/Telco-Customer-Analysis/assets/151723828/e7745baa-88b8-46a7-afcf-0fe7eca24aff)

### B. Customer Contracts
![2](https://github.com/eliasright/Telco-Customer-Analysis/assets/151723828/940da848-b837-437d-9c51-ef05da14937a)

### C. Customer Services
![3](https://github.com/eliasright/Telco-Customer-Analysis/assets/151723828/5d52e97f-b7ac-4359-b9ee-2ab3f0518b7c)

## 2. Preprocessing Data

1. No clear outliers for continuous values using the boxplot.
![No Outlier](https://github.com/eliasright/Telco-Customer-Analysis/assets/151723828/c232e505-f968-40a9-b09a-7fb2857b928a)

2. Pearson's Correlation shows a mean correlation between the continuous features with the target feature `churn` but shows a strong correlation between each other. Hence all were kept for the model.
![Pearson](https://github.com/eliasright/Telco-Customer-Analysis/assets/151723828/6e48373b-335e-494e-817e-c10bb3fc6015)

3. Using a Chi-squared test shows that PhoneService and Gender were not significant at 5% confidence and hence will not be used in the model.
![Chi-squared](https://github.com/eliasright/Telco-Customer-Analysis/assets/151723828/8263989e-bbc0-42d2-938a-fc24ee2ae81a)

## 3. Model Performance and Summary

I used a Support Vector Machine (SVM) with an RBF kernel, optimizing the C parameter to balance error minimization and model complexity, and the gamma parameter to control the influence range of training examples. I applied Bayesian Optimization for hyperparameter tuning, using cross-validation and the F1 score to evaluate performance, alongside data preprocessing and SMOTE for class imbalance in the target variable `churn`.

The resulting output is
```
              precision    recall  f1-score   support

           0       0.89      0.71      0.79      1022
           1       0.50      0.77      0.60       387

    accuracy                           0.72      1409
   macro avg       0.69      0.74      0.70      1409
weighted avg       0.78      0.72      0.74      1409
```

The classification report indicates that the model is more accurate in identifying customers who remain with the company (class 0), with a high precision of 0.89. However, its ability to correctly predict customers who churn (class 1) is lower, with a precision of just 0.50. This suggests that while the model is reliable in recognizing customers who stay, it struggles to accurately identify churners, as half of the predicted churn cases are false positives.

In terms of recall, the model performs better in detecting actual churn cases, with a recall of 0.77 for class 1, compared to 0.71 for customers who stay. This means it's more likely to catch customers at risk of churning, but at the cost of misclassifying some who stay.

The overall accuracy of the model is 72%, which is reasonable but indicates room for improvement. The F1-scores, which balance precision and recall, further highlight the model's relative strengths and weaknesses in predicting each class.

Given these results, further fine-tuning of the model might be beneficial. This could involve adjusting hyperparameters, collecting more data to address potential biases or imbalances, or exploring different feature sets. Additionally, experimenting with other modeling approaches could provide insights into better strategies for predicting customer churn.

