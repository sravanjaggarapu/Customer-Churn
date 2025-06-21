# Customer Churn Prediction 📉

This is a machine learning project that predicts whether a customer is likely to churn (i.e., leave the service) based on demographic and usage data. Customer churn prediction is a critical task for businesses, especially in telecom, SaaS, and subscription services.

## 🔍 Problem Statement

Given customer data such as tenure, services availed, and payment information, the objective is to build a classification model that predicts if a customer will churn or stay.

## 📁 Dataset

The dataset includes features like:

- `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- `tenure`, `MonthlyCharges`, `TotalCharges`
- `Contract`, `PaymentMethod`, `InternetService`, etc.
- Target variable: `Churn`

> **Note:** If the dataset is not publicly available, please remove it or include only a sample with a link to the original source.

## 📊 Exploratory Data Analysis (EDA)

- Class imbalance in churned vs non-churned customers
- Correlation of tenure and charges with churn
- Impact of contract type and payment method on churn

## 🛠️ Machine Learning Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- Model evaluation using accuracy, confusion matrix, and classification report

## ✅ Results

- Best-performing model: [Mention here]
- Accuracy: ~[Insert final accuracy]
- Key insight: [e.g., Customers with month-to-month contracts have higher churn rates]

## 📦 Installation

To run this notebook:

```bash
git clone https://github.com/your-username/customer-churn.git
cd customer-churn
pip install -r requirements.txt
jupyter notebook
