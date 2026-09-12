# Customer Churn Prediction

## Week 1: Exploratory Data Analysis

### Dataset
- Source: Telco Customer Churn (Kaggle)
- Size: 7,043 customers, 21 features
- Target: Predict customer churn (Yes/No)

### Key Findings
- Overall churn rate is 26.54% (1,869 of 7,043 customers).
- Month-to-month contracts churn at ~42%, compared to ~11% for one-year and ~3% for two-year contracts — the strongest churn driver found.
- Customers with shorter tenure are far more likely to churn; most churn happens early in the customer relationship.
- Fiber optic internet subscribers churn at ~42%, notably higher than DSL (~19%) or customers with no internet service (~7%).
- Electronic check users have the highest churn rate (~45%), nearly three times that of customers on automatic payment methods like bank transfer or credit card.

### Setup
Open the Kaggle notebook or run locally:
```
pip install pandas numpy matplotlib seaborn
```
