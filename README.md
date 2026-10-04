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

## Week 2: Building ML Models

Trained and compared 5 models to predict customer churn, moving beyond raw 
accuracy to business-driven evaluation.

**The accuracy trap:** A dummy model that always predicts "customer stays" hits 
**73.5% accuracy** — while catching **zero** actual churners. This is why every 
model below is judged on recall and AUC, not just accuracy.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Baseline (dummy) | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.807 | 0.658 | 0.567 | 0.609 | **0.842** |
| LR (balanced) | 0.739 | 0.505 | 0.781 | 0.613 | 0.841 |
| Decision Tree (d=5) | 0.796 | 0.632 | 0.551 | 0.589 | 0.829 |
| Random Forest | 0.807 | 0.673 | 0.529 | 0.593 | **0.842** |

### Top churn drivers (permutation importance)
1. **Tenure** — longer-tenured customers are far less likely to churn
2. **TotalCharges**
3. **Two-year contract** — cuts odds of churn by ~45% vs month-to-month

### Deployment decision
Chose **Logistic Regression at threshold 0.15** (not the default 0.5). Since a 
missed churner costs **6x more** than an unnecessary retention offer (PKR 6,000 
vs 1,000), the cost-optimal threshold is 0.14–0.15, not 0.5. At this threshold, 
recall jumps to ~0.92, catching 132 more churners than the default threshold — 
at the cost of 327 more false alarms, which is the right trade-off given the 
cost asymmetry. LR was preferred over the near-identical Random Forest (both 
AUC 0.842) for its interpretability via odds ratios.

### Feature engineering result
Added `n_services`, `is_new`, `charge_per_mo`, and `price_jump` — AUC moved from 
0.8422 to 0.8420, essentially no change. The Random Forest likely already 
captures these interactions through existing splits on tenure and charges.

### Biggest lesson
Accuracy alone can hide a completely useless model — the "right" evaluation 
metric and decision threshold depend on what the business actually pays for 
mistakes, not a statistical default like 0.5.

Notebook: [View Week 2 Notebook](week2-ml-models.ipynb)


## Week 3: Model Optimization and Unsupervised Learning

- Split-to-split accuracy range across 20 seeds: 0.780 to 0.828
- 5-fold CV AUC:
  - Logistic Regression: 0.8464 ± 0.0129
  - Random Forest: 0.8464 ± 0.0114
  - XGBoost: 0.8504 ± 0.0125
- Best Random Forest parameters from Random Search:
  - `max_depth = 15`
  - `max_features = 0.21`
  - `min_samples_leaf = 15`
- Grid Search vs Random Search:
  - Grid Search: 113 seconds
  - Random Search: 122 seconds
  - Both used 120 fits
- Final XGBoost test AUC: 0.8478
- Customer segments (`k = 4`):
  - Mid-tenure, high bill — 43% churn
  - New, low spend — 32% churn
  - Loyal premium — 14% churn
  - Loyal basics — 5% churn
- PCA: 15 of 30 components explain 90% of the variance

### Key Lesson

Tuning gave only a small gain (+0.006 AUC over my Week 2 Random Forest). The main lesson was that an honest cross-validated estimate mattered more than chasing one more point.

Notebook: [View Week 3 Notebook](week3-optimization.ipynb)
