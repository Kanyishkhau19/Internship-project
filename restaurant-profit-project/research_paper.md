# Predictive Modeling and Profit Optimization for Multi-Channel Restaurant Operations

*A student mini-project report*

---

## Abstract

Restaurants increasingly sell through multiple channels - dining in-store and delivering via
aggregators such as Uber Eats and DoorDash, as well as their own self-delivery service. Each
channel carries a different cost structure, making it hard to judge, at a glance, how a change in
commission rate or channel mix affects overall profit. This project uses the SkyCity Auckland
Restaurants & Bars dataset (1,696 monthly restaurant snapshots) to build a machine learning model
that predicts Total Monthly Net Profit from operating characteristics such as order volume,
average order value, channel shares, commission rate, and cost rates. Three regression models -
Linear Regression, Random Forest, and Gradient Boosting - were trained and compared. Gradient
Boosting performed best, achieving an R² of 0.98 and a Mean Absolute Error of about $600 on
held-out test data. The trained model was wrapped in an interactive Streamlit dashboard offering
profit prediction, What-If scenario analysis, sensitivity analysis, and a simple grid-search-based
channel-mix optimizer.

## 1. Introduction

Multi-channel restaurant operations are common today, but understanding how each channel
contributes to overall profitability is not always obvious, since commission rates, delivery
costs, and cost-of-goods rates interact in non-trivial ways. This project explores whether a
simple, interpretable machine learning approach can predict a restaurant's total monthly net
profit accurately enough to support basic "what-if" business decisions.

## 2. Problem Statement

Given a restaurant's monthly operating data (order counts, average order value, channel-level
revenue, cost rates, commission rate, and delivery cost), predict its Total Monthly Net Profit,
and provide tools for exploring how changes to individual factors affect that prediction.

## 3. Objectives

- Clean and explore the SkyCity Auckland Restaurants & Bars dataset.
- Engineer explainable, business-meaningful features.
- Train and compare Linear Regression, Random Forest, and Gradient Boosting models.
- Identify and eliminate data-leakage columns before modeling.
- Build a What-If Analysis and simple channel-mix Optimization tool on top of the trained model.
- Present findings through an interactive Streamlit dashboard.

## 4. Dataset Description

The dataset contains 1,696 rows and 30 columns, each row representing one restaurant's monthly
performance snapshot across 8 cuisine types, 4 operating segments (Cafe, QSR, Ghost Kitchen,
Full-service), and 4 Auckland subregions (CBD, North Shore, South Auckland, West Auckland). It
includes order counts, revenue, and net profit broken down by channel (In-Store, Uber Eats,
DoorDash, Self-Delivery), along with cost drivers such as COGS rate, OPEX rate, commission rate,
and delivery cost per order. The dataset had no missing values, no duplicate rows, and no
duplicate restaurant IDs.

## 5. Methodology

The workflow followed was: data cleaning, exploratory data analysis, feature engineering,
data-leakage screening, model training and comparison, model evaluation, and finally building
What-If and optimization tools on top of the selected model. All feature-engineering logic is
centralized in a single shared module so that the training pipeline and the interactive dashboard
never drift out of sync.

## 6. Exploratory Data Analysis

Twelve charts were produced covering the distribution of orders, average order value, revenue,
and profit; profit broken down by cuisine, segment, and subregion; revenue and profit by channel;
two scatter plots examining commission rate against Uber Eats profit and delivery cost against
self-delivery profit; and a correlation heatmap across all numeric features. The scatter analyses
showed a negative correlation between commission rate and Uber Eats net profit, and a negative
correlation between delivery cost per order and self-delivery net profit - both consistent with
business intuition, though correlational rather than causal.

## 7. Feature Engineering

The target variable, Total Monthly Net Profit, was calculated as the sum of the four
channel-level net profit columns already present in the dataset. Additional engineered features
included Total Revenue, Revenue per Order, Commission Cost, per-channel revenue shares, two simple
interaction features capturing commission and delivery-cost impact, and a growth-adjusted order
count. Two further derived metrics, Profit per Order and Profit Margin, were computed for display
purposes only and deliberately excluded from the model inputs (see Section 8).

## 8. Data Leakage Prevention

Because the target variable is a direct sum of four channel-level net-profit columns, those four
columns were excluded from the model's input features - including them would let the model simply
reconstruct the answer rather than learn genuine predictive patterns. The two profit-derived
metrics (Profit per Order, Profit Margin) were excluded for the same reason, since both are
calculated directly from the target. Restaurant ID and Restaurant Name were also excluded as they
carry no predictive business meaning.

## 9. Machine Learning Models

Three regression models were trained using an 80/20 train-test split with a fixed random seed
(random_state = 42): Linear Regression as a baseline, Random Forest Regressor as the primary
model, and Gradient Boosting Regressor as a comparison model. Categorical features (Cuisine Type,
Segment, Subregion) were encoded using One-Hot Encoding inside a scikit-learn Pipeline, keeping
preprocessing and modeling reproducible and leakage-safe.

## 10. Model Evaluation

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Gradient Boosting | 599.62 | 787.76 | 0.9807 |
| Random Forest | 794.75 | 1104.89 | 0.9620 |
| Linear Regression | 1001.65 | 1476.29 | 0.9322 |

All three models performed reasonably well, but Gradient Boosting achieved the lowest error and
highest R² on the held-out test set, and was therefore selected as the final model.

## 11. Feature Importance

Feature importance analysis of the Gradient Boosting model showed that OPEX Rate and COGS Rate
were the most influential features, followed by Self-Delivery Revenue and Total Revenue. This
matches business logic: cost rates directly determine what fraction of revenue becomes profit,
independent of channel mix. This is an observation about what the model relies on for its
predictions, not a claim of causation.

## 12. What-If Analysis

A What-If Analysis tool was built on top of the trained model, allowing a user to start from a
restaurant's current values and adjust commission rate, delivery cost, delivery radius, growth
factor, and channel shares. The tool reports the current predicted profit, the scenario predicted
profit, the absolute difference, and the percentage change - all computed live from the trained
model rather than hard-coded.

## 13. Optimization

A simple grid-search optimizer was implemented: for a small, user-selected set of candidate
percentages for In-Store, Uber Eats, and DoorDash share (with Self-Delivery share set to whatever
percentage remains so the four shares sum to 100%), the trained model predicts profit for every
valid combination, and the highest-predicted combination is recommended alongside an Optimization
Uplift percentage relative to the current mix.

## 14. Results

The final Gradient Boosting model explained approximately 98% of the variance in Total Monthly Net
Profit on unseen test data, with an average absolute prediction error of about $600. EDA and
feature importance results were consistent with straightforward business reasoning about cost
rates and channel economics, giving confidence that the model is learning genuine patterns rather
than noise.

## 15. Insights

- Cost structure (COGS Rate, OPEX Rate) has the strongest influence on predicted profit among the
  features examined.
- Commission rate and delivery cost per order both show a negative relationship with the
  respective channel's net profit in this dataset.
- Profit varies by cuisine type and operating segment, but this variation should be treated as an
  observed pattern in the data rather than a proven causal effect.

## 16. Limitations

The dataset represents a single monthly snapshot per restaurant rather than a time series, so
seasonal or long-term trends cannot be captured. All EDA relationships are correlational, not
causal. The optimizer searches only a small, user-defined grid of channel-mix percentages rather
than a continuous or exhaustive search, and does not account for real-world operational
constraints such as kitchen capacity or staffing limits.

## 17. Future Scope

Future extensions could include collecting multi-month time-series data per restaurant to model
trends and seasonality, incorporating additional cost drivers such as staffing and rent, applying
more advanced optimization techniques for a more thorough channel-mix search, and validating the
model on restaurant data from regions beyond Auckland.

## 18. Conclusion

This project demonstrates that a simple, carefully leakage-checked machine learning pipeline can
predict multi-channel restaurant profit with high accuracy on this dataset, and that the resulting
model can power practical, interpretable decision-support tools - a What-If scenario calculator
and a basic channel-mix optimizer - without requiring complex or opaque techniques.
