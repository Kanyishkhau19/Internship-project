# Predictive Modeling and Profit Optimization for Multi-Channel Restaurant Operations

**A B.E. Computer Science Engineering (2nd Year) Machine Learning Mini-Project**

---

## 1. Introduction

Restaurants today sell food through multiple channels at once - dining In-Store, and delivering
through Uber Eats, DoorDash, and their own Self-Delivery fleet. Each channel has a different cost
structure: aggregators like Uber Eats and DoorDash charge commission, self-delivery has its own
delivery cost per order, and every channel is affected by food cost (COGS) and operating cost
(OPEX) rates.

This project builds a simple Machine Learning model that predicts a restaurant's **Total Monthly
Net Profit** from its operating characteristics, and wraps that model in an interactive Streamlit
dashboard so a manager could (hypothetically) test "what if" scenarios and get a channel-mix
recommendation.

## 2. Problem Statement

Given monthly operating data for a restaurant (orders, average order value, channel mix,
commission rate, delivery cost, cost rates, etc.), predict the restaurant's **Total Monthly Net
Profit**, and allow a user to explore how changing individual inputs (commission rate, channel
shares, delivery cost, etc.) would change predicted profit.

## 3. Objectives

1. Clean and understand the SkyCity Auckland Restaurants & Bars dataset.
2. Perform Exploratory Data Analysis (EDA) to find patterns in profit across cuisine, segment,
   subregion, and channel.
3. Engineer simple, explainable features from the raw columns.
4. Train and compare 3 regression models to predict Total Monthly Net Profit.
5. Carefully avoid data leakage (explained in Section 9).
6. Build a What-If Analysis tool so a user can test different scenarios.
7. Build a simple channel-mix optimizer using grid search.
8. Present everything in an interactive Streamlit dashboard.

## 4. Dataset Description

**File:** `data/SkyCity Auckland Restaurants & Bars.csv`
**Rows:** 1,696 (each row = one restaurant's monthly snapshot)
**Columns:** 30

Key columns:

| Column | Meaning |
|---|---|
| `CuisineType`, `Segment`, `Subregion` | Categorical descriptors of the restaurant |
| `GrowthFactor` | A multiplier reflecting expected order growth |
| `AOV` | Average Order Value ($) |
| `MonthlyOrders`, `InStoreOrders`, `UberEatsOrders`, `DoorDashOrders`, `SelfDeliveryOrders` | Order counts overall and by channel |
| `InStoreRevenue`, `UberEatsRevenue`, `DoorDashRevenue`, `SelfDeliveryRevenue` | Revenue by channel |
| `COGSRate`, `OPEXRate` | Cost of goods sold rate and operating expense rate |
| `CommissionRate` | Commission % charged by delivery aggregators |
| `DeliveryRadiusKM`, `DeliveryCostPerOrder`, `SD_DeliveryTotalCost` | Self-delivery cost drivers |
| `InStoreNetProfit`, `UberEatsNetProfit`, `DoorDashNetProfit`, `SelfDeliveryNetProfit` | Net profit by channel (used to build the target) |
| `InStoreShare`, `UE_share`, `DD_share`, `SD_share` | Planned/target channel share values already in the data |

The dataset had **no missing values, no duplicate rows, and no duplicate RestaurantIDs** when
inspected (see the notebook, Section 2). Channel-level net profit columns do contain some
negative values (a channel losing money after commission/delivery costs) - this is a realistic
business outcome and was kept, not removed.

## 5. Technologies Used

- Python 3
- Pandas, NumPy - data handling
- Matplotlib, Seaborn - visualization
- Scikit-learn - machine learning (Linear Regression, Random Forest, Gradient Boosting)
- Streamlit - interactive dashboard
- Joblib - model persistence

## 6. Methodology

```
Dataset -> Data Cleaning -> EDA -> Feature Engineering -> ML Model
        -> Prediction -> What-If Analysis -> Simple Optimization -> Streamlit Dashboard
```

All data-processing logic lives in **`ml_utils.py`**, and is imported by both `train_model.py`
(offline training) and `app.py` (the live dashboard), so the two always stay in sync.

## 7. Data Cleaning

- Removed duplicate rows / duplicate RestaurantIDs (none were found, but the check is in the code
  for safety on future data refreshes).
- Filled missing numeric values with the median and missing categorical values with the most
  frequent category (defensive - the dataset had none).
- Removed only rows with **impossible** values (negative revenue or negative order counts) - none
  were found in this dataset.
- Did **not** remove outliers automatically - a restaurant with very high orders is a real busy
  restaurant, not necessarily bad data.
- Kept negative channel-level net profit values, since a channel legitimately losing money in a
  given month is a realistic business scenario.

## 8. EDA - Key Observations

(See `eda_charts/` for all 13 charts, or re-run `train_model.py` / the notebook to regenerate them.)

- Commission Rate has a negative correlation with Uber Eats Net Profit in this dataset - higher
  commission rates tend to go with lower Uber Eats profit.
- Delivery Cost per Order has a negative correlation with Self-Delivery Net Profit - higher
  self-delivery cost per order tends to go with lower self-delivery profit.
- Profit varies by cuisine type and segment, but this is an observed pattern in the data, not
  proof of causation.

## 9. Feature Engineering & Avoiding Data Leakage

**Target variable:** `TotalNetProfit = InStoreNetProfit + UberEatsNetProfit + DoorDashNetProfit + SelfDeliveryNetProfit`

**Engineered features:** `TotalRevenue`, `RevenuePerOrder`, `CommissionCost`, `SelfDeliveryCost`,
per-channel `*RevenueShare` columns, `Commission_UE_Impact`, `DeliveryCost_SD_Impact`,
`GrowthAdjustedOrders`. (Full list and formulas in `ml_utils.py`.)

**Columns excluded from the model (data leakage prevention):**

| Column | Why excluded |
|---|---|
| `InStoreNetProfit`, `UberEatsNetProfit`, `DoorDashNetProfit`, `SelfDeliveryNetProfit` | These four columns are directly summed to create the target. Feeding them to the model would let it "cheat" by just adding them up. |
| `ProfitPerOrder` | Calculated as `TotalNetProfit / MonthlyOrders` - derived from the target, so it's leakage too. |
| `ProfitMargin` | Calculated as `TotalNetProfit / TotalRevenue` - same reason. |
| `RestaurantID`, `RestaurantName` | Just identifiers, no real predictive meaning. |

This check matters because a model that "cheats" using leaked columns will show a misleadingly
perfect score during testing, but will fail in a real scenario where those leaked values aren't
available in advance.

## 10. Machine Learning Models

Three regression models were trained on an 80/20 train-test split (`random_state=42`), using a
`ColumnTransformer` + `OneHotEncoder` for the 3 categorical columns (`CuisineType`, `Segment`,
`Subregion`) inside a scikit-learn `Pipeline`:

1. **Linear Regression** - simple baseline.
2. **Random Forest Regressor** - main model, handles non-linear relationships.
3. **Gradient Boosting Regressor** - comparison model, often strong on tabular data.

## 11. Model Evaluation (Actual Results)

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Gradient Boosting | 599.62 | 787.76 | 0.9807 |
| Random Forest | 794.75 | 1104.89 | 0.9620 |
| Linear Regression | 1001.65 | 1476.29 | 0.9322 |

*(These numbers come directly from running `train_model.py` on the provided dataset - see
`model_comparison.csv` for the exact machine-generated file.)*

**Best model: Gradient Boosting Regressor**, selected because it has the lowest MAE/RMSE and the
highest R² on the held-out test set. This makes sense because restaurant profit likely depends on
some non-linear interactions (e.g., commission rate combined with channel revenue) that boosted
trees can capture better than a plain linear model.

### Feature Importance (Gradient Boosting)

The top features (see `feature_importance.csv` and `eda_charts/13_feature_importance.png`) were
`OPEXRate`, `COGSRate`, `SelfDeliveryRevenue`, and `TotalRevenue` - all of which directly
determine how much of a restaurant's revenue becomes profit. This is an observation about what
the model relies on, not proof that these features *cause* profit changes.

## 12. What-If Analysis

The Streamlit app's **What-If Analysis** page lets a user pick a restaurant, then adjust
Commission Rate, Delivery Cost, Delivery Radius, Growth Factor, and the four channel shares.
It shows:

- Current Predicted Profit
- Scenario Predicted Profit
- Profit Difference
- Percentage Change

All numbers come from the actual trained model - nothing is hard-coded.

## 13. Simple Optimization

The **Optimization** page runs a simple grid search: the user picks a few candidate percentages
for In-Store, Uber Eats, and DoorDash share; Self-Delivery share is set to whatever remains so the
four shares always sum to 100%. The model predicts profit for every valid combination, and the
best one is shown alongside the current mix, with an **Optimization Uplift %**.

## 14. Streamlit Dashboard

Run with `streamlit run app.py`. Pages (via the sidebar):

1. **Dashboard** - dataset overview, KPIs, and summary charts.
2. **Profit Prediction** - pick a restaurant, adjust inputs, get a predicted profit.
3. **What-If Analysis** - sliders to compare Current vs Scenario profit.
4. **Sensitivity Analysis** - line charts showing how profit changes as one factor changes.
5. **Optimization** - grid-search-based channel-mix recommendation.

## 15. Results

- The Gradient Boosting model explains about 98% of the variance in Total Monthly Net Profit on
  unseen test data (R² = 0.98), with an average prediction error of about $600 (MAE).
- EDA showed clear (correlational) relationships between commission rate / delivery cost and
  channel profit, matching business intuition.
- The What-If and Optimization tools give a simple way to explore "what changes would help
  profit" without needing to retrain or hand-calculate anything.

## 16. Limitations

- The dataset is a single monthly snapshot per restaurant, not a time series - the model cannot
  capture seasonal trends or long-term growth patterns.
- Correlational EDA observations are **not** proof of causation.
- The optimizer only searches a small, user-chosen grid of channel-mix percentages - it does not
  guarantee a true global optimum, and does not know about real-world constraints (e.g. kitchen
  capacity, staffing) that might make some mixes impractical.
- The model was trained only on Auckland restaurant data, so it may not generalize to restaurants
  in other cities or countries.

## 17. Future Scope

- Collect multiple months per restaurant to build a time-series forecasting model.
- Add more granular cost data (e.g., staffing costs, rent) for a more complete profit model.
- Extend the optimizer to a continuous search (e.g., using `scipy.optimize`) instead of a
  fixed grid, once more advanced optimization techniques are covered in later coursework.
- Deploy the dashboard for multiple restaurant chains, not just SkyCity Auckland.

## 18. How to Run

### Requirements

Python 3.9+ and the packages in `requirements.txt`.

### Setup

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates model.pkl, EDA charts, and metrics)
python train_model.py

# 4. Launch the Streamlit dashboard
streamlit run app.py
```

The dashboard will open in your browser (usually at `http://localhost:8501`).

### Project Structure

```
restaurant-profit-project/
│
├── data/
│   └── SkyCity Auckland Restaurants & Bars.csv
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb
│
├── eda_charts/                  (generated by train_model.py)
├── ml_utils.py                  (shared cleaning + feature engineering)
├── train_model.py               (EDA + training + evaluation script)
├── app.py                       (Streamlit dashboard)
├── model.pkl                    (generated - trained model)
├── model_metadata.json          (generated - best model info)
├── model_comparison.csv         (generated - metrics table)
├── feature_importance.csv       (generated - top features)
├── requirements.txt
└── README.md
```
