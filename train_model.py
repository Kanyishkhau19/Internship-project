import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

file_path = "SkyCity Auckland Restaurants & Bars.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# --------------------------------------------------
# 2. BASIC CLEANING
# --------------------------------------------------

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing values
df = df.dropna()

print("After cleaning:", df.shape)


# --------------------------------------------------
# 3. CREATE TARGET VARIABLE
# --------------------------------------------------

df["TotalNetProfit"] = (
    df["InStoreNetProfit"]
    + df["UberEatsNetProfit"]
    + df["DoorDashNetProfit"]
    + df["SelfDeliveryNetProfit"]
)


# --------------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------------

# Profit per order
df["ProfitPerOrder"] = (
    df["TotalNetProfit"] / df["MonthlyOrders"]
)

# Total revenue
df["TotalRevenue"] = (
    df["InStoreRevenue"]
    + df["UberEatsRevenue"]
    + df["DoorDashRevenue"]
    + df["SelfDeliveryRevenue"]
)

# Profit margin
df["ProfitMargin"] = (
    df["TotalNetProfit"] / df["TotalRevenue"]
)

# Commission impact
df["CommissionImpact"] = (
    df["CommissionRate"] * df["UE_share"]
)

# Self-delivery cost impact
df["DeliveryCostImpact"] = (
    df["DeliveryCostPerOrder"] * df["SD_share"]
)

# Growth adjusted orders
df["GrowthAdjustedOrders"] = (
    df["MonthlyOrders"] * df["GrowthFactor"]
)


# --------------------------------------------------
# 5. SELECT FEATURES
# --------------------------------------------------

features = [
    "CuisineType",
    "Segment",
    "Subregion",

    "GrowthFactor",
    "AOV",
    "MonthlyOrders",

    "COGSRate",
    "OPEXRate",
    "CommissionRate",
    "DeliveryRadiusKM",
    "DeliveryCostPerOrder",

    "InStoreShare",
    "UE_share",
    "DD_share",
    "SD_share",

    "CommissionImpact",
    "DeliveryCostImpact",
    "GrowthAdjustedOrders"
]

X = df[features]
y = df["TotalNetProfit"]


# --------------------------------------------------
# 6. IDENTIFY COLUMN TYPES
# --------------------------------------------------

categorical_features = [
    "CuisineType",
    "Segment",
    "Subregion"
]

numerical_features = [
    "GrowthFactor",
    "AOV",
    "MonthlyOrders",
    "COGSRate",
    "OPEXRate",
    "CommissionRate",
    "DeliveryRadiusKM",
    "DeliveryCostPerOrder",
    "InStoreShare",
    "UE_share",
    "DD_share",
    "SD_share",
    "CommissionImpact",
    "DeliveryCostImpact",
    "GrowthAdjustedOrders"
]


# --------------------------------------------------
# 7. PREPROCESSING
# --------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# --------------------------------------------------
# 8. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 9. LINEAR REGRESSION
# --------------------------------------------------

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

linear_model.fit(X_train, y_train)

linear_prediction = linear_model.predict(X_test)

linear_rmse = np.sqrt(
    mean_squared_error(y_test, linear_prediction)
)

linear_mae = mean_absolute_error(
    y_test,
    linear_prediction
)

linear_r2 = r2_score(
    y_test,
    linear_prediction
)


# --------------------------------------------------
# 10. RANDOM FOREST
# --------------------------------------------------

random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

random_forest.fit(X_train, y_train)

rf_prediction = random_forest.predict(X_test)

rf_rmse = np.sqrt(
    mean_squared_error(y_test, rf_prediction)
)

rf_mae = mean_absolute_error(
    y_test,
    rf_prediction
)

rf_r2 = r2_score(
    y_test,
    rf_prediction
)


# --------------------------------------------------
# 11. DISPLAY RESULTS
# --------------------------------------------------

print("\n--------------------------------")
print("MODEL RESULTS")
print("--------------------------------")

print("\nLinear Regression")
print("RMSE:", round(linear_rmse, 2))
print("MAE :", round(linear_mae, 2))
print("R2  :", round(linear_r2, 4))

print("\nRandom Forest")
print("RMSE:", round(rf_rmse, 2))
print("MAE :", round(rf_mae, 2))
print("R2  :", round(rf_r2, 4))


# --------------------------------------------------
# 12. SELECT BEST MODEL
# --------------------------------------------------

if rf_rmse < linear_rmse:
    best_model = random_forest
    best_model_name = "Random Forest"
    best_rmse = rf_rmse
    best_mae = rf_mae
    best_r2 = rf_r2
else:
    best_model = linear_model
    best_model_name = "Linear Regression"
    best_rmse = linear_rmse
    best_mae = linear_mae
    best_r2 = linear_r2


print("\n--------------------------------")
print("BEST MODEL")
print("--------------------------------")

print("Model:", best_model_name)
print("RMSE:", round(best_rmse, 2))
print("MAE :", round(best_mae, 2))
print("R2  :", round(best_r2, 4))


# --------------------------------------------------
# 13. SAVE MODEL
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/profit_model.pkl"
)

joblib.dump(
    {
        "model_name": best_model_name,
        "rmse": best_rmse,
        "mae": best_mae,
        "r2": best_r2
    },
    "models/model_info.pkl"
)


# --------------------------------------------------
# 14. SAVE CLEAN DATA
# --------------------------------------------------

df.to_csv(
    "cleaned_data.csv",
    index=False
)

print("\nModel saved successfully!")
print("File: models/profit_model.pkl")
print("Cleaned data saved as cleaned_data.csv")