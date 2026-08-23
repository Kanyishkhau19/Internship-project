"""
train_model.py
---------------
This script does the "offline" part of the project:

1. Load & clean the SkyCity Auckland dataset
2. Run basic EDA and save charts to eda_charts/
3. Engineer features (see ml_utils.py)
4. Train 3 models: Linear Regression, Random Forest, Gradient Boosting
5. Evaluate them with MAE, RMSE, R^2
6. Save a model comparison table (model_comparison.csv)
7. Save a feature-importance chart for the best tree-based model
8. Save the best model + preprocessing pipeline to model.pkl
   (this is the file app.py loads to make live predictions)

Run this once before launching the Streamlit app:
    python train_model.py
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so this works without a display (server-safe)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from ml_utils import (
    get_model_ready_data,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    ALL_FEATURES,
    EXCLUDED_COLUMNS_EXPLANATION,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RANDOM_STATE = 42

CHART_DIR = "eda_charts"
os.makedirs(CHART_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"Saved chart: {path}")


def run_eda(df: pd.DataFrame):
    """Create the EDA charts requested in the project brief and print
    a few plain-language observations based on the actual data."""

    print("\n===== BASIC DATASET INFO =====")
    print("Rows:", df.shape[0], "| Columns:", df.shape[1])
    print("\nMissing values per column (only columns with >0 shown):")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "None")
    print("\nDuplicate rows:", df.duplicated().sum())

    # 1. Monthly Orders Distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["MonthlyOrders"], bins=30, kde=True, ax=ax, color="#4C72B0")
    ax.set_title("Monthly Orders Distribution")
    save_fig(fig, "01_monthly_orders_distribution.png")

    # 2. Average Order Value Distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["AOV"], bins=30, kde=True, ax=ax, color="#DD8452")
    ax.set_title("Average Order Value (AOV) Distribution")
    save_fig(fig, "02_aov_distribution.png")

    # 3. Total Revenue Distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["TotalRevenue"], bins=30, kde=True, ax=ax, color="#55A868")
    ax.set_title("Total Monthly Revenue Distribution")
    save_fig(fig, "03_total_revenue_distribution.png")

    # 4. Net Profit Distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["TotalNetProfit"], bins=30, kde=True, ax=ax, color="#C44E52")
    ax.set_title("Total Monthly Net Profit Distribution")
    save_fig(fig, "04_net_profit_distribution.png")

    # 5. Profit by Cuisine Type
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = df.groupby("CuisineType")["TotalNetProfit"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="CuisineType", y="TotalNetProfit", order=order, ax=ax, palette="Set2")
    ax.set_title("Net Profit by Cuisine Type")
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, "05_profit_by_cuisine.png")

    # 6. Profit by Segment
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=df, x="Segment", y="TotalNetProfit", ax=ax, palette="Set3")
    ax.set_title("Net Profit by Segment")
    save_fig(fig, "06_profit_by_segment.png")

    # 7. Profit by Subregion
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=df, x="Subregion", y="TotalNetProfit", ax=ax, palette="Set1")
    ax.set_title("Net Profit by Subregion")
    save_fig(fig, "07_profit_by_subregion.png")

    # 8. Revenue by Channel
    channel_revenue = df[["InStoreRevenue", "UberEatsRevenue", "DoorDashRevenue", "SelfDeliveryRevenue"]].sum()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    channel_revenue.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.set_title("Total Revenue by Channel (all restaurants, summed)")
    ax.set_ylabel("Revenue ($)")
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, "08_revenue_by_channel.png")

    # 9. Profit by Channel
    channel_profit = df[["InStoreNetProfit", "UberEatsNetProfit", "DoorDashNetProfit", "SelfDeliveryNetProfit"]].sum()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    channel_profit.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.set_title("Total Net Profit by Channel (all restaurants, summed)")
    ax.set_ylabel("Net Profit ($)")
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, "09_profit_by_channel.png")

    # 10. Commission Rate vs Uber Eats Profit
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(data=df, x="CommissionRate", y="UberEatsNetProfit", alpha=0.5, ax=ax, color="#DD8452")
    ax.set_title("Commission Rate vs Uber Eats Net Profit")
    save_fig(fig, "10_commission_vs_ue_profit.png")

    # 11. Delivery Cost vs Self-Delivery Profit
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(data=df, x="DeliveryCostPerOrder", y="SelfDeliveryNetProfit", alpha=0.5, ax=ax, color="#C44E52")
    ax.set_title("Delivery Cost per Order vs Self-Delivery Net Profit")
    save_fig(fig, "11_deliverycost_vs_sd_profit.png")

    # 12. Correlation Heatmap (numeric features + target)
    corr_cols = NUMERIC_FEATURES + ["TotalNetProfit"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True, linewidths=0.3, cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation Heatmap (numeric features + target)")
    save_fig(fig, "12_correlation_heatmap.png")

    # ---- Plain-language observations, based on the ACTUAL correlations ----
    print("\n===== EDA OBSERVATIONS (based on actual data) =====")
    ue_corr = df["CommissionRate"].corr(df["UberEatsNetProfit"])
    print(f"- Correlation between CommissionRate and UberEatsNetProfit: {ue_corr:.3f}")
    print("  Higher commission rates are associated with lower Uber Eats profit in this dataset."
          if ue_corr < 0 else
          "  Commission rate does not show a clear negative relationship with Uber Eats profit here.")

    sd_corr = df["DeliveryCostPerOrder"].corr(df["SelfDeliveryNetProfit"])
    print(f"- Correlation between DeliveryCostPerOrder and SelfDeliveryNetProfit: {sd_corr:.3f}")
    print("  Higher self-delivery cost per order is associated with lower self-delivery profit in this dataset."
          if sd_corr < 0 else
          "  Delivery cost per order does not show a clear negative relationship with self-delivery profit here.")

    best_cuisine = df.groupby("CuisineType")["TotalNetProfit"].median().idxmax()
    worst_cuisine = df.groupby("CuisineType")["TotalNetProfit"].median().idxmin()
    print(f"- Highest median net profit by cuisine: {best_cuisine}")
    print(f"- Lowest median net profit by cuisine: {worst_cuisine}")
    print("  (This is an observed pattern in the dataset, not a proven causal effect of cuisine type.)")


def build_preprocessor():
    """ColumnTransformer: one-hot encode categoricals, pass numeric through."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",  # numeric features pass through unchanged
    )
    return preprocessor


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    results = []
    fitted_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", model),
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        fitted_pipelines[name] = pipeline
        print(f"{name:20s}  MAE={mae:9.2f}  RMSE={rmse:9.2f}  R2={r2:.4f}")

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    return results_df, fitted_pipelines, (X_train, X_test, y_train, y_test)


def plot_feature_importance(pipeline, model_name):
    """Extract and plot feature importance for a tree-based model."""
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]

    # Get the final feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = cat_feature_names + NUMERIC_FEATURES

    importances = regressor.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", ax=ax, color="#4C72B0")
    ax.set_title(f"Top 15 Feature Importances ({model_name})")
    save_fig(fig, "13_feature_importance.png")

    return imp_df


def main():
    print("Loading, cleaning, and engineering features...")
    df, X, y = get_model_ready_data()

    print("\n===== EXCLUDED COLUMNS (data leakage prevention) =====")
    for col, reason in EXCLUDED_COLUMNS_EXPLANATION.items():
        print(f"- {col}: {reason}")

    run_eda(df)

    print("\n===== TRAINING MODELS =====")
    results_df, pipelines, split_data = train_and_evaluate(X, y)

    print("\n===== MODEL COMPARISON TABLE =====")
    print(results_df.to_string(index=False))
    results_df.to_csv("model_comparison.csv", index=False)

    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = pipelines[best_model_name]
    print(f"\nBest model selected (highest R^2 on test set): {best_model_name}")

    # Feature importance only makes sense for tree-based models
    if best_model_name in ("Random Forest", "Gradient Boosting"):
        imp_df = plot_feature_importance(best_pipeline, best_model_name)
        imp_df.to_csv("feature_importance.csv", index=False)
    else:
        # fall back to Random Forest chart for illustration purposes
        imp_df = plot_feature_importance(pipelines["Random Forest"], "Random Forest")
        imp_df.to_csv("feature_importance.csv", index=False)

    # Save the best model + metadata
    joblib.dump(best_pipeline, "model.pkl")
    print("\nSaved best model to model.pkl")

    metadata = {
        "best_model_name": best_model_name,
        "features_used": ALL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "metrics": results_df.to_dict(orient="records"),
    }
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved model_metadata.json")

    print("\nDone. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
