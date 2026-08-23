"""
ml_utils.py
-----------
Shared data-loading, cleaning, and feature-engineering functions.

Both the training script (train_model.py) and the Streamlit app (app.py)
import from this file, so the exact same feature engineering logic is
used everywhere. This avoids "training/serving skew" (a very common
real-world ML bug where the app computes features slightly differently
than the training script did).

Keeping this logic in one place also makes the project easier to explain
in a viva -- there is exactly ONE place where "how do we turn raw CSV
columns into model inputs" is defined.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "SkyCity Auckland Restaurants & Bars.csv"

# Columns that are IDENTIFIERS ONLY. They do not describe restaurant
# performance, so they should never be fed to the model as predictive
# features (a Random Forest could otherwise "memorize" RestaurantID and
# create fake, meaningless patterns).
ID_COLUMNS = ["RestaurantID", "RestaurantName"]

# Columns that are used to CALCULATE the target (TotalNetProfit) directly.
# These are the classic "data leakage" columns for this project:
# TotalNetProfit = InStoreNetProfit + UberEatsNetProfit + DoorDashNetProfit
#                  + SelfDeliveryNetProfit
# If we gave the model these columns as inputs, it could just add them up
# and get a (nearly) perfect score without learning anything real about
# how orders/costs/commission drive profit. So they must be excluded.
LEAKAGE_COLUMNS = [
    "InStoreNetProfit",
    "UberEatsNetProfit",
    "DoorDashNetProfit",
    "SelfDeliveryNetProfit",
]

# These two engineered columns are also DERIVED FROM the target itself
# (ProfitPerOrder = TotalNetProfit / MonthlyOrders, ProfitMargin =
# TotalNetProfit / TotalRevenue). They are extremely useful to DISPLAY
# to the user, but they must NOT be used as model inputs -- that would
# be leakage too (the model would basically be told the answer in a
# disguised form).
TARGET_DERIVED_COLUMNS = ["ProfitPerOrder", "ProfitMargin", "TotalNetProfit"]

CATEGORICAL_FEATURES = ["CuisineType", "Segment", "Subregion"]

NUMERIC_FEATURES = [
    "GrowthFactor",
    "AOV",
    "MonthlyOrders",
    "InStoreOrders",
    "UberEatsOrders",
    "DoorDashOrders",
    "SelfDeliveryOrders",
    "InStoreRevenue",
    "UberEatsRevenue",
    "DoorDashRevenue",
    "SelfDeliveryRevenue",
    "COGSRate",
    "OPEXRate",
    "CommissionRate",
    "DeliveryRadiusKM",
    "DeliveryCostPerOrder",
    "SD_DeliveryTotalCost",
    "InStoreShare",
    "UE_share",
    "DD_share",
    "SD_share",
    "TotalRevenue",
    "RevenuePerOrder",
    "CommissionCost",
    "InStoreRevenueShare",
    "UberEatsRevenueShare",
    "DoorDashRevenueShare",
    "SelfDeliveryRevenueShare",
    "Commission_UE_Impact",
    "DeliveryCost_SD_Impact",
    "GrowthAdjustedOrders",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "TotalNetProfit"


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV exactly as provided."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple, explainable cleaning steps.

    The dataset turned out to be quite clean already (checked during
    initial inspection: 0 missing values, 0 duplicate rows, 0 duplicate
    RestaurantIDs). We still keep these checks in the code so that the
    pipeline is safe to re-run if the CSV is ever updated with messier
    data.
    """
    df = df.copy()

    # 1. Remove exact duplicate rows, if any.
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"Removed {before - after} duplicate rows.")

    # 2. Remove duplicate RestaurantIDs, if any (keep the first occurrence).
    #    Each row is meant to represent one restaurant's monthly snapshot,
    #    so a repeated RestaurantID would be a data entry problem.
    before = len(df)
    df = df.drop_duplicates(subset="RestaurantID")
    after = len(df)
    if before != after:
        print(f"Removed {before - after} rows with duplicate RestaurantID.")

    # 3. Handle missing values (if any appear in future data refreshes).
    #    Numeric columns -> fill with median (robust to outliers).
    #    Categorical columns -> fill with the most frequent category.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 4. Basic sanity checks on ranges (we do NOT blindly delete outliers --
    #    a restaurant with unusually high orders is a real busy restaurant,
    #    not necessarily a data error). We only guard against impossible
    #    values such as negative order counts or negative revenue, which
    #    would indicate a data entry error rather than a real business
    #    scenario.
    impossible_mask = (
        (df["MonthlyOrders"] < 0)
        | (df["AOV"] < 0)
        | (df["InStoreRevenue"] < 0)
        | (df["UberEatsRevenue"] < 0)
        | (df["DoorDashRevenue"] < 0)
        | (df["SelfDeliveryRevenue"] < 0)
    )
    if impossible_mask.sum() > 0:
        print(f"Removing {impossible_mask.sum()} rows with impossible negative values.")
        df = df[~impossible_mask]

    # Note: channel-level NET PROFIT columns (e.g. UberEatsNetProfit) DO
    # legitimately go negative in this dataset (a channel can lose money
    # on a given month once commission and delivery costs are subtracted
    # from revenue). That is a realistic business outcome, not a data
    # error, so we deliberately do NOT remove or "fix" those rows.

    df = df.reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the target variable and all engineered features described in
    the project brief. Every feature here has a simple business meaning
    that a 2nd-year student can explain in one sentence.
    """
    df = df.copy()

    # ---- Target variable -------------------------------------------------
    # Total Monthly Net Profit = sum of the 4 channel-level net profits.
    df["TotalNetProfit"] = (
        df["InStoreNetProfit"]
        + df["UberEatsNetProfit"]
        + df["DoorDashNetProfit"]
        + df["SelfDeliveryNetProfit"]
    )

    # ---- Revenue features ---------------------------------------------
    df["TotalRevenue"] = (
        df["InStoreRevenue"]
        + df["UberEatsRevenue"]
        + df["DoorDashRevenue"]
        + df["SelfDeliveryRevenue"]
    )
    df["RevenuePerOrder"] = df["TotalRevenue"] / df["MonthlyOrders"]

    # ---- Profit features (for DISPLAY / EDA only -- NOT model inputs) --
    df["ProfitPerOrder"] = df["TotalNetProfit"] / df["MonthlyOrders"]
    df["ProfitMargin"] = df["TotalNetProfit"] / df["TotalRevenue"]

    # ---- Cost features ---------------------------------------------------
    # Commission is charged by the delivery aggregators (Uber Eats & DoorDash)
    # on the revenue generated through their platforms.
    df["CommissionCost"] = (
        df["UberEatsRevenue"] + df["DoorDashRevenue"]
    ) * df["CommissionRate"]
    # Self-delivery cost is already provided directly in the dataset.
    df["SelfDeliveryCost"] = df["SD_DeliveryTotalCost"]

    # ---- Channel revenue-share features -----------------------------
    # (computed from ACTUAL revenue -- different from the InStoreShare /
    # UE_share / DD_share / SD_share columns already in the raw data,
    # which appear to be planned/target shares rather than actual ones.
    # We keep both: the raw *_share columns as given, and these new
    # *RevenueShare columns computed from real revenue.)
    df["InStoreRevenueShare"] = df["InStoreRevenue"] / df["TotalRevenue"]
    df["UberEatsRevenueShare"] = df["UberEatsRevenue"] / df["TotalRevenue"]
    df["DoorDashRevenueShare"] = df["DoorDashRevenue"] / df["TotalRevenue"]
    df["SelfDeliveryRevenueShare"] = df["SelfDeliveryRevenue"] / df["TotalRevenue"]

    # ---- Simple interaction features --------------------------------
    # How much commission "drag" Uber Eats revenue is creating.
    df["Commission_UE_Impact"] = df["CommissionRate"] * df["UberEatsRevenue"]
    # How much delivery cost "drag" self-delivery orders are creating.
    df["DeliveryCost_SD_Impact"] = df["DeliveryCostPerOrder"] * df["SelfDeliveryOrders"]

    # ---- Growth feature -----------------------------------------------
    df["GrowthAdjustedOrders"] = df["MonthlyOrders"] * df["GrowthFactor"]

    return df


def get_model_ready_data(path: str = DATA_PATH):
    """
    Full pipeline: load -> clean -> engineer features -> split into
    X (features) and y (target), excluding all leakage columns.

    Returns
    -------
    df : full engineered dataframe (useful for EDA)
    X  : feature dataframe used for modeling
    y  : target series (TotalNetProfit)
    """
    df = load_raw_data(path)
    df = clean_data(df)
    df = engineer_features(df)

    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    return df, X, y


EXCLUDED_COLUMNS_EXPLANATION = {
    "InStoreNetProfit": "Directly summed to create the target (TotalNetProfit). Using it as an input would let the model 'cheat'.",
    "UberEatsNetProfit": "Directly summed to create the target. Same leakage reason as above.",
    "DoorDashNetProfit": "Directly summed to create the target. Same leakage reason as above.",
    "SelfDeliveryNetProfit": "Directly summed to create the target. Same leakage reason as above.",
    "ProfitPerOrder": "Calculated FROM the target (TotalNetProfit / MonthlyOrders). Leakage.",
    "ProfitMargin": "Calculated FROM the target (TotalNetProfit / TotalRevenue). Leakage.",
    "RestaurantID": "Just an identifier/serial number -- has no real business meaning for prediction.",
    "RestaurantName": "Just a text label -- has no real business meaning for prediction.",
}
