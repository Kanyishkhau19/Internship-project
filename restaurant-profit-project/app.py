"""
app.py
------
Streamlit dashboard for the SkyCity Auckland Restaurant Profit project.

Sections (selectable from the sidebar):
    1. Dashboard            - overview KPIs + charts
    2. Profit Prediction    - enter inputs, get predicted profit
    3. What-If Analysis     - sliders to compare Current vs Scenario profit
    4. Sensitivity Analysis - how profit changes as ONE factor changes
    5. Optimization         - simple grid-search for the best channel mix

Run with:
    streamlit run app.py

NOTE: run `python train_model.py` first, so that model.pkl exists.
"""

import json
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

from ml_utils import get_model_ready_data, ALL_FEATURES, CATEGORICAL_FEATURES

sns.set_style("whitegrid")

st.set_page_config(
    page_title="SkyCity Restaurant Profit Predictor",
    page_icon="🍽️",
    layout="wide",
)

# ----------------------------------------------------------------------
# Cached loaders (so the CSV & model are not reloaded on every interaction)
# ----------------------------------------------------------------------

@st.cache_data
def load_data():
    df, X, y = get_model_ready_data()
    return df


@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    with open("model_metadata.json") as f:
        metadata = json.load(f)
    return model, metadata


df = load_data()
model, metadata = load_model()
best_model_name = metadata["best_model_name"]


def build_feature_row(row: dict) -> pd.DataFrame:
    """Turn a dict of feature values into a single-row DataFrame with
    columns in the exact order the model pipeline expects."""
    return pd.DataFrame([row])[ALL_FEATURES]


def predict_profit(row: dict) -> float:
    X_row = build_feature_row(row)
    return float(model.predict(X_row)[0])


def recompute_dependent_features(row: dict) -> dict:
    """
    When a user changes a 'raw' input (like CommissionRate, a channel
    share, or DeliveryCostPerOrder) in the What-If / Optimization tools,
    a few ENGINEERED features depend on those raw inputs and must be
    recalculated so the model sees a consistent, realistic row.

    This keeps the What-If tool honest: we are not just changing one
    number in isolation, we are recalculating everything that number
    actually affects.
    """
    row = row.copy()
    row["CommissionCost"] = (row["UberEatsRevenue"] + row["DoorDashRevenue"]) * row["CommissionRate"]
    row["Commission_UE_Impact"] = row["CommissionRate"] * row["UberEatsRevenue"]
    row["DeliveryCost_SD_Impact"] = row["DeliveryCostPerOrder"] * row["SelfDeliveryOrders"]
    row["GrowthAdjustedOrders"] = row["MonthlyOrders"] * row["GrowthFactor"]
    return row


# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------

st.sidebar.title("🍽️ SkyCity Profit Predictor")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Profit Prediction", "What-If Analysis", "Sensitivity Analysis", "Optimization"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Best model in use: **{best_model_name}**")
st.sidebar.caption("College ML project - SkyCity Auckland Restaurants & Bars dataset")

# ----------------------------------------------------------------------
# Restaurant selector (used across several pages as the "base" scenario)
# ----------------------------------------------------------------------

restaurant_names = df["RestaurantName"] + " (#" + df["RestaurantID"].astype(str) + ")"
restaurant_lookup = dict(zip(restaurant_names, df["RestaurantID"]))


def get_selected_restaurant(key_suffix=""):
    choice = st.selectbox(
        "Choose a restaurant (used as the base scenario)",
        options=restaurant_names,
        key=f"restaurant_select_{key_suffix}",
    )
    rid = restaurant_lookup[choice]
    restaurant_row = df[df["RestaurantID"] == rid].iloc[0]
    return restaurant_row


# ========================================================================
# PAGE 1: DASHBOARD
# ========================================================================
if page == "Dashboard":
    st.title("📊 Dashboard - SkyCity Auckland Restaurants & Bars")
    st.write(
        "Overview of the dataset used for this project: monthly performance "
        "snapshots for restaurants operating across In-Store, Uber Eats, "
        "DoorDash, and Self-Delivery channels."
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Restaurants", f"{df['RestaurantID'].nunique():,}")
    col2.metric("Total Monthly Orders", f"{df['MonthlyOrders'].sum():,.0f}")
    col3.metric("Avg. Revenue / Restaurant", f"${df['TotalRevenue'].mean():,.0f}")
    col4.metric("Avg. Net Profit / Restaurant", f"${df['TotalNetProfit'].mean():,.0f}")
    col5.metric("Avg. Profit per Order", f"${df['ProfitPerOrder'].mean():,.2f}")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Net Profit by Cuisine Type")
        fig, ax = plt.subplots(figsize=(6, 4))
        order = df.groupby("CuisineType")["TotalNetProfit"].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x="CuisineType", y="TotalNetProfit", order=order, ax=ax, palette="Set2")
        ax.tick_params(axis="x", rotation=35)
        st.pyplot(fig)

    with c2:
        st.subheader("Net Profit by Segment")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="Segment", y="TotalNetProfit", ax=ax, palette="Set3")
        st.pyplot(fig)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Revenue by Channel (Total, all restaurants)")
        channel_revenue = df[["InStoreRevenue", "UberEatsRevenue", "DoorDashRevenue", "SelfDeliveryRevenue"]].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        channel_revenue.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
        ax.set_ylabel("Revenue ($)")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)

    with c4:
        st.subheader("Net Profit Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["TotalNetProfit"], bins=30, kde=True, ax=ax, color="#C44E52")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(20))

# ========================================================================
# PAGE 2: PROFIT PREDICTION
# ========================================================================
elif page == "Profit Prediction":
    st.title("🔮 Profit Prediction")
    st.write(
        "Select a restaurant as a starting point, then adjust its inputs "
        "to predict Total Monthly Net Profit using the trained "
        f"**{best_model_name}** model."
    )

    base = get_selected_restaurant("predict")

    st.markdown("#### Restaurant Inputs")
    col1, col2, col3 = st.columns(3)

    with col1:
        cuisine = st.selectbox("Cuisine Type", sorted(df["CuisineType"].unique()),
                                index=sorted(df["CuisineType"].unique()).index(base["CuisineType"]))
        segment = st.selectbox("Segment", sorted(df["Segment"].unique()),
                                index=sorted(df["Segment"].unique()).index(base["Segment"]))
        subregion = st.selectbox("Subregion", sorted(df["Subregion"].unique()),
                                  index=sorted(df["Subregion"].unique()).index(base["Subregion"]))
        monthly_orders = st.number_input("Monthly Orders", min_value=1, value=int(base["MonthlyOrders"]))
        aov = st.number_input("Average Order Value (AOV, $)", min_value=1.0, value=float(base["AOV"]))

    with col2:
        commission_rate = st.slider("Commission Rate", 0.0, 0.5, float(base["CommissionRate"]), 0.01)
        delivery_cost = st.number_input("Delivery Cost per Order ($)", min_value=0.0, value=float(base["DeliveryCostPerOrder"]))
        delivery_radius = st.slider("Delivery Radius (KM)", 1, 25, int(base["DeliveryRadiusKM"]))
        cogs_rate = st.slider("COGS Rate", 0.0, 0.6, float(base["COGSRate"]), 0.01)
        opex_rate = st.slider("OPEX Rate", 0.0, 0.6, float(base["OPEXRate"]), 0.01)

    with col3:
        growth_factor = st.slider("Growth Factor", 0.8, 1.3, float(base["GrowthFactor"]), 0.01)
        in_store_share = st.slider("In-Store Share", 0.0, 1.0, float(base["InStoreShare"]), 0.01)
        ue_share = st.slider("Uber Eats Share", 0.0, 1.0, float(base["UE_share"]), 0.01)
        dd_share = st.slider("DoorDash Share", 0.0, 1.0, float(base["DD_share"]), 0.01)
        sd_share = st.slider("Self-Delivery Share", 0.0, 1.0, float(base["SD_share"]), 0.01)

    # Orders are split across channels using the same proportions as the
    # base restaurant's actual order counts (kept simple, on purpose).
    order_channel_ratio = {
        "InStoreOrders": base["InStoreOrders"] / base["MonthlyOrders"],
        "UberEatsOrders": base["UberEatsOrders"] / base["MonthlyOrders"],
        "DoorDashOrders": base["DoorDashOrders"] / base["MonthlyOrders"],
        "SelfDeliveryOrders": base["SelfDeliveryOrders"] / base["MonthlyOrders"],
    }

    row = {feat: base[feat] for feat in ALL_FEATURES}
    row.update({
        "CuisineType": cuisine, "Segment": segment, "Subregion": subregion,
        "MonthlyOrders": monthly_orders, "AOV": aov,
        "CommissionRate": commission_rate, "DeliveryCostPerOrder": delivery_cost,
        "DeliveryRadiusKM": delivery_radius, "COGSRate": cogs_rate, "OPEXRate": opex_rate,
        "GrowthFactor": growth_factor,
        "InStoreShare": in_store_share, "UE_share": ue_share, "DD_share": dd_share, "SD_share": sd_share,
    })
    for col, ratio in order_channel_ratio.items():
        row[col] = monthly_orders * ratio
    row["InStoreRevenue"] = row["InStoreOrders"] * aov
    row["UberEatsRevenue"] = row["UberEatsOrders"] * aov
    row["DoorDashRevenue"] = row["DoorDashOrders"] * aov
    row["SelfDeliveryRevenue"] = row["SelfDeliveryOrders"] * aov
    row["TotalRevenue"] = row["InStoreRevenue"] + row["UberEatsRevenue"] + row["DoorDashRevenue"] + row["SelfDeliveryRevenue"]
    row["RevenuePerOrder"] = row["TotalRevenue"] / monthly_orders
    row["SD_DeliveryTotalCost"] = delivery_cost * row["SelfDeliveryOrders"]
    row["InStoreRevenueShare"] = row["InStoreRevenue"] / row["TotalRevenue"]
    row["UberEatsRevenueShare"] = row["UberEatsRevenue"] / row["TotalRevenue"]
    row["DoorDashRevenueShare"] = row["DoorDashRevenue"] / row["TotalRevenue"]
    row["SelfDeliveryRevenueShare"] = row["SelfDeliveryRevenue"] / row["TotalRevenue"]
    row = recompute_dependent_features(row)

    predicted_profit = predict_profit(row)
    profit_per_order = predicted_profit / monthly_orders

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Predicted Monthly Net Profit", f"${predicted_profit:,.2f}")
    c2.metric("Profit Per Order", f"${profit_per_order:,.2f}")

# ========================================================================
# PAGE 3: WHAT-IF ANALYSIS
# ========================================================================
elif page == "What-If Analysis":
    st.title("🎛️ What-If Analysis")
    st.write(
        "Start from a restaurant's CURRENT numbers, then move the sliders "
        "to build a SCENARIO and see how predicted profit changes. "
        "All predictions come from the trained model - nothing is hard-coded."
    )

    base = get_selected_restaurant("whatif")

    # ---- CURRENT scenario (unchanged base row) ----
    current_row = {feat: base[feat] for feat in ALL_FEATURES}
    current_row = recompute_dependent_features(current_row)
    current_profit = predict_profit(current_row)

    st.markdown("#### Adjust the Scenario")
    col1, col2 = st.columns(2)

    with col1:
        commission_rate = st.slider("Commission Rate", 0.0, 0.5, float(base["CommissionRate"]), 0.01, key="wi_comm")
        delivery_cost = st.number_input("Delivery Cost per Order ($)", min_value=0.0,
                                         value=float(base["DeliveryCostPerOrder"]), key="wi_dc")
        delivery_radius = st.slider("Delivery Radius (KM)", 1, 25, int(base["DeliveryRadiusKM"]), key="wi_radius")
        growth_factor = st.slider("Growth Factor", 0.8, 1.3, float(base["GrowthFactor"]), 0.01, key="wi_growth")

    with col2:
        in_store_share = st.slider("In-Store Share", 0.0, 1.0, float(base["InStoreShare"]), 0.01, key="wi_is")
        ue_share = st.slider("Uber Eats Share", 0.0, 1.0, float(base["UE_share"]), 0.01, key="wi_ue")
        dd_share = st.slider("DoorDash Share", 0.0, 1.0, float(base["DD_share"]), 0.01, key="wi_dd")
        sd_share = st.slider("Self-Delivery Share", 0.0, 1.0, float(base["SD_share"]), 0.01, key="wi_sd")

    share_total = in_store_share + ue_share + dd_share + sd_share
    if abs(share_total - 1.0) > 0.001:
        st.warning(
            f"⚠️ Channel shares add up to {share_total:.2f} (should be 1.00 / 100%). "
            "The prediction below still runs, but for a realistic scenario, adjust the "
            "sliders so they sum to 100%."
        )

    # ---- SCENARIO row: start from current row, override changed values ----
    scenario_row = current_row.copy()
    scenario_row.update({
        "CommissionRate": commission_rate,
        "DeliveryCostPerOrder": delivery_cost,
        "DeliveryRadiusKM": delivery_radius,
        "GrowthFactor": growth_factor,
        "InStoreShare": in_store_share,
        "UE_share": ue_share,
        "DD_share": dd_share,
        "SD_share": sd_share,
        "SD_DeliveryTotalCost": delivery_cost * current_row["SelfDeliveryOrders"],
    })
    scenario_row = recompute_dependent_features(scenario_row)
    scenario_profit = predict_profit(scenario_row)

    diff = scenario_profit - current_profit
    pct_change = (diff / current_profit) * 100 if current_profit != 0 else 0.0

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Predicted Profit", f"${current_profit:,.2f}")
    c2.metric("Scenario Predicted Profit", f"${scenario_profit:,.2f}")
    c3.metric("Profit Difference", f"${diff:,.2f}")
    c4.metric("Percentage Change", f"{pct_change:+.2f}%")

    st.markdown("#### Comparison Chart")
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Current", "Scenario"], [current_profit, scenario_profit],
                   color=["#4C72B0", "#55A868"])
    ax.set_ylabel("Predicted Net Profit ($)")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"${height:,.0f}", (bar.get_x() + bar.get_width() / 2, height),
                    ha="center", va="bottom")
    st.pyplot(fig)

# ========================================================================
# PAGE 4: SENSITIVITY ANALYSIS
# ========================================================================
elif page == "Sensitivity Analysis":
    st.title("📈 Sensitivity Analysis")
    st.write(
        "See how predicted profit changes when ONE factor changes, holding "
        "everything else constant. This helps understand which factors the "
        "model treats as most sensitive for a given restaurant."
    )

    base = get_selected_restaurant("sensitivity")
    base_row = {feat: base[feat] for feat in ALL_FEATURES}

    def sweep(feature_name, values):
        profits = []
        for v in values:
            row = base_row.copy()
            row[feature_name] = v
            if feature_name == "DeliveryCostPerOrder":
                row["SD_DeliveryTotalCost"] = v * row["SelfDeliveryOrders"]
            row = recompute_dependent_features(row)
            profits.append(predict_profit(row))
        return profits

    factors = {
        "Commission Rate": ("CommissionRate", np.linspace(0.05, 0.45, 15)),
        "Delivery Cost per Order ($)": ("DeliveryCostPerOrder", np.linspace(0.5, 10, 15)),
        "Uber Eats Share": ("UE_share", np.linspace(0.0, 1.0, 15)),
        "Self-Delivery Share": ("SD_share", np.linspace(0.0, 1.0, 15)),
        "Delivery Radius (KM)": ("DeliveryRadiusKM", np.linspace(1, 25, 15)),
    }

    for label, (feat, values) in factors.items():
        st.subheader(f"{label} vs Predicted Profit")
        profits = sweep(feat, values)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(values, profits, marker="o", color="#4C72B0")
        ax.set_xlabel(label)
        ax.set_ylabel("Predicted Net Profit ($)")
        ax.axvline(base_row[feat], color="red", linestyle="--", alpha=0.6, label="Current value")
        ax.legend()
        st.pyplot(fig)

# ========================================================================
# PAGE 5: OPTIMIZATION
# ========================================================================
elif page == "Optimization":
    st.title("🧭 Channel Mix Optimization")
    st.write(
        "A simple grid-search over reasonable channel-mix combinations. "
        "For each VALID combination (shares add up to 100%), the trained "
        "model predicts profit. The best-performing combination is "
        "recommended below."
    )

    base = get_selected_restaurant("optimize")
    base_row = {feat: base[feat] for feat in ALL_FEATURES}
    current_row = recompute_dependent_features(base_row.copy())
    current_profit = predict_profit(current_row)

    st.markdown("#### Grid Search Options (tick marks to try)")
    c1, c2, c3 = st.columns(3)
    with c1:
        in_store_options = st.multiselect("In-Store %", [10, 20, 30, 40, 50], default=[20, 30, 40])
    with c2:
        ue_options = st.multiselect("Uber Eats %", [10, 20, 30, 40, 50], default=[20, 30, 40])
    with c3:
        dd_options = st.multiselect("DoorDash %", [10, 20, 30], default=[10, 20, 30])

    st.caption("Self-Delivery % is automatically set to whatever remains, so all four shares sum to 100%.")

    if st.button("Run Optimization"):
        candidates = []
        for is_pct, ue_pct, dd_pct in itertools.product(in_store_options, ue_options, dd_options):
            sd_pct = 100 - is_pct - ue_pct - dd_pct
            if sd_pct < 0 or sd_pct > 100:
                continue  # invalid combination, skip

            row = base_row.copy()
            row.update({
                "InStoreShare": is_pct / 100,
                "UE_share": ue_pct / 100,
                "DD_share": dd_pct / 100,
                "SD_share": sd_pct / 100,
            })
            row = recompute_dependent_features(row)
            predicted = predict_profit(row)
            candidates.append({
                "InStore%": is_pct, "UberEats%": ue_pct, "DoorDash%": dd_pct, "SelfDelivery%": sd_pct,
                "PredictedProfit": predicted,
            })

        if not candidates:
            st.error("No valid combinations found. Try selecting different percentage options.")
        else:
            results_df = pd.DataFrame(candidates).sort_values("PredictedProfit", ascending=False).reset_index(drop=True)
            best = results_df.iloc[0]
            uplift_pct = ((best["PredictedProfit"] - current_profit) / current_profit) * 100 if current_profit != 0 else 0.0

            st.markdown("---")
            st.subheader("Current Channel Mix")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("In-Store", f"{base_row['InStoreShare']*100:.0f}%")
            c2.metric("Uber Eats", f"{base_row['UE_share']*100:.0f}%")
            c3.metric("DoorDash", f"{base_row['DD_share']*100:.0f}%")
            c4.metric("Self-Delivery", f"{base_row['SD_share']*100:.0f}%")
            st.metric("Current Predicted Profit", f"${current_profit:,.2f}")

            st.subheader("Recommended Channel Mix")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("In-Store", f"{best['InStore%']:.0f}%")
            c2.metric("Uber Eats", f"{best['UberEats%']:.0f}%")
            c3.metric("DoorDash", f"{best['DoorDash%']:.0f}%")
            c4.metric("Self-Delivery", f"{best['SelfDelivery%']:.0f}%")
            st.metric("Optimized Predicted Profit", f"${best['PredictedProfit']:,.2f}",
                       delta=f"{uplift_pct:+.2f}% Optimization Uplift")

            if uplift_pct > 0.5:
                st.success(
                    f"Recommendation: shifting toward In-Store {best['InStore%']:.0f}% / "
                    f"Uber Eats {best['UberEats%']:.0f}% / DoorDash {best['DoorDash%']:.0f}% / "
                    f"Self-Delivery {best['SelfDelivery%']:.0f}% is predicted to improve profit "
                    f"by {uplift_pct:.2f}% for this restaurant, based on the trained model."
                )
            else:
                st.info(
                    "The current channel mix is already close to optimal among the "
                    "combinations tested - no meaningful uplift was found."
                )

            with st.expander("See all tested combinations"):
                st.dataframe(results_df)
    else:
        st.info("Select percentage options above and click **Run Optimization**.")
