import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="SkyCity Profit Optimizer",
    page_icon="🍽️",
    layout="wide"
)


# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------

df = pd.read_csv("cleaned_data.csv")

model = joblib.load(
    "models/profit_model.pkl"
)

model_info = joblib.load(
    "models/model_info.pkl"
)


# --------------------------------------------------
# KPI CALCULATIONS
# --------------------------------------------------

# Calculate Total Net Profit if it is not already present

if "TotalNetProfit" not in df.columns:

    df["TotalNetProfit"] = (
        df["InStoreNetProfit"]
        + df["UberEatsNetProfit"]
        + df["DoorDashNetProfit"]
        + df["SelfDeliveryNetProfit"]
    )


# Calculate Profit Per Order if it is not already present

if "ProfitPerOrder" not in df.columns:

    df["ProfitPerOrder"] = (
        df["TotalNetProfit"]
        / df["MonthlyOrders"]
    )


# --------------------------------------------------
# PROFIT SENSITIVITY INDEX
# --------------------------------------------------

# Estimate the effect of a 5% increase in
# third-party delivery commission

commission_impact = (
    df["UberEatsRevenue"]
    + df["DoorDashRevenue"]
) * 0.05


# Avoid division problems when profit is zero

df["ProfitSensitivityIndex"] = np.where(
    df["TotalNetProfit"].abs() > 0,
    (
        commission_impact
        / df["TotalNetProfit"].abs()
    ) * 100,
    0
)


# --------------------------------------------------
# BREAK-EVEN COMMISSION RATE
# --------------------------------------------------

# Simplified break-even commission based on
# revenue after COGS and OPEX

df["BreakEvenCommissionRate"] = (
    1
    - df["COGSRate"]
    - df["OPEXRate"]
)


# --------------------------------------------------
# AVERAGE KPI VALUES
# --------------------------------------------------

average_net_profit = (
    df["TotalNetProfit"].mean()
)

average_profit_order = (
    df["ProfitPerOrder"].mean()
)

average_sensitivity = (
    df["ProfitSensitivityIndex"].mean()
)

average_channel_efficiency = (
    df["ProfitPerOrder"].mean()
)

average_break_even_commission = (
    df["BreakEvenCommissionRate"].mean()
)

average_current_commission = (
    df["CommissionRate"].mean()
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title(
    "🍽️ SkyCity Restaurant Profit Optimization"
)

st.write(
    "Predictive Modeling and Profit Optimization "
    "for Multi-Channel Restaurant Operations"
)

st.write(
    "This dashboard helps analyze how restaurant "
    "profit changes with different delivery channel strategies."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Profit Prediction",
        "What-If Analysis",
        "Optimization"
    ]
)


# ==================================================
# PAGE 1 - DASHBOARD
# ==================================================

if page == "Dashboard":

    st.header("📊 Restaurant Overview")


    # --------------------------------------------------
    # BASIC KPIs
    # --------------------------------------------------

    total_restaurants = (
        df["RestaurantID"].nunique()
    )

    average_orders = (
        df["MonthlyOrders"].mean()
    )


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Restaurants",
        total_restaurants
    )

    col2.metric(
        "Average Monthly Profit",
        f"${average_net_profit:,.2f}"
    )

    col3.metric(
        "Average Monthly Orders",
        f"{average_orders:,.0f}"
    )

    col4.metric(
        "Profit / Order",
        f"${average_profit_order:.2f}"
    )


    # --------------------------------------------------
    # PROJECT KPIs
    # --------------------------------------------------

    st.subheader(
        "📌 Project Key Performance Indicators"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Predicted / Average Net Profit",
            f"${average_net_profit:,.2f}"
        )


    with col2:

        st.metric(
            "Profit Sensitivity Index",
            f"{average_sensitivity:.2f}%"
        )


    with col3:

        st.metric(
            "Channel Mix Efficiency",
            f"${average_channel_efficiency:.2f}"
        )


    col4, col5 = st.columns(2)


    with col4:

        st.metric(
            "Break-Even Commission",
            f"{average_break_even_commission:.2%}"
        )


    with col5:

        st.metric(
            "Current Commission",
            f"{average_current_commission:.2%}"
        )


    # --------------------------------------------------
    # KPI INTERPRETATION
    # --------------------------------------------------

    st.info(
        f"""
        **KPI Summary**

        • Average monthly net profit is 
        **${average_net_profit:,.2f}**

        • Average profit per order is 
        **${average_profit_order:.2f}**

        • Profit sensitivity index is 
        **{average_sensitivity:.2f}%**

        • Average break-even commission is 
        **{average_break_even_commission:.2%}**

        • Current average commission is 
        **{average_current_commission:.2%}**

        The current commission rate is relatively close
        to the estimated break-even commission level,
        showing that delivery commission is an important
        factor affecting profitability.
        """
    )


    # --------------------------------------------------
    # PROFIT DISTRIBUTION
    # --------------------------------------------------

    st.subheader(
        "Restaurant Profit Distribution"
    )


    fig, ax = plt.subplots()


    ax.hist(
        df["TotalNetProfit"],
        bins=30
    )


    ax.set_xlabel(
        "Monthly Net Profit"
    )

    ax.set_ylabel(
        "Number of Restaurants"
    )

    ax.set_title(
        "Distribution of Monthly Net Profit"
    )


    st.pyplot(fig)


    # --------------------------------------------------
    # PROFIT BY CUISINE
    # --------------------------------------------------

    st.subheader(
        "Profit by Cuisine"
    )


    cuisine_profit = (
        df.groupby("CuisineType")["TotalNetProfit"]
        .mean()
        .sort_values(ascending=False)
    )


    st.bar_chart(
        cuisine_profit
    )


# ==================================================
# PAGE 2 - PROFIT PREDICTION
# ==================================================

elif page == "Profit Prediction":

    st.header(
        "🔮 Monthly Profit Prediction"
    )

    st.write(
        "Enter restaurant and channel information "
        "to predict monthly net profit."
    )


    # --------------------------------------------------
    # INPUTS
    # --------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        cuisine = st.selectbox(
            "Cuisine Type",
            sorted(
                df["CuisineType"].unique()
            )
        )


        segment = st.selectbox(
            "Business Segment",
            sorted(
                df["Segment"].unique()
            )
        )


        subregion = st.selectbox(
            "Subregion",
            sorted(
                df["Subregion"].unique()
            )
        )


        monthly_orders = st.number_input(
            "Monthly Orders",
            min_value=100,
            max_value=5000,
            value=1000
        )


        aov = st.number_input(
            "Average Order Value ($)",
            min_value=10.0,
            max_value=100.0,
            value=40.0
        )


        growth = st.slider(
            "Growth Factor",
            0.99,
            1.05,
            1.02,
            0.01
        )


    with col2:

        cogs = st.slider(
            "COGS Rate",
            0.20,
            0.40,
            0.30,
            0.01
        )


        opex = st.slider(
            "OPEX Rate",
            0.20,
            0.55,
            0.35,
            0.01
        )


        commission = st.slider(
            "Commission Rate",
            0.10,
            0.40,
            0.28,
            0.01
        )


        delivery_radius = st.slider(
            "Delivery Radius (KM)",
            3,
            18,
            10
        )


        delivery_cost = st.slider(
            "Self Delivery Cost / Order ($)",
            0.89,
            5.31,
            3.00,
            0.10
        )


    # --------------------------------------------------
    # CHANNEL MIX
    # --------------------------------------------------

    st.subheader(
        "Channel Mix"
    )


    instore_share = st.slider(
        "In-Store Share",
        0.10,
        0.80,
        0.40,
        0.05
    )


    ue_share = st.slider(
        "Uber Eats Share of Delivery",
        0.00,
        1.00,
        0.45,
        0.05
    )


    dd_share = st.slider(
        "DoorDash Share of Delivery",
        0.00,
        max(0.00, 1.00 - ue_share),
        min(0.25, max(0.00, 1.00 - ue_share)),
        0.05
    )


    sd_share = (
        1
        - ue_share
        - dd_share
    )


    st.info(
        f"Self Delivery Share: {sd_share:.0%}"
    )


    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------

    commission_impact = (
        commission
        * ue_share
    )


    delivery_impact = (
        delivery_cost
        * sd_share
    )


    growth_adjusted_orders = (
        monthly_orders
        * growth
    )


    # --------------------------------------------------
    # CREATE INPUT DATA
    # --------------------------------------------------

    input_data = pd.DataFrame({

        "CuisineType": [
            cuisine
        ],

        "Segment": [
            segment
        ],

        "Subregion": [
            subregion
        ],

        "GrowthFactor": [
            growth
        ],

        "AOV": [
            aov
        ],

        "MonthlyOrders": [
            monthly_orders
        ],

        "COGSRate": [
            cogs
        ],

        "OPEXRate": [
            opex
        ],

        "CommissionRate": [
            commission
        ],

        "DeliveryRadiusKM": [
            delivery_radius
        ],

        "DeliveryCostPerOrder": [
            delivery_cost
        ],

        "InStoreShare": [
            instore_share
        ],

        "UE_share": [
            ue_share
        ],

        "DD_share": [
            dd_share
        ],

        "SD_share": [
            sd_share
        ],

        "CommissionImpact": [
            commission_impact
        ],

        "DeliveryCostImpact": [
            delivery_impact
        ],

        "GrowthAdjustedOrders": [
            growth_adjusted_orders
        ]
    })


    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    if st.button(
        "Predict Profit"
    ):

        prediction = model.predict(
            input_data
        )[0]


        confidence = model_info[
            "rmse"
        ]


        st.success(
            f"Predicted Monthly Net Profit: "
            f"${prediction:,.2f}"
        )


        col1, col2, col3 = st.columns(3)


        col1.metric(
            "Predicted Profit",
            f"${prediction:,.2f}"
        )


        col2.metric(
            "Approx. Profit / Order",
            f"${prediction / monthly_orders:.2f}"
        )


        col3.metric(
            "Model",
            model_info["model_name"]
        )


        # --------------------------------------------------
        # PREDICTION RANGE
        # --------------------------------------------------

        st.subheader(
            "Prediction Range"
        )


        lower = max(
            0,
            prediction - confidence
        )


        upper = (
            prediction + confidence
        )


        st.write(
            f"Approximate range based on model RMSE: "
            f"${lower:,.2f} to ${upper:,.2f}"
        )


# ==================================================
# PAGE 3 - WHAT IF ANALYSIS
# ==================================================

elif page == "What-If Analysis":

    st.header(
        "🎯 What-If Scenario Analysis"
    )


    st.write(
        "Change commission rate and channel mix "
        "to understand their effect on profit."
    )


    # --------------------------------------------------
    # SELECT RESTAURANT
    # --------------------------------------------------

    restaurant = st.selectbox(
        "Choose a Restaurant",
        df["RestaurantName"].unique()
    )


    row = df[
        df["RestaurantName"] == restaurant
    ].iloc[0]


    # --------------------------------------------------
    # CURRENT SITUATION
    # --------------------------------------------------

    st.subheader(
        "Current Situation"
    )


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Current Profit",
        f"${row['TotalNetProfit']:,.2f}"
    )


    col2.metric(
        "Orders",
        f"{row['MonthlyOrders']:,.0f}"
    )


    col3.metric(
        "Commission",
        f"{row['CommissionRate']:.0%}"
    )


    col4.metric(
        "Delivery Radius",
        f"{row['DeliveryRadiusKM']} km"
    )


    # --------------------------------------------------
    # NEW STRATEGY
    # --------------------------------------------------

    st.subheader(
        "Change the Strategy"
    )


    new_commission = st.slider(
        "New Commission Rate",
        0.10,
        0.40,
        float(row["CommissionRate"]),
        0.01
    )


    new_ue = st.slider(
        "Uber Eats Delivery Share",
        0.00,
        1.00,
        float(row["UE_share"]),
        0.05
    )


    max_dd = max(
        0.00,
        1.00 - new_ue
    )


    new_dd = st.slider(
        "DoorDash Delivery Share",
        0.00,
        max_dd,
        min(
            float(row["DD_share"]),
            max_dd
        ),
        0.05
    )


    new_sd = (
        1
        - new_ue
        - new_dd
    )


    st.info(
        f"New Self Delivery Share: {new_sd:.0%}"
    )


    # --------------------------------------------------
    # NEW INPUT DATA
    # --------------------------------------------------

    new_input = pd.DataFrame({

        "CuisineType": [
            row["CuisineType"]
        ],

        "Segment": [
            row["Segment"]
        ],

        "Subregion": [
            row["Subregion"]
        ],

        "GrowthFactor": [
            row["GrowthFactor"]
        ],

        "AOV": [
            row["AOV"]
        ],

        "MonthlyOrders": [
            row["MonthlyOrders"]
        ],

        "COGSRate": [
            row["COGSRate"]
        ],

        "OPEXRate": [
            row["OPEXRate"]
        ],

        "CommissionRate": [
            new_commission
        ],

        "DeliveryRadiusKM": [
            row["DeliveryRadiusKM"]
        ],

        "DeliveryCostPerOrder": [
            row["DeliveryCostPerOrder"]
        ],

        "InStoreShare": [
            row["InStoreShare"]
        ],

        "UE_share": [
            new_ue
        ],

        "DD_share": [
            new_dd
        ],

        "SD_share": [
            new_sd
        ],

        "CommissionImpact": [
            new_commission
            * new_ue
        ],

        "DeliveryCostImpact": [
            row["DeliveryCostPerOrder"]
            * new_sd
        ],

        "GrowthAdjustedOrders": [
            row["MonthlyOrders"]
            * row["GrowthFactor"]
        ]
    })


    # --------------------------------------------------
    # SCENARIO PREDICTION
    # --------------------------------------------------

    scenario_profit = model.predict(
        new_input
    )[0]


    change = (
        scenario_profit
        - row["TotalNetProfit"]
    )


    if row["TotalNetProfit"] != 0:

        percentage_change = (
            change
            / abs(row["TotalNetProfit"])
        ) * 100

    else:

        percentage_change = 0


    # --------------------------------------------------
    # SCENARIO RESULT
    # --------------------------------------------------

    st.subheader(
        "Scenario Result"
    )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Scenario Profit",
        f"${scenario_profit:,.2f}"
    )


    col2.metric(
        "Profit Change",
        f"${change:,.2f}"
    )


    col3.metric(
        "Change %",
        f"{percentage_change:.2f}%"
    )


    st.write(
        f"New Delivery Mix: "
        f"Uber Eats {new_ue:.0%}, "
        f"DoorDash {new_dd:.0%}, "
        f"Self Delivery {new_sd:.0%}"
    )


# ==================================================
# PAGE 4 - OPTIMIZATION
# ==================================================

elif page == "Optimization":

    st.header(
        "🚀 Profit Optimization"
    )


    st.write(
        "The system tests different delivery channel "
        "combinations and finds the strategy with "
        "the highest predicted profit."
    )


    # --------------------------------------------------
    # SELECT RESTAURANT
    # --------------------------------------------------

    restaurant = st.selectbox(
        "Select Restaurant",
        df["RestaurantName"].unique()
    )


    row = df[
        df["RestaurantName"] == restaurant
    ].iloc[0]


    # --------------------------------------------------
    # INITIAL VALUES
    # --------------------------------------------------

    best_profit = -999999999

    best_ue = 0

    best_dd = 0

    best_sd = 0


    # --------------------------------------------------
    # SEARCH THROUGH CHANNEL COMBINATIONS
    # --------------------------------------------------

    for ue in np.arange(
        0.10,
        0.81,
        0.10
    ):

        for dd in np.arange(
            0.10,
            0.81 - ue,
            0.10
        ):

            sd = (
                1
                - ue
                - dd
            )


            if sd < 0:

                continue


            test_data = pd.DataFrame({

                "CuisineType": [
                    row["CuisineType"]
                ],

                "Segment": [
                    row["Segment"]
                ],

                "Subregion": [
                    row["Subregion"]
                ],

                "GrowthFactor": [
                    row["GrowthFactor"]
                ],

                "AOV": [
                    row["AOV"]
                ],

                "MonthlyOrders": [
                    row["MonthlyOrders"]
                ],

                "COGSRate": [
                    row["COGSRate"]
                ],

                "OPEXRate": [
                    row["OPEXRate"]
                ],

                "CommissionRate": [
                    row["CommissionRate"]
                ],

                "DeliveryRadiusKM": [
                    row["DeliveryRadiusKM"]
                ],

                "DeliveryCostPerOrder": [
                    row["DeliveryCostPerOrder"]
                ],

                "InStoreShare": [
                    row["InStoreShare"]
                ],

                "UE_share": [
                    ue
                ],

                "DD_share": [
                    dd
                ],

                "SD_share": [
                    sd
                ],

                "CommissionImpact": [
                    row["CommissionRate"]
                    * ue
                ],

                "DeliveryCostImpact": [
                    row["DeliveryCostPerOrder"]
                    * sd
                ],

                "GrowthAdjustedOrders": [
                    row["MonthlyOrders"]
                    * row["GrowthFactor"]
                ]
            })


            predicted_profit = model.predict(
                test_data
            )[0]


            if predicted_profit > best_profit:

                best_profit = (
                    predicted_profit
                )

                best_ue = ue

                best_dd = dd

                best_sd = sd


    # --------------------------------------------------
    # CURRENT PROFIT
    # --------------------------------------------------

    current_profit = (
        row["TotalNetProfit"]
    )


    # --------------------------------------------------
    # OPTIMIZATION UPLIFT
    # --------------------------------------------------

    if current_profit != 0:

        uplift = (
            (
                best_profit
                - current_profit
            )
            / abs(current_profit)
        ) * 100

    else:

        uplift = 0


    # --------------------------------------------------
    # RECOMMENDED STRATEGY
    # --------------------------------------------------

    st.subheader(
        "Recommended Strategy"
    )


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Best Predicted Profit",
        f"${best_profit:,.2f}"
    )


    col2.metric(
        "Uber Eats",
        f"{best_ue:.0%}"
    )


    col3.metric(
        "DoorDash",
        f"{best_dd:.0%}"
    )


    col4.metric(
        "Self Delivery",
        f"{best_sd:.0%}"
    )


    # --------------------------------------------------
    # OPTIMIZATION UPLIFT
    # --------------------------------------------------

    st.success(
        f"Potential Optimization Uplift: "
        f"{uplift:.2f}%"
    )


    # --------------------------------------------------
    # CURRENT VS OPTIMIZED
    # --------------------------------------------------

    st.subheader(
        "Current vs Optimized"
    )


    comparison = pd.DataFrame({

        "Strategy": [
            "Current",
            "Optimized"
        ],

        "Profit": [
            current_profit,
            best_profit
        ]
    })


    st.bar_chart(
        comparison.set_index(
            "Strategy"
        )
    )


    # --------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------

    st.info(
        "Recommendation: Consider the optimized "
        "channel mix as a decision-support suggestion, "
        "not an automatic business decision."
    )