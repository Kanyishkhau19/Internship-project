import pandas as pd


# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv("SkyCity Auckland Restaurants & Bars.csv")


# -----------------------------------
# TOTAL NET PROFIT
# -----------------------------------

df["TotalNetProfit"] = (
    df["InStoreNetProfit"]
    + df["UberEatsNetProfit"]
    + df["DoorDashNetProfit"]
    + df["SelfDeliveryNetProfit"]
)


# ===================================
# KPI 1: PROFIT PER ORDER
# ===================================

df["ProfitPerOrder"] = (
    df["TotalNetProfit"] / df["MonthlyOrders"]
)


# ===================================
# KPI 2: PROFIT SENSITIVITY INDEX
# ===================================

# Increase commission by 5 percentage points

commission_increase = 0.05

commission_impact = (
    df["UberEatsRevenue"] + df["DoorDashRevenue"]
) * commission_increase

df["ProfitAfterCommissionIncrease"] = (
    df["TotalNetProfit"] - commission_impact
)

df["ProfitSensitivityIndex"] = (
    commission_impact /
    df["TotalNetProfit"].abs()
) * 100


# ===================================
# KPI 3: CHANNEL MIX EFFICIENCY
# ===================================

df["DeliveryShare"] = (
    df["UE_share"]
    + df["DD_share"]
    + df["SD_share"]
)

df["ChannelMixEfficiency"] = (
    df["TotalNetProfit"] /
    df["MonthlyOrders"]
)


# ===================================
# KPI 4: BREAK-EVEN COMMISSION RATE
# ===================================

# At this commission level:
# Revenue - COGS - OPEX - Commission = 0

df["BreakEvenCommissionRate"] = (
    1
    - df["COGSRate"]
    - df["OPEXRate"]
)


# ===================================
# DISPLAY RESULTS
# ===================================

print("\n===================================")
print("SKYCITY KPI ANALYSIS")
print("===================================")

print("\nAverage Monthly Net Profit:")
print(
    round(df["TotalNetProfit"].mean(), 2)
)

print("\nAverage Profit per Order:")
print(
    round(df["ProfitPerOrder"].mean(), 2)
)

print("\nAverage Profit Sensitivity Index:")
print(
    round(df["ProfitSensitivityIndex"].mean(), 2),
    "%"
)

print("\nAverage Channel Mix Efficiency:")
print(
    round(df["ChannelMixEfficiency"].mean(), 2)
)

print("\nAverage Break-Even Commission Rate:")
print(
    round(
        df["BreakEvenCommissionRate"].mean() * 100,
        2
    ),
    "%"
)


# ===================================
# CURRENT COMMISSION RATE
# ===================================

print("\nAverage Current Commission Rate:")

print(
    round(
        df["CommissionRate"].mean() * 100,
        2
    ),
    "%"
)


# ===================================
# COMPARISON
# ===================================

print("\n===================================")
print("KPI SUMMARY")
print("===================================")

print(
    "Average Net Profit: $",
    round(df["TotalNetProfit"].mean(), 2)
)

print(
    "Profit per Order: $",
    round(df["ProfitPerOrder"].mean(), 2)
)

print(
    "Profit Sensitivity Index:",
    round(df["ProfitSensitivityIndex"].mean(), 2),
    "%"
)

print(
    "Channel Mix Efficiency: $",
    round(df["ChannelMixEfficiency"].mean(), 2),
    "per order"
)

print(
    "Break-Even Commission:",
    round(
        df["BreakEvenCommissionRate"].mean() * 100,
        2
    ),
    "%"
)