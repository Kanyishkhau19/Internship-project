import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv("SkyCity Auckland Restaurants & Bars.csv")

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# -----------------------------------
# CREATE TOTAL NET PROFIT
# -----------------------------------

df["TotalNetProfit"] = (
    df["InStoreNetProfit"]
    + df["UberEatsNetProfit"]
    + df["DoorDashNetProfit"]
    + df["SelfDeliveryNetProfit"]
)


# -----------------------------------
# BASIC INFORMATION
# -----------------------------------

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics:")
print(df.describe())
# -----------------------------------
# GRAPH 1: PROFIT DISTRIBUTION
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["TotalNetProfit"],
    bins=30
)

plt.xlabel("Total Net Profit ($)")
plt.ylabel("Number of Restaurants")

plt.title("Distribution of Monthly Net Profit")

plt.xlim(
    df["TotalNetProfit"].min() - 1000,
    df["TotalNetProfit"].max() + 1000
)

plt.tight_layout()

plt.show()
# -----------------------------------
# GRAPH 2: PROFIT BY CUISINE
# -----------------------------------

cuisine_profit = (
    df.groupby("CuisineType")["TotalNetProfit"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 5))

cuisine_profit.plot(kind="bar")

plt.xlabel("Cuisine Type")
plt.ylabel("Average Net Profit ($)")

plt.title("Average Net Profit by Cuisine")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
# -----------------------------------
# GRAPH 3: ORDERS VS PROFIT
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["MonthlyOrders"],
    df["TotalNetProfit"]
)

plt.xlabel("Monthly Orders")
plt.ylabel("Total Net Profit ($)")

plt.title("Monthly Orders vs Net Profit")

plt.tight_layout()

plt.show()
# -----------------------------------
# GRAPH 4: COMMISSION VS PROFIT
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["CommissionRate"],
    df["TotalNetProfit"]
)

plt.xlabel("Commission Rate")
plt.ylabel("Total Net Profit ($)")

plt.title("Commission Rate vs Net Profit")

plt.tight_layout()

plt.show()
# -----------------------------------
# GRAPH 5: DELIVERY COST VS PROFIT
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["DeliveryCostPerOrder"],
    df["SelfDeliveryNetProfit"]
)

plt.xlabel("Delivery Cost per Order ($)")
plt.ylabel("Self Delivery Net Profit ($)")

plt.title("Delivery Cost vs Self Delivery Profit")

plt.tight_layout()

plt.show()