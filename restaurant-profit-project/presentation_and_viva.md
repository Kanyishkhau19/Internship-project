# Presentation & Viva Preparation

## Project Explanation (2-3 minutes, spoken)

"Good morning/afternoon. My project is called **Predictive Modeling and Profit Optimization for
Multi-Channel Restaurant Operations**.

Today, restaurants don't just sell food across the counter - they also sell through Uber Eats,
DoorDash, and their own delivery staff. Each of these channels has a different cost: Uber Eats and
DoorDash charge commission, self-delivery has its own delivery cost per order, and every channel
is affected by food cost and operating cost rates. So it's not always obvious which channel mix
actually makes a restaurant the most money.

For this project, I used a real dataset of 1,696 monthly restaurant snapshots from Auckland,
called SkyCity Auckland Restaurants and Bars. I first cleaned the data and checked it for missing
values and duplicates - this dataset turned out to be quite clean already. Then I did exploratory
data analysis to understand how profit varies by cuisine type, segment, and channel.

The main target I predicted is **Total Monthly Net Profit**, which I calculated by adding up the
net profit from all four channels. One important step I took was **checking for data leakage** -
since the four channel profit columns are literally used to build the target, I made sure to
exclude them from the model's inputs. Otherwise the model would just be adding numbers instead of
actually learning.

I trained three models - Linear Regression as a baseline, Random Forest, and Gradient Boosting -
and compared them using MAE, RMSE, and R² on a held-out test set. Gradient Boosting performed the
best, explaining about 98% of the variation in profit.

Finally, I built a Streamlit dashboard around this model with four extra tools: a Profit
Prediction page, a **What-If Analysis** tool where you can change commission rate or channel mix
and see how predicted profit changes, a Sensitivity Analysis showing how profit responds to one
factor at a time, and a simple grid-search **Optimization** tool that recommends a better channel
mix based on the model's own predictions.

Overall, this project shows how a fairly simple, well-checked machine learning pipeline can turn
raw operational data into a practical decision-support tool for a restaurant business."

---

## Viva Questions and Answers

**1. What is the objective of this project?**
To predict a restaurant's Total Monthly Net Profit from its operating data (orders, channel mix,
commission rate, cost rates, etc.), and to let a user explore how changing those inputs affects
predicted profit, using a What-If tool and a simple channel-mix optimizer.

**2. Why did you choose this dataset?**
It's a real, reasonably clean dataset with channel-level revenue, cost, and profit data for many
restaurants, which fits naturally into a multi-channel profit prediction problem - it has both
categorical business attributes (cuisine, segment, subregion) and numeric operational drivers.

**3. What is the target variable?**
`TotalNetProfit`, calculated by adding the four channel-level net profit columns already present
in the dataset: In-Store, Uber Eats, DoorDash, and Self-Delivery net profit.

**4. What is feature engineering?**
Creating new, more useful input columns from the raw data - for example, calculating Total
Revenue, Revenue per Order, Commission Cost, and channel revenue shares from the original columns,
so the model has more directly meaningful signals to learn from.

**5. Why did you use Linear Regression?**
As a simple, interpretable baseline model. It assumes a straight-line relationship between inputs
and profit, so comparing it against more flexible models tells us how much non-linearity actually
matters in this data.

**6. Why did you use Random Forest?**
It's a strong general-purpose model that can capture non-linear relationships and interactions
between features (like commission rate combined with revenue) without much tuning, and it's fairly
resistant to overfitting due to averaging many decision trees.

**7. Why did you use Gradient Boosting?**
As a second, often more accurate tree-based model, since it builds trees sequentially and
corrects the errors of previous trees. In this project it turned out to be the best-performing
model.

**8. What is MAE?**
Mean Absolute Error - the average of the absolute differences between predicted and actual profit
values. It's easy to interpret directly in dollars.

**9. What is RMSE?**
Root Mean Squared Error - similar to MAE, but it squares the errors before averaging and then
takes the square root, which penalizes larger errors more heavily than small ones.

**10. What is R²?**
R-squared (coefficient of determination) - it measures how much of the variation in the actual
profit values is explained by the model's predictions. An R² of 1.0 would mean perfect prediction;
our best model achieved about 0.98.

**11. What is data leakage?**
When information that wouldn't actually be available at prediction time (or that is directly
derived from the target) is accidentally used as a model input. In this project, the four
channel-level net-profit columns are used to calculate the target, so using them as inputs would
let the model "cheat" instead of learning genuine patterns - I identified and excluded them.

**12. What is What-If Analysis?**
A tool that lets a user start from a restaurant's current values, change one or more inputs (like
commission rate or channel share), and immediately see how the model's predicted profit changes,
along with the dollar difference and percentage change.

**13. What is channel mix?**
The proportion of a restaurant's business that comes through each channel - In-Store, Uber Eats,
DoorDash, and Self-Delivery - usually expressed as percentages that add up to 100%.

**14. What is optimization uplift?**
The percentage improvement in predicted profit when moving from the current channel mix to the
best channel mix found during the grid search: `((Optimized Profit - Current Profit) / Current Profit) x 100`.

**15. What are the limitations?**
The data is a single monthly snapshot per restaurant, not a time series, so trends and seasonality
can't be modeled. EDA relationships are correlational, not causal. The optimizer only searches a
small, user-chosen grid of channel-mix combinations rather than every possible combination, and it
doesn't account for real-world constraints like kitchen capacity or staffing.

**16. What is the future scope?**
Collecting multi-month data to model trends over time, adding more cost drivers like staffing and
rent, using more advanced optimization methods for a finer channel-mix search, and testing the
model on restaurant data from other cities to check how well it generalizes.
