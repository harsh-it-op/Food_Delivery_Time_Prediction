# Food Delivery Time Prediction -- Final Report

End-to-end ML project: from raw `Food_Delivery_Time_Prediction.csv` (200 orders, 15 raw columns) to two pipelines for predicting **delivery time** and classifying **Fast vs Delayed** orders.

## 1. Dataset Overview

- **Rows:** 200  
- **Raw columns:** 15  
- **Columns after feature engineering:** 22  
- **No missing values** in any column.
- Numerical: `Distance`, `Delivery_Person_Experience`, `Restaurant_Rating`, `Customer_Rating`, `Order_Cost`, `Tip_Amount`, `Delivery_Time` (target).
- Categorical: `Weather_Conditions`, `Traffic_Conditions`, `Order_Priority`, `Order_Time`, `Vehicle_Type`.
- **Key observation:** every feature has near-zero correlation with `Delivery_Time` (max |r| = 0.171). The dataset behaves like a near-uniform random sample -- see Section 6.

## 2. Feature Engineering

- Parsed `(lat, lon)` strings into numeric `cust_lat/lon`, `rest_lat/lon`.
- Computed **Haversine_km** between customer and restaurant.
- Added **Is_Rush_Hour** (1 for Evening / Lunch / Dinner, 0 otherwise).
- Bucketed `Delivery_Person_Experience` into **Junior / Mid / Senior**.
- Standardized numeric features and one-hot encoded categorical features before both models.

## 3. EDA Highlights

- `Delivery_Time` is roughly uniform between ~20 and ~120 minutes (mean 70.5, std 29.8).
- Group means by category all fall within a 4-minute band (e.g. Rainy 73.1 vs Cloudy 69.8; Bicycle 74.3 vs Bike 66.7).
- Boxplots revealed mild outliers in `Distance` and `Order_Cost` -- retained because they reflect real operational variance.
- See `figures/01_..05_*.png` for the full set of plots.

## 4. Linear Regression -- Continuous Prediction

Pipeline: `StandardScaler(numeric) + OneHotEncoder(categorical) -> LinearRegression`.

| Metric | Model | Baseline (predict mean) |
|---|---|---|
| **MAE** | 26.39 | 25.53 |
| **R2**  | -0.008 | -0.001 |
| RMSE | 30.54 | -- |
| MSE | 932.61 | -- |

## 5. Logistic Regression -- Fast vs Delayed

- Threshold: **Delivery_Time > 72.78 min** => Delayed.

- **Accuracy:** 0.450
- **Precision:** 0.455
- **Recall:** 0.500
- **F1-score:** 0.476
- **ROC AUC:** 0.435

## 6. Honest Evaluation of Predictive Power

The dataset supplied for this assignment contains **very little signal**:

- The strongest single-feature correlation with `Delivery_Time` is **0.171** (e.g. `Distance` r = -0.075), and most features are within +/- 0.05 of zero.
- Linear regression produces an R2 of **-0.008** -- essentially equal to the constant-mean baseline (R2 = -0.001).
- Logistic regression reaches ~0.5 ROC AUC, which is the same as random guessing on a balanced binary problem.

**Interpretation:** the assignment asks for linear + logistic regression as a teaching exercise. We delivered both pipelines end-to-end, but with this dataset neither model can beat a trivial baseline. To make a production-worthy ETA predictor we would need: (a) more orders (>> 200), (b) real-time features like kitchen prep time, rider GPS, and order-load at the moment of dispatch, and (c) features that actually move with delivery time (currently none of the supplied ones do).

## 7. Operational Recommendations

Even with weak predictive models, the EDA surfaces patterns that are useful from an operations standpoint:

1. **Vehicle mix** -- Bicycle 74.3 min, Car 70.4 min, Bike 66.7 min. Bicycles are the slowest on average; reserve Cars for distance > 8 km.
2. **Weather SOPs** -- Rainy 73.1 min, Cloudy 69.8 min, Sunny 69.5 min, Snowy 69.2 min. Rainy / Snowy shifts need more riders and a Car bias.
3. **Traffic pattern** -- Low 71.9 min, Medium 71.0 min, High 67.6 min. Treat Low vs High as operationally equivalent in this sample until the dataset grows.
4. **Experience cross-training** -- pair Junior riders with Senior riders during peak hours.
5. **Data-collection backlog** -- the missing live signals (prep time, rider GPS, dispatch load) are the real levers. Instrument them before rebuilding the model.

## 8. Deliverables

- `analysis.py` -- full reproducible pipeline.
- `Food_Delivery_Time_Prediction.ipynb` -- notebook export of the same logic.
- `figures/` -- 12 PNG visualizations (distributions, scatter, ROC, confusion matrix, residuals, comparison, coefficients).
- `figures/model_metrics.json` -- all numeric metrics in one place.
- `Final_Report.md` -- this document.
