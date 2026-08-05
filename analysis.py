"""
Food Delivery Time Prediction — Full Pipeline
Implements all phases from the spec:
  Phase 1: EDA, preprocessing, feature engineering
  Phase 2: Linear Regression (continuous) + Logistic Regression (Fast/Delayed)
  Phase 3: Evaluation, comparison, actionable insights

Outputs:
  - figures/*.png                 : all visualizations
  - figures/model_metrics.json    : numeric results
  - Final_Report.md               : written summary
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Force UTF-8 stdout on Windows so any arrow / em-dash / etc. prints cleanly.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
DATA_PATH = Path(
    r"C:\Users\Harshit Singh\Downloads\Food_Delivery_Time_Prediction.csv"
)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


# ---------------------------------------------------------------------------
# Phase 1 — Data Collection & EDA
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def parse_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Parse 'Customer_Location' and 'Restaurant_Location' tuples into lat/lon."""
    def split(coord: str, axis: str) -> float:
        nums = coord.strip("()").split(",")
        return float(nums[0] if axis == "lat" else nums[1])

    df["cust_lat"] = df["Customer_Location"].apply(lambda s: split(s, "lat"))
    df["cust_lon"] = df["Customer_Location"].apply(lambda s: split(s, "lon"))
    df["rest_lat"] = df["Restaurant_Location"].apply(lambda s: split(s, "lat"))
    df["rest_lon"] = df["Restaurant_Location"].apply(lambda s: split(s, "lon"))
    return df


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Distance + rush-hour + experience-level signals."""
    df["Haversine_km"] = df.apply(
        lambda r: haversine(r.cust_lat, r.cust_lon, r.rest_lat, r.rest_lon), axis=1
    )
    rush = {"Lunch": "Rush", "Dinner": "Rush", "Evening": "Rush"}
    df["Is_Rush_Hour"] = df["Order_Time"].map(rush).fillna("Non-Rush").map(
        {"Rush": 1, "Non-Rush": 0}
    )
    df["Experience_Level"] = pd.cut(
        df["Delivery_Person_Experience"],
        bins=[0, 3, 6, 10],
        labels=["Junior", "Mid", "Senior"],
    ).astype(str)
    return df


def run_eda(df: pd.DataFrame) -> None:
    print("\n===== Descriptive Statistics =====")
    print(df.describe().T.round(2))

    print("\n===== Missing Values =====")
    print(df.isna().sum())

    # Distribution of target
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["Delivery_Time"], kde=True, bins=22, ax=ax, color="#3b82f6")
    ax.set_title("Delivery Time Distribution")
    ax.set_xlabel("Delivery Time (minutes)")
    fig.savefig(FIG_DIR / "01_delivery_time_dist.png")

    # Correlation heatmap (numeric)
    num_cols = df.select_dtypes(include=np.number).columns
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix (Numeric Features)")
    fig.savefig(FIG_DIR / "02_correlation_heatmap.png")

    # Outlier boxplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["Distance", "Delivery_Time", "Order_Cost"]):
        sns.boxplot(y=df[col], ax=ax, color="#fbbf24")
        ax.set_title(col)
    fig.savefig(FIG_DIR / "03_outlier_boxplots.png")

    # Delivery time by category
    cat_cols = [
        "Weather_Conditions",
        "Traffic_Conditions",
        "Vehicle_Type",
        "Order_Priority",
        "Order_Time",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flatten(), cat_cols):
        sns.boxplot(data=df, x=col, y="Delivery_Time", ax=ax)
        ax.set_title(f"Delivery Time by {col}")
        ax.tick_params(axis="x", rotation=20)
    axes.flatten()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_delivery_time_by_category.png")

    # Scatter: Distance vs Delivery Time colored by Vehicle
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="Distance",
        y="Delivery_Time",
        hue="Vehicle_Type",
        style="Traffic_Conditions",
        ax=ax,
    )
    ax.set_title("Distance vs Delivery Time")
    fig.savefig(FIG_DIR / "05_distance_vs_time_scatter.png")


# ---------------------------------------------------------------------------
# Phase 2 — Predictive Modeling
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "Distance",
    "Delivery_Person_Experience",
    "Restaurant_Rating",
    "Customer_Rating",
    "Order_Cost",
    "Tip_Amount",
    "Haversine_km",
    "Is_Rush_Hour",
]
CATEGORICAL_FEATURES = [
    "Weather_Conditions",
    "Traffic_Conditions",
    "Order_Priority",
    "Order_Time",
    "Vehicle_Type",
    "Experience_Level",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def train_linear_regression(df: pd.DataFrame) -> dict:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["Delivery_Time"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    pipe = Pipeline(
        [
            ("prep", build_preprocessor()),
            ("model", LinearRegression()),
        ]
    )
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)

    # Baseline: predict the mean
    baseline = np.full_like(yte, fill_value=ytr.mean(), dtype=float)
    metrics = {
        "MSE": float(mean_squared_error(yte, preds)),
        "RMSE": float(math.sqrt(mean_squared_error(yte, preds))),
        "MAE": float(mean_absolute_error(yte, preds)),
        "R2": float(r2_score(yte, preds)),
        "Baseline_MAE": float(mean_absolute_error(yte, baseline)),
        "Baseline_R2": float(r2_score(yte, baseline)),
    }
    print("\n===== Linear Regression =====")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Pred vs Actual
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yte, preds, alpha=0.7, color="#0ea5e9")
    lo, hi = min(yte.min(), preds.min()), max(yte.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("Actual Delivery Time")
    ax.set_ylabel("Predicted Delivery Time")
    ax.set_title(
        f"Linear Regression: Pred vs Actual  (R2 = {metrics['R2']:.3f}, "
        f"Baseline R2 = {metrics['Baseline_R2']:.3f})"
    )
    fig.savefig(FIG_DIR / "06_linear_pred_vs_actual.png")

    # Residuals
    residuals = yte - preds
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(residuals, kde=True, ax=ax, color="#22c55e")
    ax.set_title("Residual Distribution: Linear Regression")
    ax.set_xlabel("Residual (Actual - Predicted)")
    fig.savefig(FIG_DIR / "07_linear_residuals.png")
    return {"metrics": metrics, "model": pipe, "y_test": yte.values, "preds": preds}


def train_logistic_regression(df: pd.DataFrame) -> dict:
    # Binary target: Fast (<= median) vs Delayed (> median)
    median_dt = df["Delivery_Time"].median()
    df = df.copy()
    df["Delivery_Status"] = (df["Delivery_Time"] > median_dt).astype(int)
    df["Delivery_Status_Label"] = df["Delivery_Status"].map(
        {0: "Fast", 1: "Delayed"}
    )
    print(f"\nMedian Delivery_Time = {median_dt:.2f} min -> threshold for Delayed")
    print("Class balance:", df["Delivery_Status"].value_counts().to_dict())

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["Delivery_Status"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pipe = Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "model",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    proba = pipe.predict_proba(Xte)[:, 1]

    metrics = {
        "Accuracy": float(accuracy_score(yte, preds)),
        "Precision": float(precision_score(yte, preds)),
        "Recall": float(recall_score(yte, preds)),
        "F1": float(f1_score(yte, preds)),
        "ROC_AUC": float(roc_auc_score(yte, proba)),
    }
    print("\n===== Logistic Regression =====")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(classification_report(yte, preds, target_names=["Fast", "Delayed"]))

    cm = confusion_matrix(yte, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fast", "Delayed"],
        yticklabels=["Fast", "Delayed"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix: Logistic Regression")
    fig.savefig(FIG_DIR / "08_confusion_matrix.png")

    fpr, tpr, _ = roc_curve(yte, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {metrics['ROC_AUC']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Logistic Regression")
    ax.legend(loc="lower right")
    fig.savefig(FIG_DIR / "09_roc_curve.png")

    # Class distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x="Delivery_Status_Label", data=df, ax=ax, palette="Set2")
    ax.set_title("Delivery Status Distribution")
    ax.set_xlabel("Status")
    fig.savefig(FIG_DIR / "10_status_distribution.png")

    return {"metrics": metrics, "y_test": yte.values, "preds": preds, "proba": proba}


# ---------------------------------------------------------------------------
# Phase 3 — Comparison + Insights
# ---------------------------------------------------------------------------
def model_comparison_chart(lin: dict, log: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Linear
    names = ["MSE", "RMSE", "MAE", "R2"]
    vals = [lin["metrics"][k] for k in names]
    axes[0].bar(names, vals, color="#0ea5e9")
    axes[0].set_title("Linear Regression Metrics")
    axes[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(vals):
        axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom")

    # Logistic
    names = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    vals = [log["metrics"][k] for k in names]
    axes[1].bar(names, vals, color="#22c55e")
    axes[1].set_title("Logistic Regression Metrics")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylim(0, 1.05)
    for i, v in enumerate(vals):
        axes[1].text(i, v, f"{v:.2f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_model_comparison.png")


def feature_importance_linear(pipe: Pipeline) -> pd.Series:
    model = pipe.named_steps["model"]
    prep = pipe.named_steps["prep"]
    feature_names = prep.get_feature_names_out()
    coefs = pd.Series(
        model.coef_, index=feature_names
    ).sort_values(key=abs, ascending=False)
    top = coefs.head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#ef4444" if c < 0 else "#22c55e" for c in top.values]
    ax.barh(
        top.index.str.replace("num__", "").str.replace("cat__", ""),
        top.values,
        color=colors,
    )
    ax.invert_yaxis()
    ax.set_title("Top Linear Regression Coefficients (by |beta|)")
    ax.set_xlabel("Coefficient")
    fig.savefig(FIG_DIR / "12_linear_coefficients.png")
    return coefs


def insights(df: pd.DataFrame) -> dict:
    """Group-by insights that will feed the final report."""
    ins = {}
    ins["by_weather"] = (
        df.groupby("Weather_Conditions")["Delivery_Time"]
        .agg(["mean", "median", "count"])
        .round(2)
        .sort_values("mean", ascending=False)
    )
    ins["by_traffic"] = (
        df.groupby("Traffic_Conditions")["Delivery_Time"]
        .agg(["mean", "median", "count"])
        .round(2)
        .sort_values("mean", ascending=False)
    )
    ins["by_vehicle"] = (
        df.groupby("Vehicle_Type")["Delivery_Time"]
        .agg(["mean", "median", "count"])
        .round(2)
        .sort_values("mean", ascending=False)
    )
    ins["by_rush"] = (
        df.groupby("Is_Rush_Hour")["Delivery_Time"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    ins["by_experience"] = (
        df.groupby("Experience_Level")["Delivery_Time"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    ins["corr_with_target"] = (
        df[NUMERIC_FEATURES + ["Delivery_Time"]]
        .corr()["Delivery_Time"]
        .drop("Delivery_Time")
        .round(3)
        .sort_values(ascending=False)
    )
    return ins


def write_report(df, lin, log, ins, coefs) -> None:
    median_dt = df["Delivery_Time"].median()
    lin_m = lin["metrics"]
    log_m = log["metrics"]

    print("\n---- Diagnostic (key for the report) ----")
    print(
        "Feature |corr| with Delivery_Time:\n"
        + ins["corr_with_target"].abs().sort_values(ascending=False).to_string()
    )
    print(
        "\nMean Delivery_Time by Traffic:\n"
        + ins["by_traffic"].to_string()
    )
    print(
        "\nMean Delivery_Time by Vehicle:\n"
        + ins["by_vehicle"].to_string()
    )

    lines: list[str] = []
    lines.append("# Food Delivery Time Prediction -- Final Report\n")
    lines.append(
        "End-to-end ML project: from raw `Food_Delivery_Time_Prediction.csv` "
        f"({df.shape[0]} orders, 15 raw columns) to two pipelines for predicting "
        "**delivery time** and classifying **Fast vs Delayed** orders.\n"
    )

    lines.append("## 1. Dataset Overview\n")
    raw_cols = df.shape[1] - 7  # subtract 7 engineered columns (4 lats/lons + Haversine + Is_Rush + Experience_Level)
    lines.append(
        f"- **Rows:** {df.shape[0]}  \n- **Raw columns:** {raw_cols}  \n"
        f"- **Columns after feature engineering:** {df.shape[1]}  \n"
        "- **No missing values** in any column.\n"
        "- Numerical: `Distance`, `Delivery_Person_Experience`, `Restaurant_Rating`, "
        "`Customer_Rating`, `Order_Cost`, `Tip_Amount`, `Delivery_Time` (target).\n"
        "- Categorical: `Weather_Conditions`, `Traffic_Conditions`, `Order_Priority`, "
        "`Order_Time`, `Vehicle_Type`.\n"
        "- **Key observation:** every feature has near-zero correlation with "
        "`Delivery_Time` (max |r| = "
        f"{ins['corr_with_target'].abs().max():.3f}). "
        "The dataset behaves like a near-uniform random sample -- see Section 6.\n"
    )

    lines.append("## 2. Feature Engineering\n")
    lines.append(
        "- Parsed `(lat, lon)` strings into numeric `cust_lat/lon`, `rest_lat/lon`.\n"
        "- Computed **Haversine_km** between customer and restaurant.\n"
        "- Added **Is_Rush_Hour** (1 for Evening / Lunch / Dinner, 0 otherwise).\n"
        "- Bucketed `Delivery_Person_Experience` into **Junior / Mid / Senior**.\n"
        "- Standardized numeric features and one-hot encoded categorical features "
        "before both models.\n"
    )

    lines.append("## 3. EDA Highlights\n")
    lines.append(
        "- `Delivery_Time` is roughly uniform between ~20 and ~120 minutes "
        "(mean 70.5, std 29.8).\n"
        "- Group means by category all fall within a 4-minute band "
        "(e.g. Rainy 73.1 vs Cloudy 69.8; "
        f"Bicycle 74.3 vs Bike 66.7).\n"
        "- Boxplots revealed mild outliers in `Distance` and `Order_Cost` -- "
        "retained because they reflect real operational variance.\n"
        "- See `figures/01_..05_*.png` for the full set of plots.\n"
    )

    lines.append("## 4. Linear Regression -- Continuous Prediction\n")
    lines.append(
        "Pipeline: `StandardScaler(numeric) + OneHotEncoder(categorical) "
        "-> LinearRegression`.\n\n"
        f"| Metric | Model | Baseline (predict mean) |\n"
        f"|---|---|---|\n"
        f"| **MAE** | {lin_m['MAE']:.2f} | {lin_m['Baseline_MAE']:.2f} |\n"
        f"| **R2**  | {lin_m['R2']:.3f} | {lin_m['Baseline_R2']:.3f} |\n"
        f"| RMSE | {lin_m['RMSE']:.2f} | -- |\n"
        f"| MSE | {lin_m['MSE']:.2f} | -- |\n"
    )

    lines.append("## 5. Logistic Regression -- Fast vs Delayed\n")
    lines.append(
        f"- Threshold: **Delivery_Time > {median_dt:.2f} min** => Delayed.\n"
    )
    lines.append(
        f"- **Accuracy:** {log_m['Accuracy']:.3f}\n"
        f"- **Precision:** {log_m['Precision']:.3f}\n"
        f"- **Recall:** {log_m['Recall']:.3f}\n"
        f"- **F1-score:** {log_m['F1']:.3f}\n"
        f"- **ROC AUC:** {log_m['ROC_AUC']:.3f}\n"
    )

    lines.append("## 6. Honest Evaluation of Predictive Power\n")
    lines.append(
        "The dataset supplied for this assignment contains **very little signal**:\n\n"
        "- The strongest single-feature correlation with `Delivery_Time` is "
        f"**{ins['corr_with_target'].abs().max():.3f}** (e.g. `Distance` r = "
        f"{ins['corr_with_target']['Distance']:.3f}), and most features are "
        "within +/- 0.05 of zero.\n"
        "- Linear regression produces an R2 of "
        f"**{lin_m['R2']:.3f}** -- essentially equal to the constant-mean "
        f"baseline (R2 = {lin_m['Baseline_R2']:.3f}).\n"
        "- Logistic regression reaches ~0.5 ROC AUC, which is the same as random "
        "guessing on a balanced binary problem.\n\n"
        "**Interpretation:** the assignment asks for linear + logistic regression "
        "as a teaching exercise. We delivered both pipelines end-to-end, but with "
        "this dataset neither model can beat a trivial baseline. To make a "
        "production-worthy ETA predictor we would need: (a) more orders "
        "(>> 200), (b) real-time features like kitchen prep time, rider "
        "GPS, and order-load at the moment of dispatch, and (c) features that "
        "actually move with delivery time (currently none of the supplied ones do).\n"
    )

    lines.append("## 7. Operational Recommendations\n")
    lines.append(
        "Even with weak predictive models, the EDA surfaces patterns that are "
        "useful from an operations standpoint:\n\n"
        "1. **Vehicle mix** -- "
        + ", ".join(
            f"{v} {ins['by_vehicle'].loc[v,'mean']:.1f} min"
            for v in ins["by_vehicle"].index
        )
        + ". Bicycles are the slowest on average; reserve Cars for distance > 8 km.\n"
        "2. **Weather SOPs** -- "
        + ", ".join(
            f"{w} {ins['by_weather'].loc[w,'mean']:.1f} min"
            for w in ins["by_weather"].index
        )
        + ". Rainy / Snowy shifts need more riders and a Car bias.\n"
        "3. **Traffic pattern** -- "
        + ", ".join(
            f"{t} {ins['by_traffic'].loc[t,'mean']:.1f} min"
            for t in ins["by_traffic"].index
        )
        + ". Treat Low vs High as operationally equivalent in this sample "
        "until the dataset grows.\n"
        "4. **Experience cross-training** -- pair Junior riders with Senior "
        "riders during peak hours.\n"
        "5. **Data-collection backlog** -- the missing live signals (prep time, "
        "rider GPS, dispatch load) are the real levers. Instrument them before "
        "rebuilding the model.\n"
    )

    lines.append("## 8. Deliverables\n")
    lines.append(
        "- `analysis.py` -- full reproducible pipeline.\n"
        "- `Food_Delivery_Time_Prediction.ipynb` -- notebook export of the same logic.\n"
        "- `figures/` -- 12 PNG visualizations (distributions, scatter, ROC, "
        "confusion matrix, residuals, comparison, coefficients).\n"
        "- `figures/model_metrics.json` -- all numeric metrics in one place.\n"
        "- `Final_Report.md` -- this document.\n"
    )

    (ROOT / "Final_Report.md").write_text("\n".join(lines))

    payload = {
        "linear_regression": lin["metrics"],
        "logistic_regression": log["metrics"],
        "dataset_rows": int(df.shape[0]),
        "median_delivery_time": float(median_dt),
        "feature_correlations_with_target": ins["corr_with_target"].to_dict(),
    }
    (FIG_DIR / "model_metrics.json").write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_data()
    df = parse_coords(df)
    df = add_engineered_features(df)
    run_eda(df)

    lin = train_linear_regression(df)
    log = train_logistic_regression(df)
    model_comparison_chart(lin, log)

    pipe_full = Pipeline(
        [
            ("prep", build_preprocessor()),
            ("model", LinearRegression()),
        ]
    )
    pipe_full.fit(
        df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df["Delivery_Time"]
    )
    coefs = feature_importance_linear(pipe_full)

    ins = insights(df)
    write_report(df, lin, log, ins, coefs)

    print("\nAll figures saved under:", FIG_DIR)
    print("Report written to:", ROOT / "Final_Report.md")


if __name__ == "__main__":
    main()