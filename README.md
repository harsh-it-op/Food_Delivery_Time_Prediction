# Food Delivery Time Prediction

End-to-end ML project built from the spec in
`Copy of Food Delivery Time Prediction.docx` and the dataset
`Food_Delivery_Time_Prediction.csv`.

## How to run

```bash
cd [https://github.com/harsh-it-op/Food_Delivery_Time_Prediction/edit/main/README.md]
python analysis.py
python build_notebook.py    # rebuilds the .ipynb from the same logic
```

Both scripts depend on `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

## Layout

| Path | Purpose |
|---|---|
| `analysis.py` | Single-file pipeline (EDA -> models -> report -> figures). |
| `Food_Delivery_Time_Prediction.ipynb` | Notebook version of the same logic. |
| `build_notebook.py` | Regenerates the `.ipynb`. |
| `figures/01..12_*.png` | All 12 visualizations. |
| `figures/model_metrics.json` | Numeric metrics (LR + LogReg). |
| `Final_Report.md` | Written report per the assignment's deliverable list. |

## Headline numbers

- **Linear Regression** — MAE 26.39 min, RMSE 30.54, R2 -0.008 (baseline predict-mean R2 -0.001).
- **Logistic Regression** — Accuracy 0.45, F1 0.48, ROC AUC 0.435.

The dataset contains very little signal (max |r| with target = 0.171).
Both pipelines are implemented correctly but cannot beat a trivial baseline
on this data; the report documents why and recommends the operational
improvements + data-collection backlog needed to make a production model.
