from pathlib import Path
import numpy as np
import pandas as pd

TARGET = "sales"
FEATURES = ["month_num", "year", "time_idx", "lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "rolling_mean_3", "rolling_mean_6"]

def load_data(path="data/raw/train.csv"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Download train.csv from Kaggle and place it in data/raw/train.csv.")
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "store", "item", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["date", "store", "item", "sales"]).copy()
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    return df.dropna(subset=["sales"])

def monthly_series(df, store, item):
    selected = df[(df["store"] == int(store)) & (df["item"] == int(item))].copy()
    if selected.empty:
        raise ValueError("No records found for the selected store and item.")
    selected["month"] = selected["date"].dt.to_period("M").dt.to_timestamp()
    monthly = selected.groupby("month", as_index=False)["sales"].sum().sort_values("month")
    months = pd.date_range(monthly["month"].min(), monthly["month"].max(), freq="MS")
    monthly = monthly.set_index("month").reindex(months, fill_value=0)
    monthly.index.name = "month"
    return monthly.reset_index().rename(columns={"index": "month"}).astype({"sales": float})

def make_features(series):
    data = series.copy().sort_values("month").reset_index(drop=True)
    data["month_num"] = data["month"].dt.month
    data["year"] = data["month"].dt.year
    data["time_idx"] = np.arange(len(data))
    for lag in [1, 2, 3, 6, 12]:
        data[f"lag_{lag}"] = data["sales"].shift(lag)
    data["rolling_mean_3"] = data["sales"].shift(1).rolling(3).mean()
    data["rolling_mean_6"] = data["sales"].shift(1).rolling(6).mean()
    return data.dropna().reset_index(drop=True)

def forecast_recursive(history, model, horizon=12):
    history = history.copy().sort_values("month").reset_index(drop=True)
    results = []
    for _ in range(horizon):
        next_month = history["month"].iloc[-1] + pd.offsets.MonthBegin(1)
        values = history["sales"].astype(float).tolist()
        def lag(n):
            return values[-n] if len(values) >= n else float(np.mean(values))
        row = {
            "month_num": next_month.month, "year": next_month.year, "time_idx": len(history),
            "lag_1": lag(1), "lag_2": lag(2), "lag_3": lag(3), "lag_6": lag(6), "lag_12": lag(12),
            "rolling_mean_3": float(np.mean(values[-3:])), "rolling_mean_6": float(np.mean(values[-6:])),
        }
        prediction = max(0.0, float(model.predict(pd.DataFrame([row])[FEATURES])[0]))
        results.append({"month": next_month, "sales": prediction})
        history = pd.concat([history, pd.DataFrame([{"month": next_month, "sales": prediction}])], ignore_index=True)
    return pd.DataFrame(results)
