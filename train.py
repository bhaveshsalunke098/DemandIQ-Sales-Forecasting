from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.forecasting import load_data, monthly_series, make_features, FEATURES

def main():
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    df = load_data()
    store, item = int(df["store"].min()), int(df["item"].min())
    series = monthly_series(df, store, item)
    featured = make_features(series)
    if len(featured) < 24:
        raise ValueError("Not enough monthly observations after lag creation.")
    test_n = min(12, max(3, len(featured) // 5))
    train, test = featured.iloc[:-test_n], featured.iloc[-test_n:]
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train["sales"])
    pred = np.maximum(0, model.predict(test[FEATURES]))
    actual = test["sales"].to_numpy()
    mae = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred) ** 0.5
    nonzero = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else None
    model.fit(featured[FEATURES], featured["sales"])
    joblib.dump(model, "models/sales_forecast_model.joblib")
    metrics = {"store": store, "item": item, "test_months": int(test_n), "MAE": float(mae), "RMSE": float(rmse), "MAPE_percent_nonzero_actuals": mape}
    Path("models/model_info.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"month": test["month"].values, "actual_sales": actual, "predicted_sales": pred}).to_csv("reports/test_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv("reports/evaluation_metrics.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print("Saved model to models/sales_forecast_model.joblib")

if __name__ == "__main__":
    main()
