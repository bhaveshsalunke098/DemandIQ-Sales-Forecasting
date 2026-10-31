# Sales Demand Forecasting

An end-to-end monthly sales demand forecasting project using Python, Pandas, Scikit-learn, and Streamlit. It aggregates daily retail sales into monthly demand, evaluates a machine-learning model using a chronological holdout, and provides an interactive dashboard for forecasting future monthly unit sales.

## Objectives
- Aggregate daily sales into monthly store-item series.
- Train a lag-feature Random Forest regression model.
- Evaluate forecasts using MAE and RMSE, with MAPE calculated only for non-zero actual values.
- Forecast up to 12 future months and download results from Streamlit.

## Dataset
Use the [Store Item Demand Forecasting Challenge dataset on Kaggle](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data). Download `train.csv` and place it at `data/raw/train.csv`. The dataset is not included in this repository.

Expected columns: `date`, `store`, `item`, `sales`.

## Setup
```bash
python -m venv .venv
```
Windows PowerShell:
```powershell
.\\.venv\\Scripts\\Activate.ps1
```
macOS/Linux:
```bash
source .venv/bin/activate
```
Install and run:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

## Methodology
Daily sales are summed into monthly totals for each store-item pair. Features include calendar month, year, time index, lagged sales, and trailing rolling means. Evaluation uses a chronological holdout rather than a random split. The starter version trains on one store-item series, then refits on its available history for forecasting.

## Evaluation
Training writes MAE and RMSE to `models/model_info.json` and `reports/evaluation_metrics.csv`; test predictions are saved to `reports/test_predictions.csv`. MAPE is reported only for non-zero actual values.

## Limitations
- The starter model is trained on one store-item series; broader validation is needed before business use.
- Forecasts may miss promotions, holidays, stockouts, pricing changes, and other external factors.
- Predictions are estimates, not guaranteed future demand.

## Future Improvements
- Compare with seasonal-naive and other forecasting models.
- Evaluate multiple store-item series and rolling-origin backtests.
- Add holiday/promotion features, forecast intervals, and monitoring.
- Deploy the app.

## Author
**Bhavesh Salunke**
- GitHub: https://github.com/bhaveshsalunke098
- LinkedIn: https://linkedin.com/in/bhavesh-salunke-7a4163283
- Portfolio: https://techbhavesh.netlify.app
