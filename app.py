
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import streamlit as st

from src.forecasting import load_data, monthly_series, forecast_recursive


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="DemandIQ | Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM UI STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #F5F7FB;
        --card: #FFFFFF;
        --text: #182230;
        --muted: #738096;
        --accent: #635BFF;
        --border: #E5EAF2;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--bg);
    }

    [data-testid="stHeader"] {
        background: rgba(245, 247, 251, 0.92);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #151C2C;
        border-right: 1px solid #252D40;
    }

    [data-testid="stSidebar"] * {
        color: #E8ECF5;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #AEB9CD;
    }

    [data-testid="stSidebar"] label {
        color: #E8ECF5 !important;
        font-weight: 600;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="slider"] {
        border-radius: 10px;
    }

    .brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 27px;
        font-weight: 700;
        letter-spacing: -1px;
        color: #FFFFFF;
        margin-bottom: 2px;
    }

    .brand span {
        color: #928BFF;
    }

    .sidebar-caption {
        color: #AEB9CD;
        font-size: 12px;
        margin-bottom: 30px;
    }

    .sidebar-section {
        color: #928BFF;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 24px;
        margin-bottom: 12px;
    }

    .sidebar-note {
        background: #202A3D;
        border: 1px solid #303B51;
        border-radius: 12px;
        padding: 13px;
        margin-top: 25px;
        color: #C5CEE0;
        font-size: 12px;
        line-height: 1.7;
    }

    /* Hero */
    .eyebrow {
        color: #635BFF;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.7px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(28px, 3vw, 40px);
        font-weight: 700;
        line-height: 1.15;
        letter-spacing: -1.5px;
        color: #182230;
        margin: 0;
    }

    .hero-description {
        color: #738096;
        font-size: 14px;
        margin-top: 10px;
        margin-bottom: 18px;
        line-height: 1.7;
    }

    .status-pill {
        display: inline-block;
        padding: 7px 12px;
        background: #E8F8F0;
        color: #13845A;
        border: 1px solid #C9EEDB;
        border-radius: 30px;
        font-size: 11px;
        font-weight: 700;
        white-space: nowrap;
    }

    .section-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 20px;
        font-weight: 700;
        color: #182230;
        letter-spacing: -0.5px;
        margin-top: 10px;
        margin-bottom: 3px;
    }

    .section-caption {
        font-size: 12px;
        color: #8490A4;
        margin-bottom: 16px;
    }

    /* KPI cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 16px;
        padding: 19px 20px;
        min-height: 132px;
        box-shadow: 0 3px 12px rgba(24, 34, 48, 0.025);
    }

    .kpi-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 17px;
        gap: 8px;
    }

    .kpi-label {
        color: #738096;
        font-size: 12px;
        font-weight: 600;
    }

    .kpi-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: #F0EEFF;
        color: #635BFF;
        font-size: 17px;
    }

    .kpi-value {
        font-family: 'Space Grotesk', sans-serif;
        color: #182230;
        font-size: 27px;
        font-weight: 700;
        letter-spacing: -0.8px;
        line-height: 1.2;
    }

    .kpi-foot {
        color: #8A95A7;
        font-size: 11px;
        margin-top: 7px;
    }

    /* Panels */
    .panel {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 3px 12px rgba(24, 34, 48, 0.025);
    }

    .panel-title {
        font-family: 'Space Grotesk', sans-serif;
        color: #182230;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .panel-subtitle {
        color: #8490A4;
        font-size: 12px;
        margin-bottom: 12px;
    }

    .forecast-highlight {
        background: linear-gradient(120deg, #F0EEFF, #F7F5FF);
        border: 1px solid #E1DDFF;
        border-radius: 14px;
        padding: 16px 18px;
        margin: 10px 0 18px 0;
    }

    .forecast-highlight-label {
        color: #6E67B9;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .forecast-highlight-value {
        font-family: 'Space Grotesk', sans-serif;
        color: #4036B8;
        font-size: 28px;
        font-weight: 700;
        margin-top: 5px;
    }

    .forecast-highlight-note {
        color: #7770AF;
        font-size: 11px;
        margin-top: 4px;
    }

    /* Buttons */
    .stDownloadButton button {
        background: #635BFF;
        color: white;
        border: 1px solid #635BFF;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 16px;
        transition: all 0.2s ease;
    }

    .stDownloadButton button:hover {
        background: #5148E5;
        color: white;
        border-color: #5148E5;
    }

    /* Data table and charts */
    [data-testid="stDataFrame"] {
        border: 1px solid #E5EAF2;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stPlotlyChart"],
    [data-testid="stImage"] {
        border-radius: 12px;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #E5EAF2;
    }

    .footer {
        text-align: center;
        color: #929CAF;
        font-size: 11px;
        padding-top: 25px;
        line-height: 1.8;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
        }

        .hero-title {
            font-size: 29px;
        }

        .kpi-card {
            padding: 14px;
        }

        .kpi-value {
            font-size: 23px;
        }
    }
    

    /* ===== GLOBAL TEXT CONTRAST FIX ===== */
    :root {
        color-scheme: light;
        --bg: #F4F6FA;
        --card: #FFFFFF;
        --text: #172033;
        --muted: #596579;
        --accent: #5145E5;
        --border: #DCE2EC;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: #F4F6FA !important;
        color: #172033 !important;
    }

    [data-testid="stMain"],
    [data-testid="stMain"] .block-container {
        color: #172033 !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #374151;
        line-height: 1.65;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #596579 !important;
    }

    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4 {
        color: #172033 !important;
    }

    [data-testid="stMain"] input,
    [data-testid="stMain"] textarea,
    [data-testid="stMain"] [data-baseweb="select"] > div {
        color: #172033 !important;
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [data-baseweb="select"] [role="listbox"] {
        background-color: #FFFFFF !important;
    }

    [data-baseweb="menu"] *,
    [data-baseweb="select"] [role="option"] {
        color: #172033 !important;
    }

    [data-baseweb="select"] [aria-selected="true"] {
        background-color: #EAE8FF !important;
    }

    [data-testid="stDataFrame"] {
        background: #FFFFFF !important;
        border: 1px solid #DCE2EC !important;
        border-radius: 12px;
    }

    [data-testid="stAlert"] {
        color: #172033 !important;
    }

    .kpi-label { color: #596579 !important; }
    .kpi-value { color: #172033 !important; }
    .kpi-foot { color: #596579 !important; }
    .forecast-highlight-label { color: #4036B8 !important; }
    .forecast-highlight-value { color: #312783 !important; }
    .forecast-highlight-note { color: #5145A8 !important; }

    .kpi-card,
    .panel {
        background: #FFFFFF !important;
        border-color: #DCE2EC !important;
    }

    .footer { color: #596579 !important; }

    .stDownloadButton button {
        background: #5145E5 !important;
        color: #FFFFFF !important;
        border: 1px solid #5145E5 !important;
    }

    .stDownloadButton button p { color: #FFFFFF !important; }

    .stDownloadButton button:hover {
        background: #4035C7 !important;
        color: #FFFFFF !important;
    }

    input:focus,
    textarea:focus,
    button:focus-visible {
        outline-color: #5145E5 !important;
    }

    /* Ensure the small hero label remains readable on the light background */
    .stApp .eyebrow,
    .stApp [class="eyebrow"] {
        color: #312783 !important;
        opacity: 1 !important;
        visibility: visible !important;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.7px;
        text-transform: uppercase;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SIDEBAR BRANDING
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand">Demand<span>IQ</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-caption">INTELLIGENT SALES ANALYTICS</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section">Forecast controls</div>',
                unsafe_allow_html=True)

    dataset_path = Path("data/raw/train.csv")
    model_path = Path("models/sales_forecast_model.joblib")

    if not dataset_path.exists():
        st.error("Dataset not found.")
        st.markdown(
            "[Download the Kaggle dataset](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data)"
        )
        st.stop()

    if not model_path.exists():
        st.warning("Model not trained yet. Run `python train.py`.")
        st.stop()

    df = load_data()
    model = joblib.load(model_path)

    store = st.selectbox(
        "Select store",
        sorted(df["store"].unique().tolist()),
        help="Choose the store you want to analyze.",
    )

    items = sorted(
        df.loc[df["store"] == store, "item"].unique().tolist()
    )
    item = st.selectbox(
        "Select product",
        items,
        help="Choose a product for demand forecasting.",
    )

    horizon = st.slider(
        "Forecast horizon",
        min_value=3,
        max_value=12,
        value=6,
        step=3,
        help="Number of future months to predict.",
    )

    st.markdown('<div class="sidebar-section">About this project</div>',
                unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-note">
            <b>DemandIQ</b><br>
            Monthly sales forecasting using Python,
            machine learning, and historical sales data.<br><br>
            <b>Dataset</b> · Store Item Demand<br>
            <b>Frequency</b> · Monthly<br>
            <b>Output</b> · Sales forecast + CSV
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='color:#7F8CA4;font-size:11px;margin-top:24px;'>"
        "Built with Python · Pandas · Scikit-learn · Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------
header_left, header_right = st.columns([5, 1.3], vertical_alignment="center")

with header_left:
    st.markdown(
        '<div class="eyebrow" style="display:inline-block !important; background:#E8E5FF !important; color:#30258F !important; opacity:1 !important; visibility:visible !important; font-size:12px !important; font-weight:800 !important; letter-spacing:1.5px !important; padding:7px 10px !important; border:1px solid #C8C0FF !important; border-radius:6px !important; margin-bottom:12px !important; text-shadow:none !important;">SALES INTELLIGENCE / OVERVIEW</div>'
        '<div class="hero-title">Demand Forecasting</div>'
        '<div class="hero-description">'
        'Understand historical sales patterns and estimate future product demand '
        'with an interactive forecasting workspace.'
        '</div>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        '<div style="text-align:right;padding-bottom:18px;">'
        '<span class="status-pill">● MODEL LOADED</span>'
        '</div>',
        unsafe_allow_html=True,
    )

st.divider()


# --------------------------------------------------
# FORECAST CALCULATION
# --------------------------------------------------
series = monthly_series(df, store, item)

if series.empty:
    st.warning("No historical sales data is available for this selection.")
    st.stop()

forecast = forecast_recursive(series, model, horizon)

if forecast.empty:
    st.warning("The model did not return any forecast values.")
    st.stop()

forecast["sales"] = forecast["sales"].clip(lower=0)

total_forecast = forecast["sales"].sum()
average_monthly = series["sales"].mean()
forecast_average = forecast["sales"].mean()
latest_month_sales = series["sales"].iloc[-1]


# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------
st.markdown(
    '<div class="section-heading">Performance at a glance</div>'
    '<div class="section-caption">Key figures for your selected store and product</div>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)

def render_kpi(column, label, value, icon, note):
    column.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">{label}</span>
                <span class="kpi-icon">{icon}</span>
            </div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-foot">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_kpi(
    k1,
    "Historical months",
    f"{len(series):,}",
    "▦",
    "Monthly observations available",
)
render_kpi(
    k2,
    "Average monthly sales",
    f"{average_monthly:,.0f}",
    "↗",
    "Units sold historically",
)
render_kpi(
    k3,
    "Forecasted total",
    f"{total_forecast:,.0f}",
    "◎",
    f"Next {horizon} months · estimated units",
)
render_kpi(
    k4,
    "Forecast monthly average",
    f"{forecast_average:,.0f}",
    "◷",
    "Estimated units per month",
)

st.write("")


# --------------------------------------------------
# FORECAST CHART
# --------------------------------------------------
st.markdown(
    '<div class="section-heading">Sales trends & forecast</div>'
    '<div class="section-caption">'
    'Compare actual historical sales against the model-generated forecast.'
    '</div>',
    unsafe_allow_html=True,
)

chart_col, insight_col = st.columns([2.4, 1], gap="large")

with chart_col:
    st.markdown(
        '<div class="panel-title">Monthly sales trajectory</div>'
        '<div class="panel-subtitle">'
        f'Store {store} · Product {item} · Historical vs predicted'
        '</div>',
        unsafe_allow_html=True,
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    ax.plot(
        series["month"],
        series["sales"],
        color="#635BFF",
        linewidth=2.4,
        marker="o",
        markersize=3.5,
        label="Historical sales",
        zorder=3,
    )

    ax.plot(
        forecast["month"],
        forecast["sales"],
        color="#F59E0B",
        linewidth=2.4,
        marker="o",
        markersize=4,
        linestyle="--",
        label="Forecast",
        zorder=4,
    )

    ax.axvline(
        series["month"].iloc[-1],
        color="#B5BDCC",
        linestyle=":",
        linewidth=1.4,
        label="Forecast starts",
    )

    ax.fill_between(
        forecast["month"],
        forecast["sales"],
        alpha=0.10,
        color="#F59E0B",
    )

    ax.set_xlabel("Month", fontsize=10, color="#738096", labelpad=10)
    ax.set_ylabel("Units sold", fontsize=10, color="#738096", labelpad=10)
    ax.tick_params(axis="both", labelsize=9, colors="#738096")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E5EAF2")
    ax.spines["bottom"].set_color("#E5EAF2")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()

    st.pyplot(fig, width="stretch")
    plt.close(fig)

with insight_col:
    st.markdown(
        '<div class="panel-title">Forecast summary</div>'
        '<div class="panel-subtitle">Expected demand for the selected period</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="forecast-highlight">
            <div class="forecast-highlight-label">Total forecast</div>
            <div class="forecast-highlight-value">{total_forecast:,.0f}</div>
            <div class="forecast-highlight-note">Estimated units over {horizon} months</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Selection details**")
    st.write(f"🏬 Store: **{store}**")
    st.write(f"📦 Product: **{item}**")
    st.write(f"🗓️ Forecast length: **{horizon} months**")
    st.write(f"📊 Historical average: **{average_monthly:,.1f} units/month**")

    st.markdown("---")
    st.caption(
        "Forecasts are estimates based on historical patterns. "
        "Actual demand may differ."
    )


# --------------------------------------------------
# FORECAST TABLE + DOWNLOAD
# --------------------------------------------------
st.write("")
st.markdown(
    '<div class="section-heading">Forecast breakdown</div>'
    '<div class="section-caption">'
    'Review the predicted demand month by month and export the results.'
    '</div>',
    unsafe_allow_html=True,
)

table_col, download_col = st.columns([2.3, 1], gap="large")

display = forecast.copy()
display["month"] = display["month"].dt.strftime("%b %Y")
display["forecast_sales"] = display["sales"].round().astype(int)
display = display[["month", "forecast_sales"]]
display.columns = ["Forecast month", "Predicted units"]

with table_col:
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        height=min(360, 38 * (len(display) + 1)),
        column_config={
            "Forecast month": st.column_config.TextColumn(
                "Forecast month",
                width="medium",
            ),
            "Predicted units": st.column_config.NumberColumn(
                "Predicted units",
                format="%d units",
            ),
        },
    )

with download_col:
    st.markdown(
        '<div class="panel-title">Export your forecast</div>'
        '<div class="panel-subtitle">'
        'Download the prediction results for reporting or further analysis.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.download_button(
        label="⬇  Download forecast CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name=f"demand_forecast_store_{store}_item_{item}.csv",
        mime="text/csv",
        width="stretch",
    )

    st.write("")
    st.markdown(
        f"""
        <div class="panel">
            <div class="kpi-label">Export contents</div>
            <br>
            <b>Store</b>: {store}<br><br>
            <b>Product</b>: {item}<br><br>
            <b>Forecast period</b>: {horizon} months<br><br>
            <b>Records</b>: {len(display)} monthly predictions
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------
metrics_path = Path("models/model_info.json")

if metrics_path.exists():
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        st.write("")
        st.markdown(
            '<div class="section-heading">Model evaluation</div>'
            '<div class="section-caption">'
            'Historical holdout metrics from the training process.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            f"Evaluation series: Store {metrics['store']}, "
            f"Item {metrics['item']} · "
            f"{metrics['test_months']} test months"
        )

        m1, m2, m3 = st.columns(3)
        render_kpi(
            m1,
            "Mean absolute error",
            f"{metrics['MAE']:,.2f}",
            "↔",
            "MAE · Lower is better",
        )
        render_kpi(
            m2,
            "Root mean squared error",
            f"{metrics['RMSE']:,.2f}",
            "⌁",
            "RMSE · Lower is better",
        )

        mape = metrics.get("MAPE_percent_nonzero_actuals")
        if mape is not None:
            render_kpi(
                m3,
                "Mean absolute percentage error",
                f"{mape:.2f}%",
                "%",
                "Non-zero actual months only",
            )
        else:
            render_kpi(
                m3,
                "Mean absolute percentage error",
                "N/A",
                "%",
                "Not available",
            )

    except (ValueError, KeyError, TypeError) as exc:
        st.warning(f"Could not read model evaluation information: {exc}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        <b>DemandIQ</b> · Sales Demand Forecasting<br>
        Built with Python, Pandas, Scikit-learn, Matplotlib & Streamlit<br>
        For analytical and planning purposes only · Predictions are not guarantees
    </div>
    """,
    unsafe_allow_html=True,
)
