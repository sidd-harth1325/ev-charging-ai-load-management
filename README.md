# AI-Based EV Charging Demand Prediction and Load Management

## Project status
Complete educational prototype containing simulated session data, hourly feature dataset, Random Forest training script, a simple load-flattening scheduler, Streamlit dashboard, report and presentation.

**Important:** the supplied dataset is synthetically generated for demonstration. It is not real station data and results must not be represented as validated real-world performance.

## Run locally
1. Install Python 3.10+.
2. In this folder run: `pip install -r requirements.txt`
3. Train and evaluate: `python src/train_model.py`
4. Launch dashboard: `streamlit run app.py`

## Files
- `data/ev_charging_sessions_simulated.csv`: session-level records.
- `data/hourly_demand_features.csv`: hourly demand and engineered features.
- `src/train_model.py`: chronological 80/20 split, Random Forest, MAE/RMSE/R² and saved predictions.
- `src/scheduler.py`: simple capacity-aware hourly shifting demonstration.
- `app.py`: Streamlit dashboard.
- `outputs/scheduling_comparison.csv`: illustrative schedule comparison.
- `Report.docx`, `Presentation.pptx`: editable deliverables.

## Method
The model predicts hourly energy demand using time-of-day, day-of-week, month, weekend, duration/temperature averages, lag-1, lag-24 and rolling 24-hour demand. The final 20% of time is held out for evaluation.

## Limitations
This is a proof-of-concept. The scheduler moves aggregate hourly energy and does not yet optimize individual vehicle sessions. Before real deployment, use verified station data, strict time-based validation, charger and grid constraints, vehicle departure deadlines, and an optimization solver such as PuLP or OR-Tools.
