from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="EV Charging Demand & Load Management", layout="wide")
st.title("AI-Based EV Charging Demand Prediction & Load Management")
st.caption("Educational capstone prototype — supplied data are simulated, not live station data.")

hourly = pd.read_csv(ROOT/"data"/"hourly_demand_features.csv", parse_dates=["hour_start"])
sessions = pd.read_csv(ROOT/"data"/"ev_charging_sessions_simulated.csv", parse_dates=["start_time","end_time"])
tab1, tab2, tab3 = st.tabs(["Demand overview", "Forecast model", "Load scheduling"])

with tab1:
    c1,c2,c3 = st.columns(3)
    c1.metric("Charging sessions", f"{len(sessions):,}")
    c2.metric("Total energy", f"{sessions.energy_kwh.sum():,.0f} kWh")
    c3.metric("Stations", f"{sessions.station_id.nunique()}")
    st.subheader("Hourly charging demand")
    fig, ax = plt.subplots()
    ax.plot(hourly.hour_start, hourly.demand_kwh, linewidth=1)
    ax.set_xlabel("Time"); ax.set_ylabel("Demand (kWh per hour)")
    st.pyplot(fig)
    st.subheader("Average demand by hour of day")
    byhour = hourly.groupby("hour")["demand_kwh"].mean()
    fig2, ax2 = plt.subplots()
    ax2.bar(byhour.index, byhour.values)
    ax2.set_xlabel("Hour of day"); ax2.set_ylabel("Mean kWh")
    st.pyplot(fig2)

with tab2:
    st.write("Run `python src/train_model.py` to train the chronological Random Forest model.")
    pred_path = ROOT/"outputs"/"model_predictions.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path, parse_dates=["hour_start"])
        st.dataframe(pred.tail(100), use_container_width=True)
        fig, ax = plt.subplots()
        ax.plot(pred.hour_start, pred.actual_kwh, label="Actual")
        ax.plot(pred.hour_start, pred.predicted_kwh, label="Predicted")
        ax.legend(); ax.set_ylabel("kWh per hour")
        st.pyplot(fig)
    else:
        st.info("Prediction file not found yet. Run the training script first.")

with tab3:
    cap = st.slider("Hourly capacity limit (kWh per hour)", min_value=10, max_value=200, value=80, step=5)
    from src.scheduler import flatten_peak_load
    scheduled, unallocated = flatten_peak_load(hourly[["hour_start","demand_kwh"]], capacity_kw=cap)
    a,b = st.columns(2)
    a.metric("Peak before", f"{scheduled.demand_kwh.max():.1f} kWh")
    b.metric("Peak after", f"{scheduled.scheduled_kwh.max():.1f} kWh")
    st.metric("Unallocated excess after available future capacity", f"{unallocated:.1f} kWh")
    fig3, ax3 = plt.subplots()
    ax3.plot(scheduled.hour_start, scheduled.demand_kwh, label="Original")
    ax3.plot(scheduled.hour_start, scheduled.scheduled_kwh, label="Scheduled")
    ax3.legend(); ax3.set_ylabel("kWh per hour")
    st.pyplot(fig3)
    st.caption("Simplified hourly energy shifting only. A production scheduler must respect each vehicle's arrival/departure time, required energy, charger power, and user priority.")
