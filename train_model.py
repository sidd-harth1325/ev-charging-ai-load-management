from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / "data" / "hourly_demand_features.csv", parse_dates=["hour_start"])
features = ["hour","day_of_week","month","is_weekend","mean_duration_min","mean_temperature_c","lag_1","lag_24","rolling_24"]
split = int(len(data) * 0.8)  # chronological split; avoid random leakage
train, test = data.iloc[:split], data.iloc[split:]
model = RandomForestRegressor(n_estimators=160, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)
model.fit(train[features], train["demand_kwh"])
pred = np.maximum(0, model.predict(test[features]))
mae = mean_absolute_error(test["demand_kwh"], pred)
rmse = mean_squared_error(test["demand_kwh"], pred) ** 0.5
r2 = r2_score(test["demand_kwh"], pred)
print(f"MAE: {mae:.3f} kWh | RMSE: {rmse:.3f} kWh | R2: {r2:.3f}")
joblib.dump({"model": model, "features": features}, ROOT / "outputs" / "demand_model.joblib")
pd.DataFrame({"hour_start": test["hour_start"], "actual_kwh": test["demand_kwh"], "predicted_kwh": pred}).to_csv(ROOT / "outputs" / "model_predictions.csv", index=False)
