import pandas as pd

def flatten_peak_load(hourly_df, capacity_kw=80.0):
    """
    Simple demonstrator: cap hourly energy at a configurable hourly capacity,
    then move excess to later hours with spare capacity. This assumes demand is
    flexible and energy can be deferred; validate against real session windows
    before deployment.
    Input columns: hour_start, demand_kwh
    """
    out = hourly_df[["hour_start", "demand_kwh"]].copy()
    out["scheduled_kwh"] = out["demand_kwh"].astype(float)
    cap = float(capacity_kw)  # kWh per one-hour interval; equivalent average kW
    excess = 0.0
    for i in out.index:
        if out.loc[i, "scheduled_kwh"] > cap:
            excess += out.loc[i, "scheduled_kwh"] - cap
            out.loc[i, "scheduled_kwh"] = cap
        elif excess > 0:
            spare = max(0.0, cap - out.loc[i, "scheduled_kwh"])
            moved = min(spare, excess)
            out.loc[i, "scheduled_kwh"] += moved
            excess -= moved
    out["capacity_violation_before"] = out["demand_kwh"] > cap
    out["capacity_violation_after"] = out["scheduled_kwh"] > cap
    return out, excess

if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root/"data"/"hourly_demand_features.csv", parse_dates=["hour_start"])
    result, unallocated = flatten_peak_load(df[["hour_start","demand_kwh"]], capacity_kw=80)
    result.to_csv(root/"outputs"/"scheduled_demand.csv", index=False)
    print("Unallocated excess energy (kWh):", round(unallocated, 2))
