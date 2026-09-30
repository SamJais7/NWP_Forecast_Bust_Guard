"""SYNTHETIC stand-in matching the assessment schema (delete once the real CSV is provided).
Physics-flavoured generation: regime chaos, lead-time growth of error, anomaly magnitudes."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models.schemas import REGIMES, REGIONS  # noqa: E402

REGIME_P = [0.22, 0.14, 0.16, 0.13, 0.13, 0.22]
REGIME_BIAS = {"Active Monsoon": 0.55, "Break Monsoon": 0.05, "Monsoon Low/Depression": 0.95,
               "Western Disturbance": 0.45, "Heatwave": 0.30, "Stable/Clear": -1.30}
REGIME_CHAOS = {"Active Monsoon": 0.22, "Break Monsoon": 0.14, "Monsoon Low/Depression": 0.30,
                "Western Disturbance": 0.18, "Heatwave": 0.10, "Stable/Clear": 0.05}
BASE_HIST = {"Active Monsoon": 0.38, "Break Monsoon": 0.26, "Monsoon Low/Depression": 0.52,
             "Western Disturbance": 0.34, "Heatwave": 0.22, "Stable/Clear": 0.08}
REGION_BIAS = {"NW India": 0.05, "Indo-Gangetic Plains": 0.25, "Bay of Bengal Coast": 0.20,
               "Central India": 0.00, "Southern Peninsula": -0.10}
PAIR_EXTRA = {("Bay of Bengal Coast", "Monsoon Low/Depression"): 0.45,
              ("NW India", "Western Disturbance"): 0.35,
              ("Indo-Gangetic Plains", "Active Monsoon"): 0.20,
              ("Central India", "Break Monsoon"): 0.15}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate_dataset(path: Path, n: int = 16000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lead = rng.integers(1, 11, n)
    regime = rng.choice(REGIMES, n, p=REGIME_P)
    region = rng.choice(REGIONS, n)
    rain = np.clip(rng.gamma(2.0, 22.0, n), 0.0, 250.0)
    temp = np.clip(rng.normal(0.8, 2.8, n), -8.0, 10.0)
    shear = np.clip(rng.normal(4.0, 11.0, n), -35.0, 45.0)
    geo = np.clip(rng.normal(5.0, 42.0, n), -120.0, 160.0)
    hist = np.clip(np.array([BASE_HIST[r] for r in regime]) + rng.normal(0, 0.06, n), 0.02, 0.95)

    z = (
        -3.35
        + np.array([REGIME_BIAS[g] for g in regime])
        + np.array([REGION_BIAS[r] for r in region])
        + np.array([PAIR_EXTRA.get((r, g), 0.0) for r, g in zip(region, regime)])
        + 0.09 * (lead - 1)                                        # linear lead growth
        + np.array([REGIME_CHAOS[g] for g in regime]) * (lead / 10.0) ** 2 * 3.2  # late-horizon blow-up
        + 0.9 * rain / 250.0 + 1.1 * np.maximum(rain - 80.0, 0.0) / 170.0
        + 0.10 * np.abs(temp) - 0.04 * temp
        + 0.035 * shear
        + 0.011 * np.abs(geo) + 0.004 * geo
        + 2.4 * hist
        + rng.normal(0.0, 1.05, n)
    )
    p = sigmoid(z)
    df = pd.DataFrame({
        "lead_time": lead, "region": region, "weather_regime": regime,
        "rainfall_anomaly_mm": rain, "temp_anomaly_c": temp,
        "wind_shear_anomaly_kt": shear, "geopotential_500hpa_err": geo,
        "historical_regime_bust_rate": hist,
        "bust_probability": np.clip(p + rng.normal(0, 0.03, n), 0.0, 1.0),
        "is_bust": rng.binomial(1, p),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[make_dataset] wrote {len(df)} rows -> {path} | bust rate={df.is_bust.mean():.3f}")
    return df


if __name__ == "__main__":
    generate_dataset(Path(__file__).resolve().parents[1] / "data" / "nwp_eval_dataset.csv")