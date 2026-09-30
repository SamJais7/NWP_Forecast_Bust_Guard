"""Model loader + inference + SHAP explainability runtime + natural-language reasoning."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.models.schemas import ForecastRequest

PRETTY = {
    "lead_time": "Lead time (days)",
    "rainfall_anomaly_mm": "Rainfall anomaly (mm)",
    "temp_anomaly_c": "Temperature anomaly (°C)",
    "wind_shear_anomaly_kt": "Wind shear anomaly (kt)",
    "geopotential_500hpa_err": "500 hPa geopot. error (gpm)",
    "historical_regime_bust_rate": "Historical regime bust rate",
}


def _pretty(feature: str) -> str:
    if feature.startswith("region_"):
        return f"Region: {feature[len('region_'):]}"
    if feature.startswith("weather_regime_"):
        return f"Regime: {feature[len('weather_regime_'):]}"
    return PRETTY.get(feature, feature)


def _is_onehot(feature: str) -> bool:
    return feature.startswith(("region_", "weather_regime_"))


def risk_level(p: float) -> str:
    if p < 0.30:
        return "LOW"
    if p < 0.55:
        return "MODERATE"
    if p < 0.75:
        return "HIGH"
    return "SEVERE"


def forecast_confidence(p: float) -> str:
    """Spec §5/§6: confidence_level is FORECAST confidence (= 1 - bust_probability).
    p=0.78 -> 'LOW', matching the assessment's example response."""
    fc = 1.0 - p
    if fc >= 0.70:
        return "HIGH"
    if fc >= 0.40:
        return "MEDIUM"
    return "LOW"


_VERDICTS = {
    "LOW": "Forecast confidence is high; NWP guidance can be trusted for planning.",
    "MODERATE": "Monitor closely — ensemble spread is likely growing.",
    "HIGH": "Cross-check alternative models and nowcasts before acting on this guidance.",
    "SEVERE": "Treat NWP guidance as unreliable; trigger contingency protocols.",
}


def _phrase(c: dict) -> str | None:
    """Plain-English driver phrase aligned with the numeric anomaly direction."""
    f, v, s = c["feature"], c["value"], c["contribution"]
    if abs(s) < 0.005:
        return None
    up = s > 0
    if f == "lead_time":
        return (f"Day-{int(v)} lead time amplifies atmospheric chaos" if up
                else f"Short Day-{int(v)} window keeps conditions predictable")
    if f == "rainfall_anomaly_mm":
        return (f"Heavy rainfall anomaly (+{v:.0f} mm) signals convective unpredictability" if up
                else f"Low rainfall anomaly ({v:.0f} mm) keeps precipitation guidance stable")
    if f == "temp_anomaly_c":
        return (f"{v:+.1f} °C temperature anomaly pushes the regime toward a bust" if up
                else f"Mild temperature anomaly ({v:+.1f} °C) keeps the thermal profile predictable")
    if f == "wind_shear_anomaly_kt":
        return (f"Elevated deep-layer shear anomaly ({v:+.0f} kt) hints at regime-transition risk" if up
                else f"Benign shear anomaly ({v:+.0f} kt) supports a steady forecast")
    if f == "geopotential_500hpa_err":
        return (f"Large 500 hPa height error ({v:+.0f} gpm) signals a circulation-pattern shift" if up
                else f"Well-behaved mid-level heights ({v:+.0f} gpm error) anchor the pattern")
    if f == "historical_regime_bust_rate":
        return (f"This regime historically busts {v:.0%} of the time" if up
                else f"This regime is historically reliable (bust rate only {v:.0%})")
    if f.startswith("weather_regime_"):
        name = f[len("weather_regime_"):]
        return (f"'{name}' regime is a known high-difficulty scenario" if up
                else f"'{name}' regime is comparatively easy to forecast")
    if f.startswith("region_"):
        name = f[len("region_"):]
        return (f"The {name} sector adds regional forecast risk" if up
                else f"The {name} sector is a lower-risk area for this setup")
    return f"{c['label']} contribution {s:+.3f}"


class MLService:
    """Loads serialized artifacts once; serves calibrated inference + SHAP explanations."""

    def __init__(self, model_path: Path, explainer_path: Path):
        bundle = joblib.load(model_path)
        self.pre: Any = bundle["preprocessor"]
        self.model: Any = bundle["model"]
        self.iso: Any = bundle["calibrator"]
        self.feature_names: list[str] = list(bundle["feature_names"])
        self.metrics: dict = bundle.get("metrics", {})
        self.version: str = bundle.get("version", "1.0.0")

        eb = joblib.load(explainer_path)
        self.explainer: Any = eb["explainer"]
        self.background: pd.DataFrame = eb["background"]

    # ------------------------------------------------------------------ core
    def _transform(self, req: ForecastRequest) -> pd.DataFrame:
        row = pd.DataFrame([{
            "lead_time": req.lead_time,
            "rainfall_anomaly_mm": req.rainfall_anomaly_mm,
            "temp_anomaly_c": req.temp_anomaly_c,
            "wind_shear_anomaly_kt": req.wind_shear_anomaly_kt,
            "geopotential_500hpa_err": req.geopotential_500hpa_err,
            "historical_regime_bust_rate": req.historical_regime_bust_rate,
            "region": req.region,
            "weather_regime": req.weather_regime,
        }])
        xt = self.pre.transform(row)
        return pd.DataFrame(xt, columns=self.feature_names)

    def predict_proba(self, req: ForecastRequest) -> float:
        x = self._transform(req)
        raw = float(self.model.predict_proba(x)[0, 1])
        return float(np.clip(self.iso.predict(np.array([raw]))[0], 0.0, 1.0))

    # ------------------------------------------------------------- endpoints
    def predict(self, req: ForecastRequest) -> dict:
        t0 = time.perf_counter()
        x = self._transform(req)
        raw = float(self.model.predict_proba(x)[0, 1])
        p = float(np.clip(self.iso.predict(np.array([raw]))[0], 0.0, 1.0))

        sv = np.asarray(self.explainer.shap_values(x))
        if sv.ndim == 3:
            sv = sv[..., -1]
        sv = sv.reshape(-1)[: len(self.feature_names)]
        vals = x.iloc[0].to_dict()

        contributions = [{
            "feature": f,
            "label": _pretty(f),
            "value": float(vals[f]),
            "contribution": float(s),
            "active": (not _is_onehot(f)) or abs(float(vals[f]) - 1.0) < 1e-9,
        } for f, s in zip(self.feature_names, sv)]

        ranked = sorted((c for c in contributions if c["active"]),
                       key=lambda c: abs(c["contribution"]), reverse=True)
        top_factors = [ph for ph in (_phrase(c) for c in ranked[:5]) if ph]

        rl = risk_level(p)
        drivers = ", ".join(f.lower() for f in top_factors[:3]) or "no single dominant driver"
        summary = (f"{rl} bust risk ({p:.0%}) on the Day-{req.lead_time} {req.weather_regime} "
                  f"forecast for {req.region}. Dominant drivers: {drivers}. {_VERDICTS[rl]}")

        return {
            "bust_probability": p,
            "risk_level": rl,
            "confidence_level": forecast_confidence(p),
            "raw_probability": raw,
            "top_factors": top_factors[:3] or ["No dominant risk drivers detected"],
            "summary": summary,
            "shap_values": contributions,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "model_version": self.version,
        }

    def predict_summary(self, req: ForecastRequest) -> dict:
        p = self.predict_proba(req)
        return {"bust_probability": p, "risk_level": risk_level(p),
                "confidence_level": forecast_confidence(p)}

    def global_feature_importance(self) -> dict:
        sv = np.asarray(self.explainer.shap_values(self.background))
        if sv.ndim == 3:
            sv = sv[..., -1]
        imp = np.abs(sv).mean(axis=0)
        ranked = sorted(zip(self.feature_names, imp), key=lambda t: t[1], reverse=True)
        return {
            "features": [{"feature": f, "label": _pretty(f), "mean_abs_shap": float(i)} for f, i in ranked],
            "n_background": int(len(self.background)),
        }
# --- Singleton instance & bridge function ---
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_ARTIFACTS_DIR = _BASE_DIR / "artifacts"
_MODEL_FILE = _ARTIFACTS_DIR / "bust_model.joblib"
_EXPLAINER_FILE = _ARTIFACTS_DIR / "explainer.joblib"

ml_service = MLService(model_path=_MODEL_FILE, explainer_path=_EXPLAINER_FILE)


def predict_bust(req):
  return ml_service.predict(req)