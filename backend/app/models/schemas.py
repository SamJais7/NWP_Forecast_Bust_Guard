"""Pydantic schemas — mirror the assessment CSV schema exactly (range validation per spec §5)."""
from typing import List, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

REGIONS = (
    "NW India",
    "Indo-Gangetic Plains",
    "Bay of Bengal Coast",
    "Central India",
    "Southern Peninsula",
)
REGIMES = (
    "Active Monsoon",
    "Break Monsoon",
    "Monsoon Low/Depression",
    "Western Disturbance",
    "Heatwave",
    "Stable/Clear",
)

Region = Literal[
    "NW India",
    "Indo-Gangetic Plains",
    "Bay of Bengal Coast",
    "Central India",
    "Southern Peninsula",
]
Regime = Literal[
    "Active Monsoon",
    "Break Monsoon",
    "Monsoon Low/Depression",
    "Western Disturbance",
    "Heatwave",
    "Stable/Clear",
]

_REGION_ALIASES = {
    "nw india": "NW India",
    "north west india": "NW India",
    "northwest india": "NW India",
    "indo gangetic plains": "Indo-Gangetic Plains",
    "gangetic plains": "Indo-Gangetic Plains",
    "bay of bengal coast": "Bay of Bengal Coast",
    "bay of bengal": "Bay of Bengal Coast",
    "east coast": "Bay of Bengal Coast",
    "central india": "Central India",
    "southern peninsula": "Southern Peninsula",
    "south peninsula": "Southern Peninsula",
}
_REGIME_ALIASES = {
    "active monsoon": "Active Monsoon",
    "break monsoon": "Break Monsoon",
    "monsoon low/depression": "Monsoon Low/Depression",
    "monsoon low depression": "Monsoon Low/Depression",
    "monsoon low": "Monsoon Low/Depression",
    "depression": "Monsoon Low/Depression",
    "cyclone": "Monsoon Low/Depression",
    "western disturbance": "Western Disturbance",
    "wd": "Western Disturbance",
    "heatwave": "Heatwave",
    "heat wave": "Heatwave",
    "stable/clear": "Stable/Clear",
    "stable clear": "Stable/Clear",
    "stable": "Stable/Clear",
    "clear": "Stable/Clear",
}


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().replace("_", " ").replace("-", " ").split())


def canonical_region(v) -> str:
    if not isinstance(v, str):
        raise ValueError("region must be a string")
    key = _norm(v)
    if key in _REGION_ALIASES:
        return _REGION_ALIASES[key]
    raise ValueError(f"Unknown region '{v}'. Valid regions: {list(REGIONS)}")


def canonical_regime(v) -> str:
    if not isinstance(v, str):
        raise ValueError("weather_regime must be a string")
    key = _norm(v)
    if key in _REGIME_ALIASES:
        return _REGIME_ALIASES[key]
    raise ValueError(f"Unknown weather_regime '{v}'. Valid regimes: {list(REGIMES)}")


class ForecastRequest(BaseModel):
    lead_time: int = Field(
        ..., ge=1, le=10, description="Forecast horizon in days (1–10)", examples=[7]
    )
    region: Region
    weather_regime: Regime
    rainfall_anomaly_mm: float = Field(
        ..., ge=0.0, le=250.0, description="Deviation from 30-yr precipitation normal"
    )
    temp_anomaly_c: float = Field(
        ..., ge=-8.0, le=10.0, description="2 m temperature anomaly vs climatology"
    )
    wind_shear_anomaly_kt: float = Field(
        ..., ge=-35.0, le=45.0, description="850–200 hPa deep-layer shear anomaly"
    )
    geopotential_500hpa_err: float = Field(
        ..., ge=-120.0, le=160.0, description="Mid-tropospheric height bias (gpm)"
    )
    historical_regime_bust_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Historical bust frequency for this regime"
    )

    @field_validator("region", mode="before")
    @classmethod
    def _region(cls, v):
        return canonical_region(v)

    @field_validator("weather_regime", mode="before")
    @classmethod
    def _regime(cls, v):
        return canonical_regime(v)


# Alias so imports using PredictionInput do not fail
PredictionInput = ForecastRequest


class PredictionSummary(BaseModel):
    bust_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["LOW", "MODERATE", "HIGH", "SEVERE"]
    confidence_level: Literal["LOW", "MEDIUM", "HIGH"]


class ShapContribution(BaseModel):
    feature: str
    label: str
    value: float
    contribution: float
    active: bool = True


class PredictionResult(PredictionSummary):
    model_config = ConfigDict(protected_namespaces=())

    raw_probability: float
    top_factors: List[str]
    summary: str
    shap_values: List[ShapContribution]
    latency_ms: float
    model_version: str


class FeatureImportanceItem(BaseModel):
    feature: str
    label: str
    mean_abs_shap: float


class FeatureImportanceResponse(BaseModel):
    features: List[FeatureImportanceItem]
    n_background: int


class Scenario(BaseModel):
    id: str
    name: str
    description: str
    request: ForecastRequest


SAMPLE_SCENARIOS: List[Scenario] = [
    Scenario(
        id="remal",
        name="Cyclone Remal Surge",
        description="Post-landfall monsoon depression over the BoB coast — extreme rainfall, shear and height errors.",
        request=ForecastRequest(
            lead_time=5,
            region="Bay of Bengal Coast",
            weather_regime="Monsoon Low/Depression",
            rainfall_anomaly_mm=185,
            temp_anomaly_c=1.2,
            wind_shear_anomaly_kt=38,
            geopotential_500hpa_err=120,
            historical_regime_bust_rate=0.52,
        ),
    ),
    Scenario(
        id="july-break",
        name="July Monsoon Break",
        description="Break-phase dry spell over Central India with a warm bias and collapsing shear.",
        request=ForecastRequest(
            lead_time=7,
            region="Central India",
            weather_regime="Break Monsoon",
            rainfall_anomaly_mm=8,
            temp_anomaly_c=3.4,
            wind_shear_anomaly_kt=-6,
            geopotential_500hpa_err=-40,
            historical_regime_bust_rate=0.26,
        ),
    ),
    Scenario(
        id="winter-wd",
        name="Winter Western Disturbance",
        description="Deep westerly trough impacting NW India — cold anomaly with strong shear forecast.",
        request=ForecastRequest(
            lead_time=6,
            region="NW India",
            weather_regime="Western Disturbance",
            rainfall_anomaly_mm=65,
            temp_anomaly_c=-4.5,
            wind_shear_anomaly_kt=18,
            geopotential_500hpa_err=85,
            historical_regime_bust_rate=0.34,
        ),
    ),
    Scenario(
        id="heatwave",
        name="Clear Heatwave",
        description="Stable heat spell over NW India — sub-seasonal ridge amplification risk.",
        request=ForecastRequest(
            lead_time=3,
            region="NW India",
            weather_regime="Heatwave",
            rainfall_anomaly_mm=0,
            temp_anomaly_c=8.2,
            wind_shear_anomaly_kt=-12,
            geopotential_500hpa_err=25,
            historical_regime_bust_rate=0.21,
        ),
    ),
    Scenario(
        id="day10-surge",
        name="Active Monsoon Surge, Day 10",
        description="Peak-season monsoon surge over the Indo-Gangetic Plains at maximum lead time.",
        request=ForecastRequest(
            lead_time=10,
            region="Indo-Gangetic Plains",
            weather_regime="Active Monsoon",
            rainfall_anomaly_mm=120,
            temp_anomaly_c=0.5,
            wind_shear_anomaly_kt=22,
            geopotential_500hpa_err=95,
            historical_regime_bust_rate=0.38,
        ),
    ),
]