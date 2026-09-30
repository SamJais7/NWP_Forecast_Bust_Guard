# from app.models.schemas import PredictionInput, PredictionResult
# from app.services.ml_service import predict_bust
# from fastapi import APIRouter

from app.models.schemas import ForecastRequest, PredictionResult
from app.services.ml_service import ml_service, predict_bust
from fastapi import APIRouter, Request, HTTPException, Depends
from app.models.schemas import (
    SAMPLE_SCENARIOS,
    FeatureImportanceResponse,
    ForecastRequest,
    PredictionResult,
    PredictionSummary,
    Scenario,
)

router = APIRouter()
MAX_BATCH = 50


def _svc(request: Request):
    svc = getattr(request.app.state, "ml", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Model not loaded — service warming up")
    return svc


@router.post("/predict", response_model=PredictionResult, summary="Predict bust probability + SHAP drivers")
async def predict(payload: ForecastRequest, request: Request):
    try:
        return _svc(request).predict(payload)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


@router.post("/batch-predict", response_model=list[PredictionSummary], summary="Slim predictions for many scenarios")
async def batch_predict(payloads: list[ForecastRequest], request: Request):
    if not 1 <= len(payloads) <= MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"batch size must be 1–{MAX_BATCH}")
    try:
        return [_svc(request).predict_summary(p) for p in payloads]
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {exc}")


@router.get("/feature-importance", response_model=FeatureImportanceResponse, summary="Global mean |SHAP| importance")
async def feature_importance(request: Request):
    imp = getattr(request.app.state, "feature_importance", None)
    if imp is None:
        raise HTTPException(status_code=503, detail="Importance not computed yet")
    return imp


@router.get("/sample-scenarios", response_model=list[Scenario], summary="Pre-configured meteorological presets")
async def sample_scenarios():
    return SAMPLE_SCENARIOS