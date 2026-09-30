"""FastAPI entrypoint — NWP Forecast Bust Guard."""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# from app.api.v1.endpoints.forecast import router as forecast_router
from app.api.v1.endpoints.forecast import router as forecast_router
from app.core.config import settings
from app.services.ml_service import MLService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload model + SHAP explainer so /predict stays well under 100 ms."""
    app.state.ml = MLService(settings.model_path, settings.explainer_path)  # fail-fast if artifacts missing
    try:
        app.state.feature_importance = app.state.ml.global_feature_importance()
    except Exception as exc:  # noqa: BLE001 — importance is optional, never block serving
        print(f"[startup] global importance precompute skipped: {exc}")
        app.state.feature_importance = None
    print(f"[startup] model v{app.state.ml.version} ready — "
          f"test PR-AUC {app.state.ml.metrics.get('xgb_calibrated', {}).get('pr_auc', 'n/a')}")
    yield
    app.state.ml = None


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Explainable prediction of NWP forecast busts (Day 1–10) with calibrated probabilities and SHAP drivers.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    return response


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):  # noqa: BLE001
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(forecast_router, prefix=settings.api_v1_prefix, tags=["forecast"])


@app.get("/", tags=["meta"])
def root():
    return {"service": settings.app_name, "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["meta"])
def health(request: Request):
    ml = getattr(request.app.state, "ml", None)
    return {"status": "ok", "model_loaded": ml is not None,
            "model_version": ml.version if ml else None, "metrics": ml.metrics if ml else None}