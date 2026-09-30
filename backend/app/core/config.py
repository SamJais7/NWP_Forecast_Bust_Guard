from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]  # -> backend/


class Settings(BaseSettings):
    app_name: str = "NWP Forecast Bust Guard API"
    api_v1_prefix: str = "/api/v1"
    model_path: Path = BASE_DIR / "artifacts" / "bust_model.joblib"
    explainer_path: Path = BASE_DIR / "artifacts" / "explainer.joblib"
    cors_origins: str = "*"  # comma-separated; override in prod, e.g. https://nwp.vercel.app

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()