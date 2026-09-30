export type Region = "NW India" | "Indo-Gangetic Plains" | "Bay of Bengal Coast" | "Central India" | "Southern Peninsula";
export type Regime = "Active Monsoon" | "Break Monsoon" | "Monsoon Low/Depression" | "Western Disturbance" | "Heatwave" | "Stable/Clear";

export interface ForecastRequest {
  lead_time: number; region: Region; weather_regime: Regime;
  rainfall_anomaly_mm: number; temp_anomaly_c: number; wind_shear_anomaly_kt: number;
  geopotential_500hpa_err: number; historical_regime_bust_rate: number;
}
export interface ShapContribution { feature: string; label: string; value: number; contribution: number; active: boolean; }
export interface PredictionSummary {
  bust_probability: number;
  risk_level: "LOW" | "MODERATE" | "HIGH" | "SEVERE";
  confidence_level: "LOW" | "MEDIUM" | "HIGH";
}
export interface PredictionResult extends PredictionSummary {
  raw_probability: number; top_factors: string[]; summary: string;
  shap_values: ShapContribution[]; latency_ms: number; model_version: string;
}
export interface Scenario { id: string; name: string; description: string; request: ForecastRequest; }
export interface FeatureImportanceItem { feature: string; label: string; mean_abs_shap: number; }