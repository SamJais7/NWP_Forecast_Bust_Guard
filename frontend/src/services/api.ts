import type { FeatureImportanceItem, ForecastRequest, PredictionResult, PredictionSummary, Scenario } from "../types";

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" }, ...init,
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      const d = j.detail ?? msg;
      msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((x: { msg?: string }) => x.msg ?? "").join("; ") : msg;
    } catch { /* ignore body parse errors */ }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  predict: (r: ForecastRequest) => req<PredictionResult>("/api/v1/predict", { method: "POST", body: JSON.stringify(r) }),
  predictBatch: (rs: ForecastRequest[]) => req<PredictionSummary[]>("/api/v1/batch-predict", { method: "POST", body: JSON.stringify(rs) }),
  scenarios: () => req<Scenario[]>("/api/v1/sample-scenarios"),
  featureImportance: () => req<{ features: FeatureImportanceItem[]; n_background: number }>("/api/v1/feature-importance"),
};