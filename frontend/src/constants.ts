import type { Regime, Region } from "./types";

export const REGIONS: Region[] = ["NW India", "Indo-Gangetic Plains", "Bay of Bengal Coast", "Central India", "Southern Peninsula"];
export const REGIMES: Regime[] = ["Active Monsoon", "Break Monsoon", "Monsoon Low/Depression", "Western Disturbance", "Heatwave", "Stable/Clear"];

/** Climatology defaults — auto-fill when the regime changes (editable via slider). */
export const REGIME_DEFAULT_RATE: Record<Regime, number> = {
  "Active Monsoon": 0.38, "Break Monsoon": 0.26, "Monsoon Low/Depression": 0.52,
  "Western Disturbance": 0.34, "Heatwave": 0.21, "Stable/Clear": 0.09,
};

/** Spec §6 bands (forecast confidence = 1 - bust probability):
 *  green <30% risk, amber 30–60%, red >=60%. */
export const riskColor = (p: number) => (p >= 0.6 ? "#ef4444" : p >= 0.3 ? "#f59e0b" : "#22c55e");