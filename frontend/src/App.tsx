import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, Brain, ChevronRight, Gauge as GaugeIcon, Loader2, Map as MapIcon, Satellite, TrendingUp, Zap } from "lucide-react";
import ConfidenceMap from "./components/ConfidenceMap";
import Controls from "./components/Controls";
import GaugeChart from "./components/GaugeChart";
import LeadTimeCurve from "./components/LeadTimeCurve";
import ShapWaterfall from "./components/ShapWaterfall";
import { api } from "./services/api";
import { REGIONS } from "./constants";
import type { FeatureImportanceItem, ForecastRequest, PredictionResult, PredictionSummary, Scenario } from "./types";

const DEFAULT_FORM: ForecastRequest = {
  lead_time: 7, region: "Indo-Gangetic Plains", weather_regime: "Active Monsoon",
  rainfall_anomaly_mm: 85, temp_anomaly_c: 1.5, wind_shear_anomaly_kt: 18,
  geopotential_500hpa_err: 60, historical_regime_bust_rate: 0.38,
};

const Card = ({ title, icon: Icon, children, className = "" }: { title: string; icon: React.ElementType; children: React.ReactNode; className?: string }) => (
  <section className={`rounded-xl border border-slate-800 bg-slate-900/60 p-5 ${className}`}>
    <h2 className="mb-4 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
      <Icon size={14} className="text-sky-400" />{title}
    </h2>
    {children}
  </section>
);

export default function App() {
  const [form, setForm] = useState<ForecastRequest>(DEFAULT_FORM);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [regionResults, setRegionResults] = useState<Record<string, PredictionSummary | null>>({});
  const [leadCurve, setLeadCurve] = useState<{ day: number; p: number }[]>([]);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [importance, setImportance] = useState<FeatureImportanceItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [presetId, setPresetId] = useState<string | null>(null);
  const seq = useRef(0);

  useEffect(() => {
    api.scenarios().then(setScenarios).catch(() => { /* presets optional */ });
    api.featureImportance().then((r) => setImportance(r.features)).catch(() => { /* optional */ });
  }, []);

  const refresh = useCallback(async (f: ForecastRequest) => {
    const id = ++seq.current;
    setLoading(true);
    setError(null);
    try {
      const [main, regions, leads] = await Promise.all([
        api.predict(f),
        api.predictBatch(REGIONS.map((r) => ({ ...f, region: r }))),
        api.predictBatch(Array.from({ length: 10 }, (_, i) => ({ ...f, lead_time: i + 1 }))),
      ]);
      if (seq.current !== id) return;
      setResult(main);
      setRegionResults(Object.fromEntries(REGIONS.map((r, i) => [r, regions[i]])));
      setLeadCurve(leads.map((x, i) => ({ day: i + 1, p: x.bust_probability })));
    } catch (e) {
      if (seq.current === id) setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      if (seq.current === id) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const t = setTimeout(() => void refresh(form), 300);
    return () => clearTimeout(t);
  }, [form, refresh]);

  const patch = (p: Partial<ForecastRequest>) => { setPresetId(null); setForm((f) => ({ ...f, ...p })); };
  const loadPreset = (s: Scenario) => { setPresetId(s.id); setForm(s.request); };
  const maxImp = importance[0]?.mean_abs_shap ?? 1;

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b border-slate-800 bg-slate-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-6 py-3">
          <Satellite size={20} className="text-sky-400" />
          <div>
            <h1 className="text-sm font-bold tracking-wide">NWP FORECAST BUST GUARD</h1>
            <p className="text-[11px] text-slate-500">Explainable bust-risk monitoring · Day 1–10 · free-tier architecture</p>
          </div>
          <div className="ml-auto flex items-center gap-2 text-[11px]">
            {loading && <Loader2 size={14} className="animate-spin text-sky-400" />}
            {error ? (
              <span className="flex items-center gap-1.5 rounded-full border border-rose-500/40 bg-rose-500/10 px-3 py-1 text-rose-300">
                <span className="h-2 w-2 rounded-full bg-rose-400" />API unreachable
              </span>
            ) : (
              <span className="flex items-center gap-1.5 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-emerald-300">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />API live
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-5 px-6 py-6 lg:grid-cols-[340px_1fr]">
        <aside className="space-y-5">
          <Card title="Controls" icon={GaugeIcon}><Controls form={form} scenarios={scenarios} activePreset={presetId} onChange={patch} onPreset={loadPreset} /></Card>
          <Card title="Global drivers (mean |SHAP|)" icon={Brain}>
            {importance.length ? (
              <div className="space-y-2">
                {importance.slice(0, 8).map((f) => (
                  <div key={f.feature} className="text-[11px]">
                    <div className="flex justify-between text-slate-400"><span>{f.label}</span><span className="font-mono">{f.mean_abs_shap.toFixed(3)}</span></div>
                    <div className="mt-0.5 h-1.5 rounded bg-slate-800">
                      <div className="h-1.5 rounded bg-sky-500/70" style={{ width: `${(f.mean_abs_shap / maxImp) * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            ) : <p className="text-xs text-slate-500">loading…</p>}
          </Card>
        </aside>

        <div className="space-y-5">
          {error && (
            <div className="flex items-center gap-2 rounded-lg border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
              <AlertTriangle size={16} /> {error}
            </div>
          )}

          <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
            <Card title="Bust probability gauge" icon={GaugeIcon}>
              {result ? (
                <GaugeChart probability={result.bust_probability} riskLevel={result.risk_level}
                  confidenceLevel={result.confidence_level} latencyMs={result.latency_ms} />
              ) : <p className="py-16 text-center text-sm text-slate-500">waiting for first prediction…</p>}
            </Card>

            <Card title="Operational summary" icon={Zap}>
              {result ? (
                <div className="flex h-full flex-col">
                  <p className="text-sm leading-relaxed text-slate-200">{result.summary}</p>
                  <div className="mt-4 space-y-1.5">
                    {result.top_factors.map((f, i) => (
                      <div key={i} className="flex items-start gap-1.5 text-xs text-slate-400">
                        <ChevronRight size={13} className="mt-0.5 shrink-0 text-sky-400" />{f}
                      </div>
                    ))}
                  </div>
                  <p className="mt-auto pt-4 text-[10px] text-slate-600">
                    model v{result.model_version} · raw xgb p={result.raw_probability.toFixed(3)} · isotonic-calibrated
                  </p>
                </div>
              ) : <p className="py-16 text-center text-sm text-slate-500">…</p>}
            </Card>
          </div>

          <Card title="SHAP driver breakdown (probability points)" icon={Brain}>
            {result ? <ShapWaterfall contributions={result.shap_values} /> : <p className="py-16 text-center text-sm text-slate-500">…</p>}
          </Card>

          <Card title="Lead-time risk curve (Days 1–10, current setup)" icon={TrendingUp}>
            {leadCurve.length ? <LeadTimeCurve data={leadCurve} selectedDay={form.lead_time} /> : <p className="py-16 text-center text-sm text-slate-500">…</p>}
          </Card>

          <Card title="Regional vulnerability choropleth" icon={MapIcon}>
            <ConfidenceMap results={regionResults} selected={form.region} onSelect={(r) => patch({ region: r })} />
          </Card>
        </div>
      </main>

      <footer className="border-t border-slate-800 py-4 text-center text-[11px] text-slate-600">
        NWP Forecast Bust Guard · XGBoost + isotonic calibration + SHAP · FastAPI · React · 100% free-tier hosting
      </footer>
    </div>
  );
}