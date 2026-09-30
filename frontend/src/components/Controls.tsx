import { CalendarDays, CloudLightning, CloudRain, Flame, History, MapPin, Mountain, Snowflake, Sun, Thermometer, Wind } from "lucide-react";
import type { ForecastRequest, Regime, Region, Scenario } from "../types";
import { REGIME_DEFAULT_RATE, REGIMES, REGIONS } from "../constants";

const PRESET_ICONS = [CloudLightning, Sun, Snowflake, Flame, CloudRain];

interface Props {
  form: ForecastRequest;
  scenarios: Scenario[];
  activePreset: string | null;
  onChange: (patch: Partial<ForecastRequest>) => void;
  onPreset: (s: Scenario) => void;
}

function SliderField({ label, unit, icon: Icon, min, max, step, value, onChange, fmt }:
  { label: string; unit: string; icon: React.ElementType; min: number; max: number; step: number; value: number; onChange: (v: number) => void; fmt?: (v: number) => string }) {
  return (
    <label className="block">
      <div className="mb-1.5 flex items-center justify-between text-xs text-slate-400">
        <span className="flex items-center gap-1.5"><Icon size={13} className="text-sky-400" />{label}</span>
        <span className="font-mono text-slate-200">{fmt ? fmt(value) : value}{unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-sky-400" />
    </label>
  );
}

export default function Controls({ form, scenarios, activePreset, onChange, onPreset }: Props) {
  const handleRegime = (r: Regime) => onChange({ weather_regime: r, historical_regime_bust_rate: REGIME_DEFAULT_RATE[r] });
  const zone = form.lead_time <= 3 ? "nowcast window" : form.lead_time <= 6 ? "short–medium range" : "chaos-dominated";

  return (
    <div className="space-y-5">
      <div>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">Preset scenarios</h2>
        <div className="grid grid-cols-2 gap-2">
          {scenarios.map((s, i) => {
            const Icon = PRESET_ICONS[i % PRESET_ICONS.length];
            return (
              <button key={s.id} onClick={() => onPreset(s)} title={s.description}
                className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-left text-[11px] leading-tight transition
                ${activePreset === s.id ? "border-sky-500 bg-sky-500/10 text-sky-200" : "border-slate-700 bg-slate-800/60 text-slate-300 hover:border-slate-500"}`}>
                <Icon size={15} className="shrink-0" />{s.name}
              </button>
            );
          })}
        </div>
      </div>

      <div className="rounded-lg border border-slate-700 bg-slate-800/70 p-3">
        <div className="flex items-center gap-3">
          <CalendarDays className="text-sky-400" size={22} />
          <div>
            <div className="text-2xl font-bold leading-none">Day {form.lead_time}</div>
            <div className="text-[11px] text-slate-400">{zone}</div>
          </div>
        </div>
        <input type="range" min={1} max={10} step={1} value={form.lead_time}
          onChange={(e) => onChange({ lead_time: parseInt(e.target.value, 10) })}
          className="mt-3 w-full accent-sky-400" />
        <div className="mt-1 flex justify-between px-0.5 text-[10px] text-slate-500">
          {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((d) => <span key={d}>{d}</span>)}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3">
        <label className="block">
          <span className="mb-1.5 flex items-center gap-1.5 text-xs text-slate-400"><MapPin size={13} className="text-sky-400" />Region</span>
          <select value={form.region} onChange={(e) => onChange({ region: e.target.value as Region })}
            className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm focus:border-sky-500 focus:outline-none">
            {REGIONS.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
        </label>
        <label className="block">
          <span className="mb-1.5 flex items-center gap-1.5 text-xs text-slate-400"><CloudRain size={13} className="text-sky-400" />Weather regime</span>
          <select value={form.weather_regime} onChange={(e) => handleRegime(e.target.value as Regime)}
            className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm focus:border-sky-500 focus:outline-none">
            {REGIMES.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
          <span className="mt-1 block text-[10px] text-slate-500">Climatological bust rate auto-fills on regime change</span>
        </label>
      </div>

      <div className="space-y-4 rounded-lg border border-slate-700 bg-slate-800/70 p-3">
        <SliderField label="Rainfall anomaly" unit=" mm" icon={CloudRain} min={0} max={250} step={1} value={form.rainfall_anomaly_mm} onChange={(v) => onChange({ rainfall_anomaly_mm: v })} />
        <SliderField label="Temperature anomaly" unit=" °C" icon={Thermometer} min={-8} max={10} step={0.1} value={form.temp_anomaly_c} onChange={(v) => onChange({ temp_anomaly_c: v })} fmt={(v) => v.toFixed(1)} />
        <SliderField label="Wind shear anomaly" unit=" kt" icon={Wind} min={-35} max={45} step={1} value={form.wind_shear_anomaly_kt} onChange={(v) => onChange({ wind_shear_anomaly_kt: v })} />
        <SliderField label="500 hPa geopotential error" unit=" gpm" icon={Mountain} min={-120} max={160} step={1} value={form.geopotential_500hpa_err} onChange={(v) => onChange({ geopotential_500hpa_err: v })} />
        <SliderField label="Historical regime bust rate" unit="%" icon={History} min={0} max={1} step={0.01} value={form.historical_regime_bust_rate} onChange={(v) => onChange({ historical_regime_bust_rate: v })} fmt={(v) => (v * 100).toFixed(0)} />
      </div>
    </div>
  );
}