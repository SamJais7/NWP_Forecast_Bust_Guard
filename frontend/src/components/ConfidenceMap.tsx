import { MapContainer, Polygon, TileLayer, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import type { PredictionSummary, Region } from "../types";
import { REGIONS, riskColor } from "../constants";

const REGION_SHAPES: Record<Region, [number, number][]> = {
  "NW India": [[24.0, 69.5], [24.5, 75.5], [27.0, 77.0], [30.5, 76.8], [32.5, 74.5], [32.0, 72.0], [30.0, 70.5], [27.5, 69.0]],
  "Indo-Gangetic Plains": [[29.0, 74.5], [28.5, 80.0], [27.5, 84.0], [26.0, 88.0], [24.5, 88.3], [24.3, 84.0], [24.8, 79.0], [26.0, 75.5]],
  "Bay of Bengal Coast": [[21.5, 87.0], [19.5, 85.5], [17.5, 83.5], [16.0, 81.5], [13.5, 80.3], [11.5, 79.9], [10.3, 80.2], [10.5, 81.5], [13.0, 82.3], [15.5, 84.5], [18.5, 86.5], [21.0, 88.0]],
  "Central India": [[21.0, 73.5], [20.5, 78.0], [21.5, 81.0], [21.0, 84.0], [23.5, 84.5], [24.5, 82.0], [24.8, 78.5], [24.5, 75.0], [23.0, 73.8]],
  "Southern Peninsula": [[15.5, 74.0], [15.0, 78.0], [13.5, 80.3], [11.0, 78.0], [8.3, 77.0], [8.5, 75.5], [10.5, 74.5], [13.0, 73.8], [14.5, 74.0]],
};

interface Props { results: Record<string, PredictionSummary | null>; selected: Region; onSelect: (r: Region) => void; }

export default function ConfidenceMap({ results, selected, onSelect }: Props) {
  return (
    <div className="relative h-[460px] overflow-hidden rounded-xl border border-slate-800">
      <MapContainer center={[21.5, 79.5]} zoom={5} minZoom={4} maxZoom={7} scrollWheelZoom
        style={{ height: "100%", width: "100%" }}>
        <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        {REGIONS.map((r) => {
          const res = results[r] ?? null;
          const p = res ? res.bust_probability : null;
          const c = p == null ? "#64748b" : riskColor(p);
          return (
            <Polygon key={r} positions={REGION_SHAPES[r]} eventHandlers={{ click: () => onSelect(r) }}
              pathOptions={{ color: c, weight: selected === r ? 3.5 : 1.5, fillColor: c,
                             fillOpacity: selected === r ? 0.55 : 0.35, opacity: 0.9 }}>
              <Tooltip direction="top" opacity={1}>
                <b>{r}</b>
                <br />
                {p != null ? `Bust risk ${(p * 100).toFixed(0)}% · ${res!.risk_level}` : "predicting…"}
              </Tooltip>
            </Polygon>
          );
        })}
      </MapContainer>
      <div className="pointer-events-none absolute right-3 top-3 z-[1000] rounded-lg border border-slate-700 bg-slate-900/90 px-3 py-2 text-[11px] leading-5 text-slate-300 shadow-lg">
        <div className="mb-1 font-semibold text-slate-100">Regional bust risk</div>
        <div className="flex items-center gap-2"><span className="inline-block h-2.5 w-2.5 rounded-sm bg-emerald-500" />&lt; 30% (high conf.)</div>
        <div className="flex items-center gap-2"><span className="inline-block h-2.5 w-2.5 rounded-sm bg-amber-500" />30 – 60%</div>
        <div className="flex items-center gap-2"><span className="inline-block h-2.5 w-2.5 rounded-sm bg-rose-500" />&ge; 60% (low conf.)</div>
        <div className="mt-1 text-[10px] text-slate-500">click a region to select it</div>
      </div>
    </div>
  );
}