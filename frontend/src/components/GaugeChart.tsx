interface Props { probability: number; riskLevel: string; confidenceLevel: string; latencyMs: number | null; }

const W = 360, H = 240, CX = 180, CY = 160, R = 140;
const polar = (r: number, deg: number) => ({ x: CX + r * Math.cos((deg * Math.PI) / 180), y: CY + r * Math.sin((deg * Math.PI) / 180) });
const arc = (r: number, a0: number, a1: number) => {
  const s = polar(r, a0), e = polar(r, a1);
  return `M ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${r} ${r} 0 0 1 ${e.x.toFixed(2)} ${e.y.toFixed(2)}`;
};

export default function GaugeChart({ probability, riskLevel, confidenceLevel, latencyMs }: Props) {
  const p = Math.min(1, Math.max(0, probability));
  const confPill = confidenceLevel === "HIGH"
    ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-300"
    : confidenceLevel === "MEDIUM" ? "border-amber-500/40 bg-amber-500/15 text-amber-300" : "border-rose-500/40 bg-rose-500/15 text-rose-300";
  const riskPill = { LOW: "border-emerald-500/40 bg-emerald-500/15 text-emerald-300", MODERATE: "border-sky-500/40 bg-sky-500/15 text-sky-300",
                     HIGH: "border-amber-500/40 bg-amber-500/15 text-amber-300", SEVERE: "border-rose-500/40 bg-rose-500/15 text-rose-300" }[riskLevel] ?? "";

  return (
    <div className="flex flex-col items-center">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-sm">
        {/* zones: green <30%, amber 30-60%, red >=60% (spec §6 confidence bands) */}
        <path d={arc(R, 180, 234)} stroke="#22c55e" strokeWidth={16} fill="none" />
        <path d={arc(R, 234, 288)} stroke="#f59e0b" strokeWidth={16} fill="none" />
        <path d={arc(R, 288, 360)} stroke="#ef4444" strokeWidth={16} fill="none" />
        {[0, 0.3, 0.6, 1].map((t) => {
          const a = 180 + t * 180, p1 = polar(R - 26, a), p2 = polar(R - 14, a), pl = polar(R - 46, a);
          return (
            <g key={t}>
              <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke="#475569" strokeWidth={2} />
              <text x={pl.x} y={pl.y + 4} textAnchor="middle" fontSize={11} fill="#64748b">{t * 100}%</text>
            </g>
          );
        })}
        <g style={{ transform: `rotate(${p * 180 - 90}deg)`, transformOrigin: `${CX}px ${CY}px`, transition: "transform 0.4s cubic-bezier(.4,0,.2,1)" }}>
          <line x1={CX} y1={CY - 26} x2={CX} y2={CY - (R - 32)} stroke="#f8fafc" strokeWidth={3.5} strokeLinecap="round" />
        </g>
        <circle cx={CX} cy={CY} r={9} fill="#0f172a" stroke="#94a3b8" strokeWidth={2.5} />
        <text x={CX} y={CY + 38} textAnchor="middle" fontSize={46} fontWeight={700} fill="#f1f5f9">{(p * 100).toFixed(0)}%</text>
        <text x={CX} y={CY + 62} textAnchor="middle" fontSize={11} fill="#94a3b8" letterSpacing={2}>BUST PROBABILITY</text>
      </svg>
      <div className="mt-1 flex flex-wrap items-center justify-center gap-2 text-[11px] font-medium">
        <span className={`rounded-full border px-3 py-1 ${riskPill}`}>Risk: {riskLevel}</span>
        <span className={`rounded-full border px-3 py-1 ${confPill}`}>Forecast confidence: {confidenceLevel}</span>
        {latencyMs != null && <span className="rounded-full border border-slate-600 bg-slate-800 px-3 py-1 text-slate-400">⚡ {latencyMs.toFixed(0)} ms</span>}
      </div>
    </div>
  );
}
