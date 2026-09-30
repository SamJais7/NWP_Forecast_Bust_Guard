import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ShapContribution } from "../types";

export default function ShapWaterfall({ contributions }: { contributions: ShapContribution[] }) {
  const data = [...contributions].filter((c) => c.active)
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 10);
  if (!data.length) return <div className="py-10 text-center text-sm text-slate-500">No contributions to display</div>;
  return (
    <div className="h-[380px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 4 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" stroke="#64748b" fontSize={11} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}pp`} />
          <YAxis type="category" dataKey="label" width={210} fontSize={11} tick={{ fill: "#cbd5e1" }} />
          <Tooltip cursor={{ fill: "rgba(148,163,184,0.08)" }} contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }}
            formatter={(v: number) => [`${(v * 100).toFixed(1)} pp on bust probability`, "contribution"]} labelFormatter={() => ""} />
          <ReferenceLine x={0} stroke="#475569" />
          <Bar dataKey="contribution" radius={[0, 4, 4, 0]} barSize={14}>
            {data.map((d, i) => <Cell key={i} fill={d.contribution >= 0 ? "#f43f5e" : "#10b981"} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}