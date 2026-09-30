import { CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function LeadTimeCurve({ data, selectedDay }: { data: { day: number; p: number }[]; selectedDay: number }) {
  return (
    <div className="h-[260px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 24, left: 0, bottom: 4 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
          <XAxis dataKey="day" stroke="#64748b" fontSize={11} tickFormatter={(d: number) => `D${d}`} />
          <YAxis domain={[0, 1]} stroke="#64748b" fontSize={11} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
          <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }}
            formatter={(v: number) => [`${(v * 100).toFixed(1)}% bust risk`, "Day " + (data.find((d) => d.p === v)?.day ?? "")]}
            labelFormatter={(d) => `Day ${d}`} />
          <ReferenceArea y1={0.6} y2={1} fill="#ef4444" fillOpacity={0.06} />
          <ReferenceArea y1={0.3} y2={0.6} fill="#f59e0b" fillOpacity={0.06} />
          <ReferenceLine x={selectedDay} stroke="#38bdf8" strokeDasharray="4 4" />
          <Line type="monotone" dataKey="p" stroke="#38bdf8" strokeWidth={2.5} dot={{ r: 3, fill: "#38bdf8" }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}