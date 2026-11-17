import { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { GROUP_COLORS, GROUP_LABELS, type ActionGroup, type Lane } from "../../api/actions";

/** Action frequency per category, and offensive / defensive / movement distribution per fighter. */
export default function ActionCharts({ lanes }: { lanes: Lane[] }) {
  const freq = useMemo(() => {
    const m = new Map<string, { label: string; group: ActionGroup; count: number }>();
    lanes.forEach((l) => l.actions.filter((a) => a.review_status !== "rejected").forEach((a) => {
      const e = m.get(a.category) ?? { label: a.label, group: a.group, count: 0 };
      e.count++;
      m.set(a.category, e);
    }));
    return [...m.values()].sort((a, b) => b.count - a.count);
  }, [lanes]);

  const dist = useMemo(() => lanes.filter((l) => l.actions.length).map((l) => {
    const c: Record<string, number> = { offensive: 0, defensive: 0, movement: 0, unknown: 0 };
    l.actions.filter((a) => a.review_status !== "rejected").forEach((a) => c[a.group]++);
    return { name: l.name, ...c };
  }), [lanes]);

  const tip = { contentStyle: { background: "#18181d", border: "1px solid #33333c", borderRadius: 6, fontSize: 12 }, labelStyle: { color: "#d6d6dc" }, itemStyle: { color: "#f1f1f4" }, cursor: { fill: "rgba(255,255,255,0.04)" } };
  if (!freq.length) return <p className="px-5 py-8 text-center text-sm text-ink-500">No actions to chart yet.</p>;
  return (
    <div className="grid gap-4 p-4 lg:grid-cols-2">
      <div>
        <div className="label-caps mb-2">Action frequency</div>
        <div style={{ height: Math.max(160, freq.length * 30 + 30) }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={freq} layout="vertical" margin={{ left: 10, right: 16, top: 0, bottom: 0 }}>
              <CartesianGrid horizontal={false} stroke="#26262d" />
              <XAxis type="number" allowDecimals={false} tick={{ fill: "#75757f", fontSize: 11 }} tickLine={false} axisLine={{ stroke: "#33333c" }} />
              <YAxis type="category" dataKey="label" width={150} tick={{ fill: "#d6d6dc", fontSize: 11 }} tickLine={false} axisLine={false} />
              <Tooltip {...tip} />
              <Bar dataKey="count" name="Actions" radius={[0, 4, 4, 0]} maxBarSize={18}>
                {freq.map((f) => <Cell key={f.label} fill={GROUP_COLORS[f.group]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div>
        <div className="label-caps mb-2">Offensive vs defensive (per fighter)</div>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={dist} margin={{ left: -16, right: 8, top: 0, bottom: 0 }}>
              <CartesianGrid vertical={false} stroke="#26262d" />
              <XAxis dataKey="name" tick={{ fill: "#d6d6dc", fontSize: 11 }} tickLine={false} axisLine={{ stroke: "#33333c" }} />
              <YAxis allowDecimals={false} tick={{ fill: "#75757f", fontSize: 11 }} tickLine={false} axisLine={false} />
              <Tooltip {...tip} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {(Object.keys(GROUP_LABELS) as ActionGroup[]).map((g) => (
                <Bar key={g} dataKey={g} name={GROUP_LABELS[g]} stackId="a" fill={GROUP_COLORS[g]} maxBarSize={48} stroke="#131317" strokeWidth={1} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
