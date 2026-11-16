import { useEffect, useState } from "react";
import { actionsApi, GROUP_COLORS, type ActionGroup, type Category } from "../../api/actions";

let cache: Promise<{ categories: Category[]; active_model: string | null }> | null = null;

export function useTaxonomy() {
  const [cats, setCats] = useState<Category[]>([]);
  useEffect(() => {
    cache = cache ?? actionsApi.taxonomy();
    cache.then((d) => setCats(d.categories)).catch(() => {
      cache = null;
    });
  }, []);
  return cats;
}

export function refreshTaxonomy() {
  cache = null;
}

/** Shape encodes the action family, colour the group (offensive / defensive / movement / unknown). */
export function Marker({ family, group, size = 12, title }: { family: string; group: ActionGroup; size?: number; title?: string }) {
  const c = GROUP_COLORS[group] ?? "#a3a3ad";
  const s = size;
  const hollow = group === "unknown";
  const common = { fill: hollow ? "none" : c, stroke: c, strokeWidth: hollow ? 1.8 : 0 };
  let shape;
  switch (family) {
    case "kick":
      shape = <polygon points={`${s / 2},1 ${s - 1},${s - 1} 1,${s - 1}`} {...common} />;
      break;
    case "knee_elbow":
      shape = <polygon points={`${s / 2},1 ${s - 1},${s / 2} ${s / 2},${s - 1} 1,${s / 2}`} {...common} />;
      break;
    case "grappling":
      shape = <rect x={1.5} y={1.5} width={s - 3} height={s - 3} {...common} />;
      break;
    case "defense":
      shape = <polygon points={`1,1 ${s - 1},1 ${s / 2},${s - 1}`} {...common} />;
      break;
    case "movement":
      shape = <rect x={1} y={s / 3} width={s - 2} height={s / 3} rx={1} {...common} />;
      break;
    default:
      shape = <circle cx={s / 2} cy={s / 2} r={s / 2 - 1} {...common} />;
  }
  return (
    <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`} className="shrink-0" role="img" aria-label={title}>
      {title && <title>{title}</title>}
      {shape}
    </svg>
  );
}

export const pct = (v: number | null | undefined, d = 0) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
export const fmtT = (t: number) => {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
};
