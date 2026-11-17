import { useEffect, useState, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { Activity, Film, LayoutDashboard, Menu, Network, ScanEye, Sparkles, Settings, Swords, Upload, Users, X } from "lucide-react";
import { api } from "../api/client";
import { useUploads } from "../context/UploadContext";
import { cn } from "../lib/format";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/videos", label: "Video Library", icon: Film },
  { to: "/upload", label: "Upload Videos", icon: Upload },
  { to: "/fighters", label: "Fighters", icon: Users },
  { to: "/tracking", label: "Fighter Tracking", icon: ScanEye },
  { to: "/actions", label: "Action Recognition", icon: Swords },
  { to: "/patterns", label: "Pattern Intelligence", icon: Network },
  { to: "/predictions", label: "Next-Move Prediction", icon: Sparkles },
  { to: "/jobs", label: "Processing Jobs", icon: Activity },
  { to: "/settings", label: "Settings", icon: Settings },
];

function Logo() {
  return (
    <div className="flex items-center gap-3 px-5 py-5">
      <svg viewBox="0 0 32 32" className="size-8 shrink-0" aria-hidden>
        <path d="M16 3 27.3 9.5v13L16 29 4.7 22.5v-13z" fill="none" stroke="#e11d2e" strokeWidth="2.2" />
        <path d="M16 9.5 21.6 12.7v6.6L16 22.5l-5.6-3.2v-6.6z" fill="#e11d2e" opacity="0.22" />
        <circle cx="16" cy="16" r="2.6" fill="#fff" />
      </svg>
      <div className="leading-tight">
        <div className="font-display text-lg font-bold uppercase tracking-wider text-white">Fight Intel</div>
        <div className="text-[10px] font-medium uppercase tracking-[0.2em] text-ink-400">Phase 5 · Prediction</div>
      </div>
    </div>
  );
}

function ApiStatus() {
  const [state, setState] = useState<"ok" | "down" | "checking">("checking");
  const [queue, setQueue] = useState(0);
  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const res = await fetch("/api/health");
        const j = await res.json();
        if (!alive) return;
        setState(res.ok ? "ok" : "down");
        setQueue(j.queue_size ?? 0);
      } catch {
        if (alive) setState("down");
      }
    };
    void check();
    const t = window.setInterval(check, 10000);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, []);
  return (
    <div className="flex items-center gap-2 text-xs text-ink-400">
      <span
        className={cn(
          "size-2 rounded-full",
          state === "ok" && "bg-emerald-400",
          state === "down" && "bg-blood-500",
          state === "checking" && "bg-ink-500",
        )}
      />
      {state === "ok" ? `API online${queue ? ` · ${queue} queued` : ""}` : state === "down" ? "API offline" : "Connecting…"}
    </div>
  );
}

export default function Layout({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const { activeCount } = useUploads();
  const location = useLocation();
  useEffect(() => setOpen(false), [location.pathname]);
  // warm the settings endpoint so failures show early
  useEffect(() => {
    api.getSettings().catch(() => undefined);
  }, []);

  const nav = (
    <nav className="flex flex-1 flex-col gap-0.5 px-3">
      {NAV.map(({ to, label, icon: Icon, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          className={({ isActive }) =>
            cn(
              "group relative flex items-center gap-3 rounded-md px-3 py-2.5 text-sm transition-colors",
              isActive ? "bg-ink-800 text-white" : "text-ink-300 hover:bg-ink-850 hover:text-white",
            )
          }
        >
          {({ isActive }) => (
            <>
              {isActive && <span className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-blood-500" />}
              <Icon className={cn("size-4", isActive ? "text-blood-400" : "text-ink-400 group-hover:text-ink-200")} />
              <span className="flex-1">{label}</span>
              {to === "/upload" && activeCount > 0 && (
                <span className="rounded-full bg-blood-500 px-1.5 text-[10px] font-semibold text-white tabular">{activeCount}</span>
              )}
            </>
          )}
        </NavLink>
      ))}
    </nav>
  );

  return (
    <div className="flex min-h-full">
      {/* Desktop sidebar */}
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col border-r border-ink-700/70 bg-ink-950 lg:flex">
        <Logo />
        <div className="label-caps px-6 pb-2 pt-3">Navigation</div>
        {nav}
        <div className="border-t border-ink-700/70 px-6 py-4">
          <ApiStatus />
        </div>
      </aside>

      {/* Mobile sidebar */}
      {open && (
        <div className="fixed inset-0 z-40 bg-black/70 lg:hidden" onClick={() => setOpen(false)}>
          <aside className="flex h-full w-64 flex-col border-r border-ink-700 bg-ink-950" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between pr-3">
              <Logo />
              <button className="btn-ghost p-2" onClick={() => setOpen(false)} aria-label="Close menu">
                <X className="size-5" />
              </button>
            </div>
            {nav}
            <div className="border-t border-ink-700/70 px-6 py-4">
              <ApiStatus />
            </div>
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex items-center gap-3 border-b border-ink-700/70 bg-ink-900/90 px-4 py-3 backdrop-blur lg:hidden">
          <button className="btn-ghost p-2" onClick={() => setOpen(true)} aria-label="Open menu">
            <Menu className="size-5" />
          </button>
          <span className="font-display text-lg font-bold uppercase tracking-wider">Fight Intel</span>
        </header>
        <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6 sm:px-6 lg:px-10 lg:py-8">{children}</main>
      </div>
    </div>
  );
}
