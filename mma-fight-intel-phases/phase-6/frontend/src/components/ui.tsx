import { useEffect, useRef, type ReactNode } from "react";
import { Film, LoaderCircle, X } from "lucide-react";
import type { JobStatus, VideoStatus } from "../api/types";
import { cn } from "../lib/format";

const STATUS_STYLES: Record<VideoStatus | JobStatus, { dot: string; text: string; label: string }> = {
  uploaded: { dot: "bg-ink-300", text: "text-ink-200 border-ink-600 bg-ink-800", label: "Uploaded" },
  queued: { dot: "bg-amber-400", text: "text-amber-200 border-amber-500/30 bg-amber-500/10", label: "Queued" },
  processing: { dot: "bg-sky-400 animate-pulse", text: "text-sky-200 border-sky-500/30 bg-sky-500/10", label: "Processing" },
  completed: { dot: "bg-emerald-400", text: "text-emerald-200 border-emerald-500/30 bg-emerald-500/10", label: "Completed" },
  failed: { dot: "bg-blood-400", text: "text-blood-300 border-blood-600/40 bg-blood-900/40", label: "Failed" },
  cancelled: { dot: "bg-ink-400", text: "text-ink-300 border-ink-600 bg-ink-800", label: "Cancelled" },
};

export function StatusBadge({ status, className }: { status: VideoStatus | JobStatus; className?: string }) {
  const s = STATUS_STYLES[status];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium tracking-wide",
        s.text,
        className,
      )}
    >
      <span className={cn("size-1.5 rounded-full", s.dot)} />
      {s.label}
    </span>
  );
}

export function ProgressBar({ value, className, tone = "red" }: { value: number; className?: string; tone?: "red" | "sky" | "green" }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div className={cn("h-1.5 w-full overflow-hidden rounded-full bg-ink-700", className)} role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
      <div
        className={cn(
          "h-full rounded-full transition-[width] duration-300",
          tone === "red" && "bg-blood-500",
          tone === "sky" && "bg-sky-400",
          tone === "green" && "bg-emerald-400",
        )}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <h1 className="font-display text-3xl font-bold uppercase tracking-wide text-white">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-ink-400">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return <LoaderCircle className={cn("size-4 animate-spin text-ink-300", className)} />;
}

export function LoadingBlock({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-16 text-sm text-ink-400">
      <Spinner /> {label}
    </div>
  );
}

export function ErrorBlock({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="panel flex flex-col items-center gap-3 px-6 py-10 text-center">
      <p className="text-sm text-blood-300">{message}</p>
      {onRetry && (
        <button className="btn-secondary" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function EmptyState({ icon, title, body, action }: { icon?: ReactNode; title: string; body?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-14 text-center">
      <div className="grid size-12 place-items-center rounded-full border border-ink-700 bg-ink-800 text-ink-400">
        {icon ?? <Film className="size-5" />}
      </div>
      <div>
        <p className="font-medium text-ink-100">{title}</p>
        {body && <p className="mt-1 max-w-sm text-sm text-ink-400">{body}</p>}
      </div>
      {action}
    </div>
  );
}

export function StatCard({
  label,
  value,
  hint,
  icon,
  accent,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  icon?: ReactNode;
  accent?: boolean;
}) {
  return (
    <div className={cn("panel relative overflow-hidden px-5 py-4", accent && "border-blood-700/50")}>
      {accent && <div className="absolute inset-y-0 left-0 w-0.5 bg-blood-500" />}
      <div className="flex items-center justify-between">
        <span className="label-caps">{label}</span>
        <span className="text-ink-500">{icon}</span>
      </div>
      <div className="mt-2 font-display text-4xl font-semibold leading-none text-white tabular">{value}</div>
      {hint && <div className="mt-2 text-xs text-ink-400">{hint}</div>}
    </div>
  );
}

export function Modal({
  open,
  onClose,
  title,
  children,
  footer,
  width = "max-w-lg",
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  footer?: ReactNode;
  width?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    ref.current?.querySelector<HTMLElement>("input,select,textarea,button")?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-[2px]" onMouseDown={onClose}>
      <div
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={cn("panel flex max-h-[90vh] w-full flex-col bg-ink-850 shadow-2xl shadow-black/60", width)}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="panel-header">
          <h2 className="font-display text-lg font-semibold uppercase tracking-wide">{title}</h2>
          <button className="btn-ghost -mr-2 p-1.5" onClick={onClose} aria-label="Close">
            <X className="size-4" />
          </button>
        </div>
        <div className="overflow-y-auto px-5 py-4">{children}</div>
        {footer && <div className="flex justify-end gap-2 border-t border-ink-700/80 px-5 py-3">{footer}</div>}
      </div>
    </div>
  );
}

export function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel = "Delete",
  busy,
  onConfirm,
  onClose,
}: {
  open: boolean;
  title: string;
  body: ReactNode;
  confirmLabel?: string;
  busy?: boolean;
  onConfirm: () => void;
  onClose: () => void;
}) {
  return (
    <Modal
      open={open}
      onClose={onClose}
      title={title}
      width="max-w-md"
      footer={
        <>
          <button className="btn-secondary" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button className="btn-primary" onClick={onConfirm} disabled={busy}>
            {busy && <Spinner className="text-white" />}
            {confirmLabel}
          </button>
        </>
      }
    >
      <div className="text-sm text-ink-300">{body}</div>
    </Modal>
  );
}

export function Field({ label, hint, children, className }: { label: string; hint?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <label className={cn("block", className)}>
      <span className="mb-1.5 block text-xs font-medium text-ink-300">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-ink-500">{hint}</span>}
    </label>
  );
}

export function Thumb({ url, alt, className, status }: { url: string | null; alt: string; className?: string; status?: VideoStatus }) {
  return (
    <div className={cn("relative overflow-hidden bg-ink-950", className)}>
      {url ? (
        <img src={url} alt={alt} loading="lazy" className="size-full object-cover" />
      ) : (
        <div className="grid size-full place-items-center text-ink-600">
          {status === "processing" || status === "queued" ? <Spinner className="size-5 text-ink-500" /> : <Film className="size-6" />}
        </div>
      )}
    </div>
  );
}

export function KV({ k, v, mono }: { k: string; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-ink-700/50 py-2 text-sm last:border-0">
      <dt className="text-ink-400">{k}</dt>
      <dd className={cn("text-right text-ink-100 tabular", mono && "font-mono text-xs break-all")}>{v ?? "—"}</dd>
    </div>
  );
}
