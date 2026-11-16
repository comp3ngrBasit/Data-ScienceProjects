import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import { CircleAlert, CircleCheck, Info, X } from "lucide-react";
import { cn } from "../lib/format";

type Kind = "success" | "error" | "info";
interface Toast {
  id: number;
  kind: Kind;
  message: string;
}

const Ctx = createContext<(kind: Kind, message: string) => void>(() => {});

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const dismiss = (id: number) => setToasts((t) => t.filter((x) => x.id !== id));
  const push = useCallback((kind: Kind, message: string) => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t.slice(-3), { id, kind, message }]);
    window.setTimeout(() => dismiss(id), kind === "error" ? 7000 : 3500);
  }, []);

  return (
    <Ctx.Provider value={push}>
      {children}
      <div className="pointer-events-none fixed right-4 bottom-4 z-[100] flex w-[min(380px,calc(100vw-2rem))] flex-col gap-2">
        {toasts.map((t) => {
          const Icon = t.kind === "success" ? CircleCheck : t.kind === "error" ? CircleAlert : Info;
          return (
            <div
              key={t.id}
              role="status"
              className={cn(
                "pointer-events-auto flex items-start gap-3 rounded-lg border bg-ink-800 px-4 py-3 text-sm shadow-lg shadow-black/40",
                t.kind === "error" ? "border-blood-700/70" : "border-ink-600",
              )}
            >
              <Icon
                className={cn(
                  "mt-0.5 size-4 shrink-0",
                  t.kind === "success" && "text-emerald-400",
                  t.kind === "error" && "text-blood-400",
                  t.kind === "info" && "text-ink-300",
                )}
              />
              <span className="flex-1 text-ink-100">{t.message}</span>
              <button onClick={() => dismiss(t.id)} className="text-ink-400 hover:text-ink-100" aria-label="Dismiss">
                <X className="size-4" />
              </button>
            </div>
          );
        })}
      </div>
    </Ctx.Provider>
  );
}

export const useToast = () => useContext(Ctx);
