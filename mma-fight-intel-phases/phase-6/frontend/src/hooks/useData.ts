import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Load data and optionally keep it fresh by polling.
 * `pollMs` may be a function of the current data so polling only runs while something is active.
 */
export function useData<T>(
  loader: () => Promise<T>,
  deps: unknown[],
  pollMs?: number | ((data: T | undefined) => number | undefined),
) {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<string>();
  const [loading, setLoading] = useState(true);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;
  const dataRef = useRef<T | undefined>(undefined);

  const reload = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const d = await loaderRef.current();
      dataRef.current = d;
      setData(d);
      setError(undefined);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    if (!pollMs) return;
    let timer: number | undefined;
    let cancelled = false;
    const tick = async () => {
      const ms = typeof pollMs === "function" ? pollMs(dataRef.current) : pollMs;
      if (cancelled) return;
      if (ms) {
        timer = window.setTimeout(async () => {
          if (document.visibilityState === "visible") await reload(true);
          void tick();
        }, ms);
      } else {
        timer = window.setTimeout(tick, 1500); // re-evaluate later
      }
    };
    void tick();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reload, ...deps]);

  return { data, error, loading, reload, setData };
}
