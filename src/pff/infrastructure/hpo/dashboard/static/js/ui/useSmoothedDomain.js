import { useEffect, useMemo, useRef, useState } from "react";

const EPSILON = 1e-6;

const asFiniteNumber = (value) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return value;
};

const computeTargetDomain = (values, { paddingRatio = 0.08, minSpan = 1e-3 } = {}) => {
  const numeric = values.map(asFiniteNumber).filter((v) => v != null);
  if (numeric.length === 0) return null;

  let min = Math.min(...numeric);
  let max = Math.max(...numeric);
  const span = Math.max(max - min, minSpan);
  const pad = span * paddingRatio;

  min -= pad;
  max += pad;

  if (Math.abs(max - min) < minSpan) {
    max = min + minSpan;
  }
  return [min, max];
};

const easeOutCubic = (t) => 1 - (1 - t) ** 3;
const isSameDomain = (a, b) => {
  if (!a || !b) return false;
  return Math.abs(a[0] - b[0]) < EPSILON && Math.abs(a[1] - b[1]) < EPSILON;
};

export const useSmoothedDomain = (
  values,
  { durationMs = 260, clampMin = null, clampMax = null, paddingRatio = 0.08, minSpan = 1e-3 } = {}
) => {
  const target = useMemo(
    () => computeTargetDomain(values, { paddingRatio, minSpan }),
    [values, paddingRatio, minSpan]
  );

  const [domain, setDomain] = useState(() => target);
  const domainRef = useRef(target);
  const rafRef = useRef(0);

  useEffect(() => {
    if (!target) return undefined;

    const from = domainRef.current ?? target;
    const start = performance.now();
    const deltaMin = target[0] - from[0];
    const deltaMax = target[1] - from[1];

    if (Math.abs(deltaMin) < EPSILON && Math.abs(deltaMax) < EPSILON) {
      domainRef.current = target;
      setDomain((prev) => (isSameDomain(prev, target) ? prev : target));
      return undefined;
    }

    const step = (now) => {
      const elapsed = now - start;
      const t = Math.max(0, Math.min(1, elapsed / durationMs));
      const eased = easeOutCubic(t);
      const next = [from[0] + deltaMin * eased, from[1] + deltaMax * eased];
      domainRef.current = next;
      setDomain((prev) => (isSameDomain(prev, next) ? prev : next));
      if (t < 1) {
        rafRef.current = requestAnimationFrame(step);
      }
    };

    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(step);

    return () => cancelAnimationFrame(rafRef.current);
  }, [target, durationMs]);

  const raw = domain ?? target;
  if (!raw) return [0, 1];

  const min = clampMin != null ? Math.max(clampMin, raw[0]) : raw[0];
  const max = clampMax != null ? Math.min(clampMax, raw[1]) : raw[1];
  if (max - min < EPSILON) return [min, min + 1];
  return [min, max];
};
