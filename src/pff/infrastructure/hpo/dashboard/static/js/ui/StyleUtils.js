// @ts-check

/**
 * Build normalized sparkline points from scalar history.
 * @param {Array<number | null | undefined>} values
 * @returns {{x:number,y:number}[]}
 */
export const buildSparklineFromScalars = (values) => {
  if (!Array.isArray(values) || values.length === 0) return [];
  return values
    .map((v, idx) => ({ x: idx, y: typeof v === "number" ? v : null }))
    .filter((p) => p.y != null);
};

/**
 * Extract points from an object series using a metric key.
 * @param {Array<Record<string, any>>} history
 * @param {string} metricKey
 * @returns {{x:number,y:number}[]}
 */
export const extractSparklineData = (history, metricKey) => {
  if (!Array.isArray(history) || history.length === 0 || !metricKey) return [];
  return history
    .map((item, idx) => {
      const raw = item?.[metricKey];
      const y = typeof raw === "number" ? raw : Number(raw);
      return Number.isFinite(y) ? { x: idx, y } : null;
    })
    .filter((p) => p != null);
};

/**
 * Generate compact SVG path for sparkline.
 * @param {{x:number,y:number}[]} points
 * @param {number} width
 * @param {number} height
 * @returns {string}
 */
export const buildSparklinePath = (points, width = 100, height = 20) => {
  if (!Array.isArray(points) || points.length < 2) return "";

  let min = Infinity;
  let max = -Infinity;
  for (const point of points) {
    if (point.y < min) min = point.y;
    if (point.y > max) max = point.y;
  }

  const range = max - min || 1;
  const stepX = width / Math.max(1, points.length - 1);
  const padY = 1;
  const usableHeight = Math.max(1, height - 2 * padY);

  return points
    .map((point, idx) => {
      const x = idx * stepX;
      const y = padY + ((max - point.y) / range) * usableHeight;
      const cmd = idx === 0 ? "M" : "L";
      return `${cmd}${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
};

/**
 * Compute directional delta between two numeric values.
 * @param {number | null | undefined} current
 * @param {number | null | undefined} previous
 * @returns {{trend:'up'|'down'|'neutral',diff:string}}
 */
export const calculateTrend = (current, previous) => {
  if (!Number.isFinite(current) || !Number.isFinite(previous)) {
    return { trend: "neutral", diff: "—" };
  }

  const delta = current - previous;
  const absDelta = Math.abs(delta);
  if (absDelta < 1e-9) {
    return { trend: "neutral", diff: "0.00" };
  }

  return {
    trend: delta > 0 ? "up" : "down",
    diff: absDelta.toFixed(2),
  };
};
