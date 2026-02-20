const trimZeros = (value) => value.replace(/\.?0+$/, "");

export const formatCompactTick = (value) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  const abs = Math.abs(n);
  if (abs >= 1000) return n.toExponential(1);
  if (abs >= 1) return trimZeros(n.toFixed(3));
  if (abs > 0) return trimZeros(n.toFixed(4));
  return "0";
};
