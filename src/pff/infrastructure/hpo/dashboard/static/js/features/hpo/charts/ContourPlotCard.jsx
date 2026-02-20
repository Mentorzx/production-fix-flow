/**
 * Provide ContourPlotCard module functionality for the HPO dashboard.
 */

import { useMemo } from "react";
import { Card, Layers, ChartFrame, WithData, colors } from "../../../ui/BaseComponents.jsx";
import { renderParamWithHints } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

const toNumber = (value) => {
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : null;
};

const pickAxisKeys = (numericKeys) => {
  const lowerMap = numericKeys.map((k) => ({ raw: k, lower: String(k).toLowerCase() }));
  const pick = (candidates) => {
    for (const key of candidates) {
      const found = lowerMap.find((k) => k.lower === key || k.lower.includes(key));
      if (found) return found.raw;
    }
    return null;
  };
  const xKey = pick(["learning_rate", "lr"]) || numericKeys[0] || null;
  const yKey =
    pick(["embedding", "embed", "hidden"]) || numericKeys.find((k) => k !== xKey) || null;
  return { xKey, yKey };
};

const lerp = (a, b, t) => a + (b - a) * t;

const hexToRgb = (hex) => {
  const clean = hex.replace("#", "");
  const bigint = parseInt(clean, 16);
  return {
    r: (bigint >> 16) & 255,
    g: (bigint >> 8) & 255,
    b: bigint & 255,
  };
};

const lerpColor = (a, b, t) => {
  const c1 = hexToRgb(a);
  const c2 = hexToRgb(b);
  const r = Math.round(lerp(c1.r, c2.r, t));
  const g = Math.round(lerp(c1.g, c2.g, t));
  const bVal = Math.round(lerp(c1.b, c2.b, t));
  return `rgb(${r}, ${g}, ${bVal})`;
};

/**
 * Expose contour plot card for dashboard usage.
 */
export const ContourPlotCard = ({ trials }) => {
  const contour = useMemo(() => {
    const items = Array.isArray(trials) ? trials : [];
    const completed = items.filter((t) => t?.state === "COMPLETE" && t?.params && t?.value != null);

    if (completed.length < 6) {
      return { points: [], grid: [], xKey: null, yKey: null };
    }

    const keyStats = {};
    completed.forEach((t) => {
      Object.entries(t.params || {}).forEach(([key, value]) => {
        const num = toNumber(value);
        if (num == null) return;
        if (!keyStats[key]) keyStats[key] = { count: 0 };
        keyStats[key].count += 1;
      });
    });

    const numericKeys = Object.entries(keyStats)
      .filter(([, meta]) => meta.count >= completed.length * 0.7)
      .map(([key]) => key);

    if (numericKeys.length < 2) {
      return { points: [], grid: [], xKey: null, yKey: null };
    }

    const { xKey, yKey } = pickAxisKeys(numericKeys);
    if (!xKey || !yKey || xKey === yKey) {
      return { points: [], grid: [], xKey: null, yKey: null };
    }

    const points = completed
      .map((t) => {
        const x = toNumber(t.params?.[xKey]);
        const y = toNumber(t.params?.[yKey]);
        const z = toNumber(t.value);
        if (x == null || y == null || z == null) return null;
        return { x, y, z };
      })
      .filter(Boolean);

    if (points.length < 6) {
      return { points: [], grid: [], xKey, yKey };
    }

    const xVals = points.map((p) => p.x);
    const yVals = points.map((p) => p.y);
    const zVals = points.map((p) => p.z);

    const xMin = Math.min(...xVals);
    const xMax = Math.max(...xVals);
    const yMin = Math.min(...yVals);
    const yMax = Math.max(...yVals);
    const zMin = Math.min(...zVals);
    const zMax = Math.max(...zVals);

    const xBins = 16;
    const yBins = 12;
    const grid = Array.from({ length: yBins }, () =>
      Array.from({ length: xBins }, () => ({ sum: 0, count: 0 }))
    );

    points.forEach((p) => {
      const nx = xMax === xMin ? 0.5 : (p.x - xMin) / (xMax - xMin);
      const ny = yMax === yMin ? 0.5 : (p.y - yMin) / (yMax - yMin);
      const xi = Math.min(xBins - 1, Math.max(0, Math.floor(nx * (xBins - 1))));
      const yi = Math.min(yBins - 1, Math.max(0, Math.floor(ny * (yBins - 1))));
      grid[yi][xi].sum += p.z;
      grid[yi][xi].count += 1;
    });

    const cells = grid.map((row, yIdx) =>
      row.map((cell, xIdx) => {
        const value = cell.count > 0 ? cell.sum / cell.count : null;
        return { xIdx, yIdx, value };
      })
    );

    return {
      points,
      grid: cells,
      xKey,
      yKey,
      xMin,
      xMax,
      yMin,
      yMax,
      zMin,
      zMax,
      xBins,
      yBins,
    };
  }, [trials]);

  const hasData = contour.points.length > 0 && contour.grid.length > 0;
  const helpText = ChartRegistry.get("contour");

  return (
    <Card title="Contour Plot" className="h-full" icon={Layers} helpText={helpText}>
      <ChartFrame className="flex flex-col h-full gap-3">
        <WithData
          when={hasData}
          empty="Dados insuficientes para superfície 3D"
          emptyClassName="text-zinc-500"
        >
          <div className="flex items-center justify-between text-[10px] text-zinc-400">
            <div className="flex items-center gap-2">
              <span className="uppercase tracking-widest">Eixo X</span>
              {renderParamWithHints(contour.xKey || "-")}
            </div>
            <div className="flex items-center gap-2">
              <span className="uppercase tracking-widest">Eixo Y</span>
              {renderParamWithHints(contour.yKey || "-")}
            </div>
            <div className="flex items-center gap-2">
              <span className="uppercase tracking-widest">Z</span>
              <span className="text-zinc-300">Score</span>
            </div>
          </div>

          <div className="flex-1 min-h-0 grid gap-3" style={{ gridTemplateRows: "1fr auto" }}>
            <div className="relative h-full min-h-[220px] border border-zinc-800 rounded-lg bg-zinc-950/40 p-2">
              <div
                className="w-full h-full grid"
                style={{
                  gridTemplateColumns: `repeat(${contour.xBins}, minmax(0, 1fr))`,
                  gridTemplateRows: `repeat(${contour.yBins}, minmax(0, 1fr))`,
                }}
              >
                {contour.grid.flat().map((cell) => {
                  const t =
                    cell.value == null || contour.zMax === contour.zMin
                      ? 0
                      : (cell.value - contour.zMin) / (contour.zMax - contour.zMin);
                  const fill =
                    cell.value == null
                      ? "rgba(39,39,42,0.35)"
                      : lerpColor(colors.orange, colors.lime, Math.min(1, Math.max(0, t)));
                  const title = cell.value == null ? "Sem dados" : `score=${cell.value.toFixed(4)}`;
                  return (
                    <div
                      key={`${cell.xIdx}-${cell.yIdx}`}
                      title={title}
                      className="border border-zinc-900/60"
                      style={{ backgroundColor: fill }}
                    ></div>
                  );
                })}
              </div>
            </div>

            <div className="flex items-center justify-between text-[10px] text-zinc-500 font-mono">
              <div className="flex items-center gap-3">
                <span>{contour.xMin?.toFixed(3)}</span>
                <span className="uppercase tracking-widest text-zinc-600">{contour.xKey}</span>
                <span>{contour.xMax?.toFixed(3)}</span>
              </div>
              <div className="flex items-center gap-2">
                <span>{contour.yMin?.toFixed(3)}</span>
                <span className="uppercase tracking-widest text-zinc-600">{contour.yKey}</span>
                <span>{contour.yMax?.toFixed(3)}</span>
              </div>
              <div className="flex items-center gap-2">
                <span>{contour.zMin?.toFixed(3)}</span>
                <span
                  className="h-1 w-24 rounded-full"
                  style={{ background: `linear-gradient(90deg, ${colors.orange}, ${colors.lime})` }}
                ></span>
                <span>{contour.zMax?.toFixed(3)}</span>
              </div>
            </div>
          </div>
        </WithData>
      </ChartFrame>
    </Card>
  );
};
