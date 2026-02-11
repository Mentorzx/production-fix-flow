import { useState, useEffect } from "react";

import { Card, Cpu, Activity } from "../../../ui/BaseComponents.jsx";
import { renderParamWithHints } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const BestTrialCard = ({ trial }) => {
  const [displayId, setDisplayId] = useState(trial?.id || 0);
  const Icon = Cpu || Activity || (() => null);

  useEffect(() => {
    let start = 0;
    const end = trial?.id || 0;
    if (start === end) return;
    let timer = null;
    const duration = 2000;
    const stepTime = 20;
    const steps = duration / stepTime;
    const increment = (end - start) / steps;
    let current = start;
    timer = setInterval(() => {
      current += increment;
      if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
        setDisplayId(end);
        clearInterval(timer);
      } else {
        setDisplayId(Math.floor(current));
      }
    }, stepTime);
    return () => clearInterval(timer);
  }, [trial?.id]);

  if (!trial || !trial.params)
    return (
      <div
        className="min-h-[250px] flex items-center justify-center italic border rounded-xl px-8 text-center text-xs"
        style={{
          color: "var(--viz-text-muted)",
          borderColor: "var(--viz-border)",
          backgroundColor: "var(--viz-bg-surface)",
        }}
      >
        Nenhum melhor trial ainda
      </div>
    );

  return (
    <Card
      title="Melhor Trial"
      icon={Icon}
      className="h-full"
      helpText={ChartRegistry.get("params")}
    >
      <div className="flex flex-col h-full">
        <div className="flex justify-center mb-6 mt-1 relative">
          <div className="w-20 h-20 relative flex items-center justify-center">
            {/* Orbits - Outside the glow container to avoid clipping */}
            <div className="pff-orbit pff-orbit-cw" aria-hidden="true">
              <div className="pff-orbit-dot"></div>
            </div>
            <div className="pff-orbit pff-orbit-ccw" aria-hidden="true" style={{ inset: "-12px" }}>
              <div className="pff-orbit-dot" style={{ width: "5px", height: "5px" }}></div>
            </div>
            <div className="pff-micro-orbit" aria-hidden="true"></div>

            {/* Chip Core - Contains the glow and content */}
            <div className="absolute inset-0 rounded-full border-2 border-lime-500/40 bg-zinc-900/80 pff-glow-sweep flex items-center justify-center z-10">
              <span className="text-4xl font-bold text-lime-400 font-mono relative z-10 pff-flip">
                #{displayId}
              </span>
            </div>

            <div className="absolute -bottom-3 left-1/2 -translate-x-1/2 z-20">
              <div className="px-2 py-0.5 rounded-full bg-lime-500/10 border border-lime-500/40 text-[10px] font-bold tracking-widest text-lime-300 pff-breath text-center whitespace-nowrap">
                MELHOR
              </div>
            </div>
          </div>
        </div>
        <div className="space-y-4 flex-1 overflow-y-auto max-h-[400px] custom-scrollbar pff-stagger">
          {Object.entries(trial.params).map(([key, val], idx) => (
            <div
              key={key}
              className="flex justify-between items-center border-b border-zinc-800/60 pb-2"
              style={{ "--stagger": idx * 35 }}
            >
              {renderParamWithHints(
                key.replace(/_/g, " "),
                typeof val === "number" ? val.toFixed(4) : String(val)
              )}
              <span
                className="font-mono tabular-nums font-bold text-sm"
                style={{ color: "var(--viz-text-primary)" }}
              >
                {typeof val === "number" ? val.toFixed(4) : String(val)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
};
