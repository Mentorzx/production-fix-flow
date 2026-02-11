import { useMemo } from "react";
import { AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import {
  Card,
  Cpu,
  EmptyState,
  ChartContainer,
  PortalTooltip,
} from "../../../ui/BaseComponents.jsx";
import { Theme } from "../../../ui/Theme.js";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { MetricRegistry } from "../../../domain/metrics/MetricRegistry.js";

const normalizeHardware = (hardware) => {
  if (!hardware || typeof hardware !== "object") return null;

  const gpu0 = Array.isArray(hardware.gpus) && hardware.gpus.length > 0 ? hardware.gpus[0] : null;

  const cpu =
    typeof hardware.cpu_usage === "number" ? hardware.cpu_usage : hardware.cpu_utilization;
  const gpu =
    gpu0 && typeof gpu0.utilization === "number" ? gpu0.utilization : hardware.gpu_utilization;
  const vramUsagePct =
    gpu0 && typeof gpu0.vram_usage_pct === "number"
      ? gpu0.vram_usage_pct
      : hardware.vram_utilization;
  const ramUsagePct =
    typeof hardware.ram_usage_pct === "number" ? hardware.ram_usage_pct : hardware.ram_utilization;

  // Attempt to find totals for raw value calculation
  const ramTotalGb =
    typeof hardware.ram_total_gb === "number"
      ? hardware.ram_total_gb
      : hardware.ram_total
        ? hardware.ram_total / 1024 ** 3
        : 0;
  const ramUsedGb =
    typeof hardware.ram_used_gb === "number"
      ? hardware.ram_used_gb
      : hardware.ram_used
        ? hardware.ram_used / 1024 ** 3
        : null;

  const vramTotalGb = gpu0?.vram_total
    ? gpu0.vram_total / 1024 ** 3
    : hardware.vram_total
      ? hardware.vram_total / 1024 ** 3
      : 0;
  const vramUsedGb = gpu0?.vram_used
    ? gpu0.vram_used / 1024 ** 3
    : hardware.vram_used
      ? hardware.vram_used / 1024 ** 3
      : null;

  const ram =
    typeof ramUsagePct === "number"
      ? ramUsagePct
      : ramUsedGb != null && ramTotalGb > 0
        ? (ramUsedGb / ramTotalGb) * 100
        : null;
  const vram =
    typeof vramUsagePct === "number"
      ? vramUsagePct
      : vramUsedGb != null && vramTotalGb > 0
        ? (vramUsedGb / vramTotalGb) * 100
        : null;

  return {
    cpu,
    gpu,
    vram,
    ram,
    ramTotalGb,
    ramUsedGb,
    vramTotalGb,
    vramUsedGb,
    gpuName: gpu0 && typeof gpu0.name === "string" ? gpu0.name : null,
  };
};

export const HardwareMonitorCard = ({ hardware, history }) => {
  const hw = normalizeHardware(hardware);

  const labelWithHint = (metricKey, label, extraValue = null) => {
    const hint = MetricRegistry.get(metricKey);
    if (!hint) return <span>{label}</span>;

    const tooltipContent = (
      <div
        className="w-72 border p-3 rounded-xl shadow-2xl text-[10px]"
        style={{
          backgroundColor: Theme.ui.background,
          borderColor: Theme.ui.border,
          color: Theme.ui.text.secondary,
        }}
      >
        <div className="space-y-2">
          <div>
            <span
              className="text-[8px] font-black uppercase block mb-1"
              style={{ color: Theme.semantic.warning }}
            >
              Explicação Técnica
            </span>
            <div className="leading-tight" style={{ color: Theme.ui.text.primary }}>
              {hint.tech}
            </div>
          </div>
          {hint.simple && (
            <div className="pt-2 border-t" style={{ borderColor: Theme.ui.border }}>
              <span
                className="text-[8px] font-black uppercase block mb-1"
                style={{ color: Theme.semantic.success }}
              >
                Para Leigos
              </span>
              <div
                className="italic leading-tight border-l-2 pl-2"
                style={{ color: Theme.palette.mint, borderColor: Theme.palette.vividGreen + "33" }}
              >
                {hint.simple}
              </div>
            </div>
          )}
          {Array.isArray(hint.extra) && hint.extra.length > 0 && (
            <div className="pt-2 border-t" style={{ borderColor: Theme.ui.border }}>
              <span
                className="text-[8px] font-black uppercase block mb-1"
                style={{ color: Theme.palette.cyberYellow }}
              >
                Valores
              </span>
              <div className="space-y-1">
                {hint.extra.map((item, index) => (
                  <div
                    key={`${item.label}-${index}`}
                    className="text-[10px] leading-tight flex gap-2"
                    style={{ color: Theme.ui.text.secondary }}
                  >
                    <span
                      className="font-semibold min-w-[72px]"
                      style={{ color: Theme.palette.apricot }}
                    >
                      {item.label}:
                    </span>
                    <span>{item.value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {extraValue && (
            <div className="pt-2 border-t" style={{ borderColor: Theme.ui.border }}>
              <span
                className="text-[8px] font-black uppercase block mb-1"
                style={{ color: Theme.palette.cyberYellow }}
              >
                Valor Atual
              </span>
              <div
                className="text-[11px] font-mono font-black tabular-nums"
                style={{ color: Theme.ui.text.primary }}
              >
                {extraValue}
              </div>
            </div>
          )}
        </div>
      </div>
    );

    return (
      <PortalTooltip content={tooltipContent}>
        <span className="cursor-help border-b border-dotted border-zinc-600 hover:text-orange-400 transition-colors">
          {label}
        </span>
      </PortalTooltip>
    );
  };

  const items = hw
    ? [
        { key: "cpu", l: "CPU", v: hw.cpu, color: "bg-orange-500", raw: null },
        { key: "gpu", l: "GPU", v: hw.gpu, color: "bg-rose-500", raw: null },
        {
          key: "vram",
          l: "VRAM",
          v: hw.vram,
          color: "bg-purple-500",
          raw:
            hw.vramUsedGb != null && hw.vramTotalGb
              ? `${hw.vramUsedGb.toFixed(1)}GB / ${hw.vramTotalGb.toFixed(1)}GB`
              : null,
        },
        {
          key: "ram",
          l: "RAM",
          v: hw.ram,
          color: "bg-cyan-500",
          raw:
            hw.ramUsedGb != null && hw.ramTotalGb
              ? `${hw.ramUsedGb.toFixed(1)}GB / ${hw.ramTotalGb.toFixed(1)}GB`
              : null,
        },
      ].filter((i) => typeof i.v === "number" && Number.isFinite(i.v))
    : [];

  const chartData = useMemo(() => {
    if (!history || history.length === 0) return [];
    return history.map((h) => ({
      id: h.id,
      cpu: h.cpu_usage || 0,
      gpu: h.gpu_utilization || 0,
      vram: h.vram_usage_pct || 0,
      ram: h.ram_usage_pct || 0,
    }));
  }, [history]);

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-zinc-900 border border-zinc-700 p-2 rounded shadow-xl text-xs font-mono z-50">
          <p className="text-zinc-400 mb-2 border-b border-zinc-700 pb-1">Epoch {label}</p>
          {payload.map((p) => {
            let raw = "";
            if (p.name === "RAM" && hw?.ramTotalGb) {
              const gb = (p.value / 100) * hw.ramTotalGb;
              raw = `(${gb.toFixed(1)} GB)`;
            }
            // Simplified VRAM check if history had vram_usage_pct

            return (
              <div key={p.name} className="flex justify-between gap-4 mb-1">
                <span style={{ color: p.color }}>{p.name}:</span>
                <span className="text-zinc-200">
                  {p.value.toFixed(1)}% <span className="text-zinc-500">{raw}</span>
                </span>
              </div>
            );
          })}
        </div>
      );
    }
    return null;
  };

  return (
    <Card
      title="Monitor de Hardware"
      icon={Cpu}
      className="h-full"
      glow
      helpText={ChartRegistry.get("hardware_monitor")}
    >
      <div className="h-full flex flex-col p-4 gap-4">
        {/* Progress Bars */}
        <div className="flex-none space-y-3">
          {items.length > 0 ? (
            items.map((i) => (
              <div key={i.l}>
                <div className="flex justify-between text-xs mb-1 text-zinc-400 font-mono">
                  <span>
                    {labelWithHint(i.key, i.l, `${i.v.toFixed(1)}%${i.raw ? ` (${i.raw})` : ""}`)}
                  </span>
                  <span>
                    {i.v.toFixed(1)}%{" "}
                    {i.raw && <span className="ml-2 text-zinc-600">({i.raw})</span>}
                  </span>
                </div>
                <div className="h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${i.color}`}
                    style={{ width: `${Math.max(0, Math.min(100, i.v))}%` }}
                  ></div>
                </div>
              </div>
            ))
          ) : (
            <EmptyState className="text-sm">Sem telemetria</EmptyState>
          )}
        </div>

        {/* History Chart */}
        {chartData.length > 0 && (
          <div
            className="flex-1 min-h-[120px] rounded border overflow-hidden relative"
            style={{
              backgroundColor: "var(--viz-bg-elevated)",
              borderColor: "var(--viz-border)",
            }}
          >
            <div className="absolute top-1 right-2 text-[9px] text-zinc-600 font-mono z-10">
              HISTORY
            </div>
            <ChartContainer minHeight={120} className="h-full">
              <AreaChart data={chartData} margin={{ top: 20, right: 0, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={Theme.ui.grid} vertical={false} />
                <XAxis dataKey="id" hide />
                <YAxis domain={[0, 100]} hide />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="monotone"
                  dataKey="cpu"
                  stroke={Theme.semantic.hardware.cpu}
                  fill={Theme.semantic.hardware.cpu}
                  fillOpacity={0.08}
                  strokeWidth={1.5}
                  name="CPU"
                  dot={false}
                  activeDot={{ r: 3 }}
                />
                <Area
                  type="monotone"
                  dataKey="gpu"
                  stroke={Theme.semantic.hardware.gpu}
                  fill={Theme.semantic.hardware.gpu}
                  fillOpacity={0.08}
                  strokeWidth={1.5}
                  name="GPU"
                  dot={false}
                  activeDot={{ r: 3 }}
                />
                <Area
                  type="monotone"
                  dataKey="vram"
                  stroke={Theme.semantic.hardware.vram}
                  fill={Theme.semantic.hardware.vram}
                  fillOpacity={0.08}
                  strokeWidth={1.5}
                  name="VRAM"
                  dot={false}
                  activeDot={{ r: 3 }}
                />
                <Area
                  type="monotone"
                  dataKey="ram"
                  stroke={Theme.semantic.hardware.ram}
                  fill={Theme.semantic.hardware.ram}
                  fillOpacity={0.08}
                  strokeWidth={1.5}
                  name="RAM"
                  dot={false}
                  activeDot={{ r: 3 }}
                />
              </AreaChart>
            </ChartContainer>
          </div>
        )}
      </div>
    </Card>
  );
};
