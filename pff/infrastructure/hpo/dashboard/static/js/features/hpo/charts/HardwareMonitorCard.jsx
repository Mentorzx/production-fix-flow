import { useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';
import { Card, Cpu, EmptyState, ChartContainer } from "../../../ui/BaseComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

const normalizeHardware = (hardware) => {
    if (!hardware || typeof hardware !== "object") return null;

    const gpu0 = Array.isArray(hardware.gpus) && hardware.gpus.length > 0 ? hardware.gpus[0] : null;

    const cpu = typeof hardware.cpu_usage === "number" ? hardware.cpu_usage : hardware.cpu_utilization;
    const gpu =
        gpu0 && typeof gpu0.utilization === "number" ? gpu0.utilization : hardware.gpu_utilization;
    const vram =
        gpu0 && typeof gpu0.vram_usage_pct === "number" ? gpu0.vram_usage_pct : hardware.vram_utilization;
    const ram = typeof hardware.ram_usage_pct === "number" ? hardware.ram_usage_pct : hardware.ram_utilization;

    // Attempt to find totals for raw value calculation
    const ramTotal = hardware.ram_total || 0;
    const vramTotal = gpu0?.vram_total || hardware.vram_total || 0;

    return {
        cpu,
        gpu,
        vram,
        ram,
        ramTotal,
        vramTotal,
        gpuName: gpu0 && typeof gpu0.name === "string" ? gpu0.name : null,
    };
};

export const HardwareMonitorCard = ({ hardware, history }) => {
    const hw = normalizeHardware(hardware);

    const items = hw
        ? [
            { l: "CPU", v: hw.cpu, color: "bg-orange-500", raw: null },
            { l: "GPU", v: hw.gpu, color: "bg-rose-500", raw: null },
            { l: "VRAM", v: hw.vram, color: "bg-purple-500", raw: hw.vramTotal ? `${((hw.vram / 100) * hw.vramTotal / 1024).toFixed(1)}GB / ${(hw.vramTotal / 1024).toFixed(1)}GB` : null },
            { l: "RAM", v: hw.ram, color: "bg-cyan-500", raw: hw.ramTotal ? `${((hw.ram / 100) * hw.ramTotal / 1024 / 1024 / 1024).toFixed(1)}GB / ${(hw.ramTotal / 1024 / 1024 / 1024).toFixed(1)}GB` : null },
        ].filter((i) => typeof i.v === "number" && Number.isFinite(i.v))
        : [];

    const chartData = useMemo(() => {
        if (!history || history.length === 0) return [];
        return history.map(h => ({
            id: h.id,
            cpu: h.cpu_usage || 0,
            gpu: h.gpu_utilization || 0,
            ram: h.ram_usage_pct || 0
        }));
    }, [history]);

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div className="bg-zinc-900 border border-zinc-700 p-2 rounded shadow-xl text-xs font-mono z-50">
                    <p className="text-zinc-400 mb-2 border-b border-zinc-700 pb-1">Epoch {label}</p>
                    {payload.map((p) => {
                        let raw = "";
                        if (p.name === "RAM" && hw?.ramTotal) {
                            const gb = (p.value / 100) * hw.ramTotal / 1024 / 1024 / 1024; // Assuming bytes
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
                                    <span>{i.l}</span>
                                    <span>{i.v.toFixed(1)}% {i.raw && <span className="ml-2 text-zinc-600">({i.raw})</span>}</span>
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
                            backgroundColor: 'var(--viz-bg-elevated)',
                            borderColor: 'var(--viz-border)',
                        }}
                    >
                        <div className="absolute top-1 right-2 text-[9px] text-zinc-600 font-mono z-10">HISTORY</div>
                        <ChartContainer minHeight={120} className="h-full">
                            <AreaChart data={chartData} margin={{ top: 5, right: 0, left: 0, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                                <XAxis dataKey="id" hide />
                                <YAxis domain={[0, 100]} hide />
                                <Tooltip content={<CustomTooltip />} />
                                <Legend wrapperStyle={{ fontSize: '10px' }} />
                                <Area type="monotone" dataKey="cpu" stackId="1" stroke="#f97316" fill="#f97316" fillOpacity={0.1} strokeWidth={1} name="CPU" />
                                <Area type="monotone" dataKey="ram" stackId="1" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.1} strokeWidth={1} name="RAM" />
                                <Area type="monotone" dataKey="gpu" stackId="1" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.1} strokeWidth={1} name="GPU" />
                            </AreaChart>
                        </ChartContainer>
                    </div>
                )}
            </div>
        </Card>
    );
};
