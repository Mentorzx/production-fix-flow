/**
 * Provide ParamImportanceCard module functionality for the HPO dashboard.
 */

import { useMemo } from "react";
import { BarChart, Bar, XAxis, YAxis, Cell, Label, Tooltip } from "recharts";

import { Sliders } from "../../../ui/icons.jsx";
import { colors, DefaultCartesianGrid, ChartContainer } from "../../../ui/ChartPrimitives.jsx";
import { Card } from "../../../ui/Card.jsx";
import { WithData } from "../../../ui/EmptyStates.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { ParamRegistry } from "../../../domain/metrics/ParamRegistry.js";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";

/**
 * Expose param importance card for dashboard usage.
 */
export const ParamImportanceCard = ({ importances }) => {
  const data = useMemo(() => {
    if (!importances) return [];
    return Object.entries(importances)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value);
  }, [importances]);

  const helpText = ChartRegistry.get("fanova", {
    title: "Importância de Parâmetros",
    tech: "Importância estimada (fANOVA/heurística) dos hiperparâmetros no valor objetivo.",
    simple: "Quais botões mais mexem na nota.",
  });

  const paramHints = ParamRegistry.getAll();

  const CustomBarTooltip = ({ active, payload }) => {
    if (!active || !payload || payload.length === 0) return null;
    const entry = payload[0];
    const name = entry.payload?.name || "";
    const rawValue = entry.value;
    const pct = (Number(rawValue) * 100).toFixed(2);
    const normKey = String(name).toLowerCase().replace(/ /g, "_");
    const hint = paramHints[normKey] ?? paramHints[name] ?? null;

    return (
      <div className="w-72 p-4 bg-zinc-950 border border-zinc-800 rounded-xl shadow-2xl text-left font-sans z-50">
        <div className="space-y-3">
          <div className="pb-2 border-b border-zinc-900 flex justify-between items-center">
            <span className="text-[10px] font-black text-white uppercase tracking-wider">
              {name}
            </span>
            <span className="text-[10px] font-mono font-bold text-amber-400">{pct}%</span>
          </div>
          <div className="pt-1">
            <span className="text-[8px] font-black text-amber-500 uppercase block mb-1">
              Valor Bruto
            </span>
            <p className="text-[11px] text-zinc-100 font-mono bg-zinc-900/50 p-1 rounded-sm border border-zinc-800">
              {Number(rawValue).toFixed(6)}
            </p>
          </div>
          {hint && (
            <>
              <div>
                <span className="text-[8px] font-black text-orange-500 uppercase block mb-1">
                  Explicação Técnica
                </span>
                <p className="text-[10px] text-zinc-300 leading-tight normal-case">{hint.tech}</p>
              </div>
              {hint.simple && (
                <div className="pt-2 border-t border-zinc-900">
                  <span className="text-[8px] font-black text-lime-500 uppercase block mb-1">
                    Para Leigos
                  </span>
                  <p className="text-[10px] text-lime-400/80 italic leading-tight normal-case border-l-2 border-lime-500/20 pl-2">
                    {hint.simple}
                  </p>
                </div>
              )}
              {Array.isArray(hint.extra) && hint.extra.length > 0 && (
                <div className="pt-2 border-t border-zinc-900">
                  <span className="text-[8px] font-black text-amber-400 uppercase block mb-1">
                    Detalhes
                  </span>
                  <div className="space-y-1">
                    {hint.extra.map((item, index) => (
                      <div
                        key={`${item.label}-${index}`}
                        className="text-[10px] text-zinc-300 leading-tight flex gap-2"
                      >
                        <span className="text-amber-300/90 font-semibold min-w-[72px]">
                          {item.label}:
                        </span>
                        <span className="text-zinc-300">{item.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    );
  };

  return (
    <Card title="Importância de Parâmetros" icon={Sliders} className="h-full" helpText={helpText}>
      <div className="w-full h-full min-h-[300px]">
        <WithData when={data.length > 0} empty="Sem dados fANOVA" emptyClassName="text-zinc-500">
          <ChartContainer>
            <BarChart
              layout="vertical"
              data={data}
              margin={{ left: 5, right: 20, top: 10, bottom: 5 }}
            >
              <DefaultCartesianGrid horizontal={false} />
              <XAxis type="number" stroke={colors.text} tick={{ fontSize: 9 }} height={50}>
                <Label content={<ChartAxisLabel value="Importância" axis="x" />} />
              </XAxis>
              <YAxis
                dataKey="name"
                type="category"
                stroke={colors.text}
                width={100}
                tick={{ fontSize: 9 }}
                interval={0}
                tickFormatter={(v) => (String(v).length > 14 ? String(v).slice(0, 14) + "…" : v)}
              >
                <Label
                  content={<ChartAxisLabel value="Parâmetro" axis="y" />}
                  position="insideLeft"
                />
              </YAxis>
              <Tooltip
                content={<CustomBarTooltip />}
                cursor={{ fill: "rgba(255,255,255,0.03)" }}
                wrapperStyle={{ zIndex: 60 }}
              />
              <Bar
                isAnimationActive={false}
                dataKey="value"
                fill={colors.primary}
                radius={[0, 4, 4, 0]}
                barSize={12}
              >
                {data.map((e, i) => {
                  const fill = i === 0 ? colors.success : i <= 2 ? "#94a3b8" : colors.primary;
                  return <Cell key={e.name} fill={fill} />;
                })}
              </Bar>
            </BarChart>
          </ChartContainer>
        </WithData>
      </div>
    </Card>
  );
};
