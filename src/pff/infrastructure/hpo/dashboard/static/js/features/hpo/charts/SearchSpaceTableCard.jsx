/**
 * Provide SearchSpaceTableCard module functionality for the HPO dashboard.
 */

import { TableIcon } from "../../../ui/icons.jsx";
import { Card } from "../../../ui/Card.jsx";
import { renderParamWithHints } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

/**
 * Expose search space table card for dashboard usage.
 */
export const SearchSpaceTableCard = ({ searchSpace }) => {
  const formatRange = (attrs) => {
    const low = attrs?.low ?? attrs?.min;
    const high = attrs?.high ?? attrs?.max;
    if (low == null && high == null) return "faixa não especificada";
    if (low != null && high != null) return `${low} → ${high}`;
    return low != null ? `>= ${low}` : `<= ${high}`;
  };

  const formatAttrs = (attrs) => {
    if (!attrs || typeof attrs !== "object") return null;
    const parts = [];
    if (Array.isArray(attrs.choices)) parts.push(`opções: ${attrs.choices.join(", ")}`);
    if (attrs.step != null) parts.push(`passo: ${attrs.step}`);
    if (attrs.log != null) parts.push(`escala: ${attrs.log ? "log" : "linear"}`);
    if (attrs.q != null) parts.push(`quantização: ${attrs.q}`);
    const range = formatRange(attrs);
    if (range) parts.unshift(`faixa: ${range}`);
    return parts.length > 0 ? parts.join(" • ") : null;
  };

  const formatDist = (dist) => {
    if (dist == null) return "—";
    if (typeof dist !== "object") return String(dist);
    const name = dist?.name || dist?.type || dist?.kind;
    const attrs = dist?.attributes || dist?.params || dist;
    const details = formatAttrs(attrs);
    if (name && details) return `${name}: ${details}`;
    if (name) return String(name);
    if (details) return details;
    return Object.entries(dist)
      .map(([key, value]) => `${key}: ${Array.isArray(value) ? value.join(", ") : value}`)
      .join(" • ");
  };

  const helpText = ChartRegistry.get("search_space_table", {
    title: "Espaço de Busca",
    tech: "Distribuições/priors por hiperparâmetro usados na otimização.",
    simple: "As opções que o robô pode tentar.",
  });

  return (
    <Card title="Espaço de Busca" icon={TableIcon} className="h-full" helpText={helpText}>
      <div className="absolute inset-0 p-0 overflow-auto custom-scrollbar">
        <table className="w-full text-left text-[10px]">
          <thead className="bg-zinc-900 sticky top-0">
            <tr>
              <th className="p-2 border-b border-zinc-800">Parâmetro</th>
              <th className="p-2 border-b border-zinc-800">Distribuição</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {Object.entries(searchSpace || {}).map(([name, dist]) => (
              <tr key={name} className="dashboard-table-row border-b border-zinc-800/50">
                <td className="p-2 text-orange-400">{renderParamWithHints(name)}</td>
                <td className="p-2 text-zinc-400">{formatDist(dist)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
};
