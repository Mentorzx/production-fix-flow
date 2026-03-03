/**
 * Provide RawConfigCard module functionality for the HPO dashboard.
 */

import { Sliders } from "../../../ui/icons.jsx";
import { Card } from "../../../ui/Card.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

/**
 * Expose raw config card for dashboard usage.
 */
export const RawConfigCard = ({ config }) => {
  if (!config || Object.keys(config).length === 0) {
    return (
      <Card
        title="Configuração Bruta"
        icon={Sliders}
        className="h-full"
        helpText={ChartRegistry.get("raw_config")}
      >
        <div className="flex h-full items-center justify-center text-zinc-600 italic text-xs">
          Sem configuração
        </div>
      </Card>
    );
  }

  // Filter relevant keys if needed, or sort
  const entries = Object.entries(config).sort(([a], [b]) => a.localeCompare(b));

  return (
    <Card
      title="Configuração Bruta"
      icon={Sliders}
      className="h-full"
      helpText={ChartRegistry.get("raw_config")}
    >
      <div className="absolute inset-0 p-4 overflow-auto custom-scrollbar">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-3">
          {entries.map(([key, value]) => (
            <div key={key} className="flex flex-col border-b border-zinc-800/50 pb-1">
              <span className="text-[9px] uppercase tracking-wider text-zinc-500 font-bold mb-0.5 break-all">
                {key}
              </span>
              <span className="text-xs font-mono text-zinc-300 break-all">
                {typeof value === "object" ? JSON.stringify(value) : String(value)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
};
