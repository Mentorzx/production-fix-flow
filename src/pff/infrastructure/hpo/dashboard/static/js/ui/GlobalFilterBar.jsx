import { useStore } from "../store/store.jsx";

export const GlobalFilterBar = () => {
  const { filters, setFilters, viewMode } = useStore();

  if (viewMode !== "study") return null;

  return (
    <div
      className="flex-none px-6 py-2 border-b flex items-center justify-between z-101 relative backdrop-blur-sm"
      style={{ backgroundColor: "var(--viz-bg-surface)", borderColor: "var(--viz-border)" }}
    >
      <div className="flex items-center gap-4">
        <span
          className="text-[10px] font-bold uppercase tracking-wider flex items-center gap-2"
          style={{ color: "var(--viz-text-muted)" }}
        >
          Filtros Globais:
        </span>

        <div
          className="flex items-center gap-4 px-3 py-1 rounded-md border"
          style={{ backgroundColor: "var(--viz-bg-elevated)", borderColor: "var(--viz-border)" }}
        >
          <label className="flex items-center gap-2 cursor-pointer group select-none">
            <input
              type="checkbox"
              checked={filters.includeWarmup}
              onChange={(e) => setFilters({ ...filters, includeWarmup: e.target.checked })}
              className="w-3.5 h-3.5 rounded-sm border-zinc-600 bg-zinc-800 text-orange-500 focus:ring-1 focus:ring-orange-500/50"
            />
            <span
              className="text-[11px] group-hover:text-zinc-200 transition-colors"
              style={{ color: "var(--viz-text-secondary)" }}
            >
              Warmup
            </span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer group select-none">
            <input
              type="checkbox"
              checked={filters.includePruned}
              onChange={(e) => setFilters({ ...filters, includePruned: e.target.checked })}
              className="w-3.5 h-3.5 rounded-sm border-zinc-600 bg-zinc-800 text-orange-500 focus:ring-1 focus:ring-orange-500/50"
            />
            <span
              className="text-[11px] group-hover:text-zinc-200 transition-colors"
              style={{ color: "var(--viz-text-secondary)" }}
            >
              Pruned
            </span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer group select-none">
            <input
              type="checkbox"
              checked={filters.onlyComplete}
              onChange={(e) => setFilters({ ...filters, onlyComplete: e.target.checked })}
              className="w-3.5 h-3.5 rounded-sm border-zinc-600 bg-zinc-800 text-lime-500 focus:ring-1 focus:ring-lime-500/50"
            />
            <span
              className="text-[11px] group-hover:text-zinc-200 transition-colors"
              style={{ color: "var(--viz-text-secondary)" }}
            >
              Completos
            </span>
          </label>
        </div>
      </div>

      <div className="text-[10px] italic opacity-50" style={{ color: "var(--viz-text-muted)" }}>
        Aplicado a todas as abas
      </div>
    </div>
  );
};
