/**
 * Provide TableComponents module functionality for the HPO dashboard.
 * SOTA Design: Heatmap pills + IDE-style control headers
 */

import React, { useState } from "react";
import {
  ArrowUp,
  ArrowDown,
  Download,
  ChevronRight,
  Search,
  Sliders,
  X,
} from "./BaseComponents.jsx";
import { PortalTooltip } from "./UIComponents.jsx";
import { ExportService } from "../utils/ExportService.js";
import { Theme } from "./Theme.js";

/**
 * Expose sorted table header for dashboard usage.
 */
export const SortedTableHeader = React.memo(
  ({ label, sortKey, currentSort, onSort, helpText, directionHint, align = "left" }) => {
    const active = currentSort?.key === sortKey;
    const direction = active ? currentSort.direction : "none";

    const hintUp = directionHint === "up";
    const hintDown = directionHint === "down";

    // SOTA Theme Colors for Sort Indicators
    const upClass =
      active && direction === "asc"
        ? "text-orange-400"
        : hintUp
          ? "text-emerald-400/90 group-hover:text-emerald-300"
          : "text-zinc-800 opacity-20";

    const downClass =
      active && direction === "desc"
        ? "text-orange-400"
        : hintDown
          ? "text-rose-400/90 group-hover:text-rose-300"
          : "text-zinc-800 opacity-20";

    // Tooltip Content Component
    const TooltipContent = (
      <div
        className="w-64 border p-3 rounded-xl shadow-2xl text-left normal-case font-sans z-50"
        style={{ backgroundColor: "var(--viz-bg-surface)", borderColor: "var(--viz-border)" }}
      >
        <div className="space-y-2">
          <div>
            <span
              className="text-[8px] font-black uppercase block mb-0.5"
              style={{ color: "var(--viz-palette-3-orange)" }}
            >
              Explicação Técnica
            </span>
            <p className="text-[10px] leading-tight" style={{ color: "var(--viz-text-primary)" }}>
              {helpText?.tech || helpText}
            </p>
          </div>
          {helpText?.simple && (
            <div className="pt-2 border-t" style={{ borderColor: "var(--viz-border)" }}>
              <span
                className="text-[8px] font-black uppercase block mb-0.5"
                style={{ color: "var(--viz-palette-2-green)" }}
              >
                Para Leigos
              </span>
              <p
                className="text-[10px] italic leading-tight border-l-2 pl-2"
                style={{
                  color: "var(--viz-text-secondary)",
                  borderColor: "rgba(60, 180, 75, 0.2)",
                }}
              >
                {helpText.simple}
              </p>
            </div>
          )}
          {Array.isArray(helpText?.extra) && helpText.extra.length > 0 && (
            <div className="pt-2 border-t" style={{ borderColor: "var(--viz-border)" }}>
              <span
                className="text-[8px] font-black uppercase block mb-0.5"
                style={{ color: "var(--viz-palette-4-yellow)" }}
              >
                Valores
              </span>
              <div className="space-y-1">
                {helpText.extra.map((item, index) => (
                  <div
                    key={`${item.label}-${index}`}
                    className="text-[10px] leading-tight flex gap-2"
                  >
                    <span
                      className="font-semibold min-w-[72px]"
                      style={{ color: "var(--viz-palette-4-yellow)" }}
                    >
                      {item.label}:
                    </span>
                    <span style={{ color: "var(--viz-text-secondary)" }}>{item.value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          <div
            className="pt-2 border-t flex items-center justify-between"
            style={{ borderColor: "var(--viz-border)" }}
          >
            <span
              className="text-[8px] font-black uppercase"
              style={{ color: "var(--viz-text-muted)" }}
            >
              Melhor direção
            </span>
            <span className="text-[8px] font-mono" style={{ color: "var(--viz-text-secondary)" }}>
              {directionHint === "up"
                ? "MAIOR (UP)"
                : directionHint === "down"
                  ? "MENOR (DOWN)"
                  : "N/A"}
            </span>
          </div>
        </div>
        <div
          className="absolute top-full left-1/2 -translate-x-1/2 -mt-1 w-2 h-2 border-r border-b rotate-45"
          style={{ backgroundColor: "var(--viz-bg-surface)", borderColor: "var(--viz-border)" }}
        ></div>
      </div>
    );

    // Flex layout determination based on alignment
    // Left Align: Icons on Right
    // Right Align: Icons on Left
    // Center Align: Icons on Left (default)
    const justifyClass =
      align === "right" ? "justify-end" : align === "center" ? "justify-center" : "justify-start";
    const flexDirection = align === "right" ? "flex-row" : "flex-row-reverse";

    return (
      <div
        className={`flex items-center gap-2 select-none relative group h-full ${justifyClass} ${flexDirection} ${onSort ? "cursor-pointer" : "cursor-default"}`}
        onClick={() => onSort?.(sortKey)}
      >
        {/* Sort Indicators */}
        <div className="flex flex-col -space-y-1 flex-none">
          <ArrowUp size={14} className={upClass} />
          <ArrowDown size={14} className={downClass} />
        </div>

        <div className="flex items-center gap-1.5 truncate">
          {helpText ? (
            <PortalTooltip content={TooltipContent}>
              <span
                className={`text-[10px] font-black uppercase tracking-widest transition-colors border-b border-dotted hover:border-zinc-500 ${active ? "text-orange-400" : "text-zinc-500 group-hover:text-zinc-300"}`}
                style={{ borderColor: "var(--viz-border)" }}
              >
                {label}
              </span>
            </PortalTooltip>
          ) : (
            <span
              className={`text-[10px] font-black uppercase tracking-widest transition-colors ${active ? "text-orange-400" : "text-zinc-500 group-hover:text-zinc-300"}`}
            >
              {label}
            </span>
          )}
        </div>
      </div>
    );
  }
);

/**
 * Expose export menu for dashboard usage.
 */
export const ExportMenu = React.memo(({ data, filename = "hpo_export" }) => {
  const [isOpen, setIsOpen] = React.useState(false);

  const handleExport = async (format) => {
    setIsOpen(false);
    await ExportService.export(format, data, filename);
  };

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        aria-pressed={isOpen}
        aria-expanded={isOpen}
        aria-label="Exportar dados"
        className="btn-toggle border p-1.5 rounded-lg"
        style={{
          "--viz-icon-active": "var(--viz-palette-4-yellow)",
          borderColor: isOpen ? "var(--viz-palette-4-yellow)" : "var(--viz-border)",
          backgroundColor: isOpen
            ? "color-mix(in srgb, var(--viz-palette-4-yellow), transparent 86%)"
            : "color-mix(in srgb, var(--viz-bg-surface), var(--viz-bg-canvas) 14%)",
          color: isOpen ? "var(--viz-text-primary)" : "var(--viz-text-muted)",
        }}
      >
        <Download size={14} />
      </button>
      {isOpen && (
        <div
          className="absolute right-0 top-full mt-2 w-32 border rounded-xl shadow-2xl z-50 overflow-hidden"
          style={{
            backgroundColor: "var(--viz-bg-surface)",
            borderColor: "var(--viz-border)",
          }}
        >
          {["json", "csv", "parquet", "toon"].map((fmt) => (
            <button
              key={fmt}
              type="button"
              onClick={() => handleExport(fmt)}
              className="w-full text-left px-4 py-2 text-xs uppercase font-bold transition-colors"
              style={{
                color: "var(--viz-text-secondary)",
              }}
              onMouseEnter={(event) => {
                event.currentTarget.style.backgroundColor =
                  "color-mix(in srgb, var(--viz-palette-3-orange), transparent 88%)";
                event.currentTarget.style.color = "var(--viz-palette-3-orange)";
              }}
              onMouseLeave={(event) => {
                event.currentTarget.style.backgroundColor = "transparent";
                event.currentTarget.style.color = "var(--viz-text-secondary)";
              }}
            >
              {fmt}
            </button>
          ))}
        </div>
      )}
    </div>
  );
});

/**
 * TableControlHeader - IDE-style dense control bar
 * SOTA Design: Search + Column Toggles + Action Buttons
 */
export const TableControlHeader = React.memo(
  ({
    searchQuery,
    onSearchChange,
    columnGroups = [],
    activeGroup = "all",
    onGroupChange,
    onExport,
    onColumnsClick,
    title = "Tabela",
    totalItems = 0,
  }) => {
    const [isSearchFocused, setIsSearchFocused] = useState(false);

    return (
      <div className="flex items-center justify-between px-4 py-3 border-b bg-zinc-900/50">
        {/* Left: Title + Search */}
        <div className="flex items-center gap-4">
          <h3 className="text-xs font-black uppercase tracking-wider text-zinc-400">{title}</h3>

          {/* Search Input */}
          <div
            className={`
            flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-all
            ${isSearchFocused ? "border-zinc-600 bg-zinc-800" : "border-zinc-800 bg-zinc-950"}
          `}
          >
            <Search size={14} className="text-zinc-500" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => onSearchChange?.(e.target.value)}
              onFocus={() => setIsSearchFocused(true)}
              onBlur={() => setIsSearchFocused(false)}
              placeholder="Filtrar trials..."
              className="bg-transparent text-xs text-zinc-300 placeholder-zinc-600 outline-none w-32"
            />
            {searchQuery && (
              <button
                onClick={() => onSearchChange?.("")}
                className="text-zinc-500 hover:text-zinc-300"
              >
                <X size={12} />
              </button>
            )}
          </div>
        </div>

        {/* Center: Column Group Toggles (Segmented Control) */}
        {columnGroups.length > 0 && (
          <div className="flex items-center gap-1 p-1 rounded-lg bg-zinc-950 border border-zinc-800">
            {columnGroups.map((group) => (
              <button
                key={group.key}
                onClick={() => onGroupChange?.(group.key)}
                className={`
                px-3 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all
                ${
                  activeGroup === group.key
                    ? "bg-zinc-800 text-zinc-200"
                    : "text-zinc-500 hover:text-zinc-400"
                }
              `}
              >
                {group.label}
              </button>
            ))}
          </div>
        )}

        {/* Right: Action Buttons */}
        <div className="flex items-center gap-2">
          {/* Columns Button */}
          <button
            onClick={onColumnsClick}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-800 bg-zinc-950 text-zinc-400 hover:text-zinc-300 hover:border-zinc-700 transition-all"
          >
            <Sliders size={14} />
            <span className="text-[10px] font-bold uppercase">Colunas</span>
          </button>

          {/* Export Button */}
          <button
            onClick={onExport}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-700 bg-zinc-800/50 text-zinc-300 hover:bg-zinc-800 hover:text-white transition-all"
            style={{ borderColor: Theme.palette.indigo + "40" }}
          >
            <Download size={14} />
            <span className="text-[10px] font-bold uppercase">Export CSV</span>
          </button>

          {/* Item Count */}
          <span className="text-[10px] text-zinc-500 font-mono ml-2">{totalItems} itens</span>
        </div>
      </div>
    );
  }
);

/**
 * Expose pagination controls for dashboard usage.
 */
export const PaginationControls = React.memo(
  ({
    totalItems,
    currentPage,
    rowsPerPage,
    onPageChange,
    onRowsPerPageChange,
    footerStats = null,
  }) => {
    const totalPages = rowsPerPage === "All" ? 1 : Math.ceil(totalItems / rowsPerPage);
    const options = [10, 20, 50, 100, "All"];
    return (
      <div className="flex items-center justify-between px-4 py-3 border-t border-zinc-800 bg-zinc-900/30">
        <div className="flex items-center gap-4 text-[10px] text-zinc-500 font-mono">
          <span>Total: {totalItems}</span>
          {footerStats && <span className="border-l border-zinc-700 pl-4">{footerStats}</span>}
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-[10px] text-zinc-500 font-mono">
            <span>Exibir:</span>
            <select
              value={rowsPerPage}
              onChange={(e) =>
                onRowsPerPageChange?.(e.target.value === "All" ? "All" : Number(e.target.value))
              }
              className="bg-zinc-950 border border-zinc-800 rounded-md px-2 py-1 text-zinc-300"
            >
              {options.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          </div>
          <button
            onClick={() => onPageChange(currentPage - 1)}
            disabled={currentPage === 1}
            className="text-zinc-400 disabled:opacity-30"
          >
            <ChevronRight className="rotate-180" size={14} />
          </button>
          <span className="text-[10px] text-zinc-400 font-mono">
            Página {currentPage} de {totalPages}
          </span>
          <button
            onClick={() => onPageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
            className="text-zinc-400 disabled:opacity-30"
          >
            <ChevronRight size={14} />
          </button>
        </div>
      </div>
    );
  }
);
