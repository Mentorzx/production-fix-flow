/**
 * Global command palette launcher button.
 */

import { Search } from "./icons.jsx";

const FLOATING_PILL_WIDTH = "min(420px, calc(100vw - 2rem))";
const FLOATING_PILL_HEIGHT = "56px";

/**
 * Expose search launcher button for dashboard usage.
 */
export const SearchLauncherButton = ({ onClick, buttonRef }) => {
  return (
    <div className="relative">
      <div
        className="absolute -inset-1 rounded-full blur-xl opacity-75 pointer-events-none"
        style={{
          background:
            "radial-gradient(circle, color-mix(in srgb, var(--viz-palette-4-yellow), transparent 75%) 0%, transparent 72%)",
        }}
      />
      <div
        className="relative p-1 rounded-full backdrop-blur-xl floating-pill-cinematic"
        style={{ width: FLOATING_PILL_WIDTH, height: FLOATING_PILL_HEIGHT }}
      >
        <button
          ref={buttonRef}
          type="button"
          onClick={onClick}
          className="btn-toggle group h-full w-full rounded-full px-4 py-1.5 inline-flex items-center justify-between gap-2.5 transition-all duration-200"
          title="Buscar cards (Ctrl+K)"
          aria-label="Abrir busca global do dashboard"
          style={{
            "--viz-icon-active": "var(--viz-palette-4-yellow)",
            color: "var(--viz-text-primary)",
            borderColor: "color-mix(in srgb, var(--viz-palette-4-yellow), transparent 42%)",
            backgroundColor: "color-mix(in srgb, var(--viz-palette-4-yellow), transparent 89%)",
            boxShadow: "0 0 18px color-mix(in srgb, var(--viz-palette-4-yellow), transparent 82%)",
          }}
        >
          <span className="inline-flex items-center gap-2.5">
            <Search size={14} />
            <span className="text-[10px] leading-none font-black uppercase tracking-[0.12em]">
              Busca
            </span>
          </span>
          <kbd
            className="font-mono text-[10px] px-1.5 py-0.5 rounded border"
            style={{
              borderColor: "color-mix(in srgb, var(--viz-border), transparent 8%)",
              color: "var(--viz-text-muted)",
            }}
          >
            Ctrl+K
          </kbd>
        </button>
      </div>
    </div>
  );
};
