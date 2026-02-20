/**
 * Provide FloatingScopeSwitch module functionality for the HPO dashboard.
 */

import { Activity, Layers } from "../ui/BaseComponents.jsx";

const FLOATING_PILL_WIDTH = "min(420px, calc(100vw - 2rem))";
const FLOATING_PILL_HEIGHT = "56px";

/** @typedef {{
 *  viewMode: string,
 *  setViewMode: (mode: string) => void
 * }} FloatingScopeSwitchProps */

/** @param {FloatingScopeSwitchProps} props */
export const FloatingScopeSwitch = ({ viewMode, setViewMode }) => {
  const options = [
    {
      id: "study",
      icon: Layers,
      label: "Macro",
      detail: "Estudo",
      color: "var(--viz-palette-1-blue)",
    },
    {
      id: "trial",
      icon: Activity,
      label: "Micro",
      detail: "Trial Atual",
      color: "var(--viz-palette-3-orange)",
    },
  ];

  return (
    <div className="absolute left-1/2 -translate-x-1/2 top-4 z-30">
      <div
        className="absolute -inset-1 rounded-full blur-xl opacity-80 pointer-events-none"
        style={{
          background:
            "radial-gradient(circle, color-mix(in srgb, var(--viz-palette-4-yellow), transparent 75%) 0%, transparent 70%)",
        }}
      />

      <div
        className="relative p-1 rounded-full backdrop-blur-xl floating-pill-cinematic"
        style={{
          width: FLOATING_PILL_WIDTH,
          height: FLOATING_PILL_HEIGHT,
          backgroundColor: "color-mix(in srgb, var(--viz-bg-surface), transparent 18%)",
        }}
        role="tablist"
        aria-label="Escopo macro e micro"
      >
        <div
          className="absolute inset-x-4 top-0 h-px pointer-events-none"
          style={{
            background:
              "linear-gradient(90deg, transparent 0%, color-mix(in srgb, var(--viz-palette-4-yellow), transparent 50%) 50%, transparent 100%)",
          }}
        />

        <div
          className="absolute top-1 bottom-1 left-1 rounded-full border transition-transform duration-300 ease-out"
          style={{
            width: "calc(50% - 4px)",
            transform: viewMode === "study" ? "translateX(0%)" : "translateX(100%)",
            borderColor:
              viewMode === "study" ? "var(--viz-palette-1-blue)" : "var(--viz-palette-3-orange)",
            backgroundColor:
              viewMode === "study"
                ? "color-mix(in srgb, var(--viz-palette-1-blue), transparent 86%)"
                : "color-mix(in srgb, var(--viz-palette-3-orange), transparent 86%)",
            boxShadow:
              viewMode === "study"
                ? "0 0 18px color-mix(in srgb, var(--viz-palette-1-blue), transparent 82%)"
                : "0 0 18px color-mix(in srgb, var(--viz-palette-3-orange), transparent 82%)",
          }}
        />

        <div className="relative z-10 grid h-full grid-cols-2 gap-1">
          {options.map((option) => {
            const isActive = viewMode === option.id;
            return (
              <button
                key={option.id}
                type="button"
                onClick={() => setViewMode(option.id)}
                role="tab"
                aria-selected={isActive}
                aria-pressed={isActive}
                className="btn-toggle h-full w-full rounded-full px-4 py-1.5 flex items-center justify-center gap-2.5 transition-all duration-200"
                style={{
                  "--viz-icon-active": option.color,
                  color: isActive ? "var(--viz-text-primary)" : "var(--viz-text-muted)",
                }}
              >
                <option.icon size={14} />
                <span className="text-[10px] leading-none font-black uppercase tracking-[0.12em]">
                  {option.label}
                </span>
                <span className="text-[10px] leading-none uppercase tracking-[0.12em] opacity-80">
                  {option.detail}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
