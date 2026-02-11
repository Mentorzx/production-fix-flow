import React from "react";
import { Theme } from "./Theme.js";
import { PortalTooltip } from "./PortalTooltip.jsx";
import { Info } from "./icons.jsx";

/**
 * Card shell with header, help button, and glow effect.
 */

const HelpButton = React.memo(({ text }) => {
  const isHelpTextObject = (value) => {
    return typeof value === "object" && value !== null && "tech" in value;
  };

  const tooltipContent = (
    <div
      className="w-64 border p-3 rounded-xl shadow-2xl text-[10px]"
      style={{
        backgroundColor: Theme.ui.background,
        borderColor: Theme.ui.border,
        color: Theme.ui.text.secondary,
      }}
    >
      {isHelpTextObject(text) ? (
        <div className="space-y-2">
          <div>
            <span
              className="text-[8px] font-black uppercase block mb-1"
              style={{ color: Theme.semantic.warning }}
            >
              Explicação Técnica
            </span>
            <div className="leading-tight" style={{ color: Theme.ui.text.primary }}>
              {text.tech}
            </div>
          </div>
          {text.simple && (
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
                {text.simple}
              </div>
            </div>
          )}
          {Array.isArray(text.extra) && text.extra.length > 0 && (
            <div className="pt-2 border-t" style={{ borderColor: Theme.ui.border }}>
              <span
                className="text-[8px] font-black uppercase block mb-1"
                style={{ color: Theme.palette.cyberYellow }}
              >
                Valores
              </span>
              <div className="space-y-1">
                {text.extra.map((item, index) => (
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
        </div>
      ) : typeof text === "string" ? (
        text
      ) : (
        JSON.stringify(text)
      )}
    </div>
  );
  return (
    <div className="inline-block ml-2">
      <PortalTooltip content={tooltipContent}>
        <div
          className="cursor-help w-6 h-6 flex items-center justify-center border-2 hover:scale-110 hover:brightness-125 transition-all duration-200"
          style={{
            borderColor: Theme.palette.cyberYellow,
            backgroundColor: "rgba(229, 197, 88, 0.1)",
            color: Theme.palette.cyberYellow,
            borderRadius: "6px",
            boxShadow: "0 0 8px rgba(229, 197, 88, 0.3)",
          }}
        >
          <Info size={14} />
        </div>
      </PortalTooltip>
    </div>
  );
});

export { HelpButton };

export const Card = React.memo(
  ({ children, className = "", title, icon: Icon, action, helpText, glow, headerRight }) => (
    <article
      className={`rounded-xl flex flex-col relative overflow-hidden card-edge ${className} ${glow ? "card-edge-active" : ""}`}
      style={{ backgroundColor: Theme.ui.surface }}
    >
      <div className="pff-micro-orbit" aria-hidden="true" style={{ opacity: 0.05 }}></div>
      {(title || Icon) && (
        <header
          className="flex items-center justify-between px-5 py-5 border-b"
          style={{
            borderColor: Theme.ui.border,
            backgroundColor: Theme.ui.surfaceHighlight + "40",
          }}
        >
          <div className="flex items-center gap-2.5">
            {Icon && (
              <div
                className="p-1 rounded-sm"
                style={{ backgroundColor: Theme.ui.background, color: Theme.semantic.warning }}
              >
                <Icon size={14} />
              </div>
            )}
            <h3
              className="font-black text-[10px] uppercase tracking-widest"
              style={{ color: Theme.ui.text.primary }}
            >
              {title}
            </h3>
          </div>
          <div className="flex items-center gap-2">
            {headerRight}
            {helpText && <HelpButton text={helpText} />}
            {action}
          </div>
        </header>
      )}
      <div className="p-5 flex-1 relative flex flex-col min-h-0">{children}</div>
    </article>
  )
);
