import { AlertTriangle, CheckCircle, Terminal, X } from "./icons.jsx";

const TYPE_META = {
  success: {
    icon: CheckCircle,
    accent: "var(--viz-palette-2-green)",
    label: "SUCESSO",
  },
  warning: {
    icon: AlertTriangle,
    accent: "var(--viz-palette-4-yellow)",
    label: "ALERTA",
  },
  danger: {
    icon: AlertTriangle,
    accent: "var(--viz-palette-6-red)",
    label: "CRÍTICO",
  },
};

const clamp01 = (value) => Math.max(0, Math.min(1, value));

const getToastMeta = (type) => TYPE_META[type] || TYPE_META.warning;
const isExecutionLogToast = (toast) => String(toast?.key || "").includes(":log:");

export const NotificationToasts = ({ toasts, nowMs, dismissToast }) => {
  if (!Array.isArray(toasts) || toasts.length === 0) return null;

  return (
    <div className="pointer-events-none fixed right-5 top-5 z-[120] w-[min(420px,calc(100vw-2.5rem))]">
      <div className="flex flex-col gap-2">
        {toasts.map((toast) => {
          const meta = getToastMeta(toast.type);
          const isLogToast = isExecutionLogToast(toast);
          const Icon = isLogToast ? Terminal : meta.icon;
          const frameAccent = isLogToast ? "var(--viz-palette-7-cyan)" : meta.accent;
          const remaining = clamp01((toast.expiresAt - nowMs) / Math.max(1, toast.durationMs || 1));

          return (
            <article
              key={toast.id}
              className="pointer-events-auto animate-spring-up overflow-hidden rounded-xl border backdrop-blur-md"
              style={{
                borderColor: isLogToast
                  ? "color-mix(in srgb, var(--viz-border), var(--viz-palette-7-cyan) 18%)"
                  : `color-mix(in srgb, ${meta.accent}, var(--viz-border) 55%)`,
                background: isLogToast
                  ? "linear-gradient(180deg, color-mix(in srgb, var(--viz-bg-surface), var(--viz-bg-canvas) 14%) 0%, color-mix(in srgb, var(--viz-bg-canvas), transparent 10%) 100%)"
                  : "linear-gradient(180deg, color-mix(in srgb, var(--viz-bg-surface), transparent 8%) 0%, color-mix(in srgb, var(--viz-bg-canvas), transparent 18%) 100%)",
                boxShadow:
                  "0 12px 34px rgba(0,0,0,0.38), inset 0 1px 0 color-mix(in srgb, var(--viz-text-primary), transparent 92%)",
              }}
              role="status"
              aria-live="polite"
            >
              <div className="flex items-start gap-2.5 px-3 py-2.5">
                <span
                  className="mt-0.5 inline-flex h-6 w-6 items-center justify-center rounded-lg border"
                  style={{
                    borderColor: `color-mix(in srgb, ${frameAccent}, transparent 55%)`,
                    color: frameAccent,
                    backgroundColor: `color-mix(in srgb, ${frameAccent}, transparent 88%)`,
                  }}
                >
                  <Icon size={14} />
                </span>

                <div className="min-w-0 flex-1">
                  <div className="mb-1 flex items-center gap-2">
                    <p
                      className="truncate text-[11px] font-bold uppercase tracking-[0.12em]"
                      style={{ color: isLogToast ? meta.accent : "var(--viz-text-primary)" }}
                    >
                      {toast.title}
                    </p>
                    {isLogToast ? (
                      <span
                        className="rounded-sm border px-1 py-0 text-[8px] font-bold uppercase tracking-[0.08em]"
                        style={{
                          borderColor: `color-mix(in srgb, ${meta.accent}, transparent 55%)`,
                          color: frameAccent,
                          backgroundColor: `color-mix(in srgb, ${frameAccent}, transparent 90%)`,
                        }}
                      >
                        LOG
                      </span>
                    ) : (
                      <span
                        className="rounded-md border px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-[0.12em]"
                        style={{
                          borderColor: `color-mix(in srgb, ${meta.accent}, transparent 55%)`,
                          color: meta.accent,
                        }}
                      >
                        {meta.label}
                      </span>
                    )}
                    <span className="ml-auto text-[9px] font-mono text-[var(--viz-text-muted)]">
                      {new Date(toast.createdAt).toLocaleTimeString("pt-BR", { hour12: false })}
                    </span>
                  </div>
                  <p className="line-clamp-2 text-[11px] text-[var(--viz-text-secondary)]">
                    {toast.message}
                  </p>
                </div>

                <button
                  type="button"
                  onClick={() => dismissToast(toast.id)}
                  className="btn-toggle mt-0.5 inline-flex h-6 w-6 items-center justify-center rounded-md border"
                  style={{
                    borderColor: "var(--viz-border)",
                    backgroundColor: "color-mix(in srgb, var(--viz-bg-canvas), transparent 18%)",
                    color: "var(--viz-text-muted)",
                  }}
                  aria-label="Fechar notificação"
                >
                  <X size={12} />
                </button>
              </div>

              <div
                className="h-1 w-full origin-left"
                style={{
                  background:
                    "linear-gradient(90deg, color-mix(in srgb, var(--viz-bg-canvas), var(--viz-bg-surface) 25%) 0%, color-mix(in srgb, var(--viz-bg-canvas), transparent 45%) 100%)",
                }}
              >
                <div
                  className="h-full transition-[width] duration-100 ease-linear"
                  style={{
                    width: `${remaining * 100}%`,
                    background: `linear-gradient(90deg, ${frameAccent} 0%, color-mix(in srgb, ${frameAccent}, white 18%) 100%)`,
                  }}
                />
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
};
