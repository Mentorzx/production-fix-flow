/**
 * Global command palette for dashboard navigation.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { Search, X } from "./icons.jsx";
import { initSearchEngine, searchWithEngine } from "../search/engine.js";
import { navigateToSuggestion } from "../search/navigation.js";

const TOP_K = 10;

const isEditableElement = (target) => {
  if (!target || !(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName?.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select";
};

const toDefaultSuggestions = (catalog) => {
  return [...catalog]
    .sort((left, right) => left.title.localeCompare(right.title))
    .slice(0, TOP_K)
    .map((item) => ({
      id: item.id,
      domId: item.domId,
      title: item.title,
      snippet: item.description,
      score: 0,
      tabId: item.tabId,
      viewMode: item.viewMode,
      sectionPath: item.sectionPath,
      sectionKey: item.sectionKey || "",
      reason: "catalog",
    }));
};

/**
 * Expose command palette for dashboard usage.
 */
export const CommandPalette = ({
  open,
  onOpenChange,
  catalog,
  setActiveTab,
  setViewMode,
  launcherRef,
}) => {
  const inputRef = useRef(null);
  const listRef = useRef(null);
  const prevOpenRef = useRef(open);
  const openedByTypingRef = useRef(false);

  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const [engine, setEngine] = useState(null);

  const defaultSuggestions = useMemo(() => toDefaultSuggestions(catalog || []), [catalog]);

  useEffect(() => {
    let canceled = false;
    initSearchEngine(catalog || []).then((ctx) => {
      if (!canceled) {
        setEngine(ctx);
      }
    });
    return () => {
      canceled = true;
    };
  }, [catalog]);

  const suggestions = useMemo(() => {
    if (!open) return [];
    if (!query.trim()) return defaultSuggestions;
    const result = searchWithEngine(engine, query, TOP_K);
    return Array.isArray(result) ? result.slice(0, TOP_K) : [];
  }, [open, query, defaultSuggestions, engine]);

  useEffect(() => {
    setActiveIndex(0);
  }, [query, open]);

  useEffect(() => {
    if (!open) return;
    window.requestAnimationFrame(() => {
      inputRef.current?.focus();
      if (openedByTypingRef.current) {
        const length = inputRef.current?.value?.length ?? 0;
        inputRef.current?.setSelectionRange(length, length);
      } else {
        inputRef.current?.select();
      }
      openedByTypingRef.current = false;
    });
  }, [open]);

  useEffect(() => {
    const handleGlobalKeydown = (event) => {
      const isCtrlK = (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k";
      const isSlashShortcut = event.key === "/" && !event.ctrlKey && !event.metaKey;
      const isPrintableTypeStart =
        !open &&
        !event.ctrlKey &&
        !event.metaKey &&
        !event.altKey &&
        event.key.length === 1 &&
        /\S/u.test(event.key);
      const isPrintableWhileOpen =
        open &&
        !event.ctrlKey &&
        !event.metaKey &&
        !event.altKey &&
        event.key.length === 1 &&
        /\S/u.test(event.key);

      if (isCtrlK) {
        event.preventDefault();
        openedByTypingRef.current = false;
        setQuery("");
        onOpenChange(true);
        return;
      }

      if (isSlashShortcut && !open && !isEditableElement(event.target)) {
        event.preventDefault();
        openedByTypingRef.current = false;
        setQuery("");
        onOpenChange(true);
        return;
      }

      if (isPrintableTypeStart && !isEditableElement(event.target)) {
        event.preventDefault();
        openedByTypingRef.current = true;
        setQuery(event.key);
        onOpenChange(true);
        return;
      }

      if (isPrintableWhileOpen && !isEditableElement(event.target)) {
        event.preventDefault();
        setQuery((current) => `${current}${event.key}`);
        return;
      }

      if (event.key === "Escape" && open) {
        event.preventDefault();
        onOpenChange(false);
      }
    };

    window.addEventListener("keydown", handleGlobalKeydown);
    return () => window.removeEventListener("keydown", handleGlobalKeydown);
  }, [open, onOpenChange]);

  useEffect(() => {
    if (prevOpenRef.current && !open) {
      setQuery("");
      setActiveIndex(0);
      if (launcherRef?.current) {
        launcherRef.current.focus();
      }
    }
    prevOpenRef.current = open;
  }, [open, launcherRef]);

  const handleSelect = (suggestion) => {
    if (!suggestion) return;
    navigateToSuggestion(suggestion, { setActiveTab, setViewMode });
    onOpenChange(false);
  };

  const onInputKeyDown = (event) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((prev) => Math.min(prev + 1, Math.max(suggestions.length - 1, 0)));
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((prev) => Math.max(prev - 1, 0));
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      handleSelect(suggestions[activeIndex] || suggestions[0]);
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      onOpenChange(false);
    }
  };

  useEffect(() => {
    const listElement = listRef.current;
    if (!listElement) return;
    const node = listElement.querySelector(`[data-suggestion-index="${activeIndex}"]`);
    if (node) {
      node.scrollIntoView({ block: "nearest" });
    }
  }, [activeIndex]);

  if (!open) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-[130]" role="presentation" onClick={() => onOpenChange(false)}>
      <div className="absolute inset-0 bg-black/55 backdrop-blur-[3px] command-palette-backdrop" />
      <div className="absolute inset-x-0 top-20 px-4 md:px-10 lg:px-16">
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Busca global de gráficos e tabelas"
          className="mx-auto w-full max-w-3xl rounded-2xl border overflow-hidden command-palette-enter"
          style={{
            borderColor: "color-mix(in srgb, var(--viz-palette-4-yellow), transparent 58%)",
            background:
              "linear-gradient(180deg, color-mix(in srgb, var(--viz-bg-surface), transparent 4%) 0%, color-mix(in srgb, var(--viz-bg-canvas), transparent 9%) 100%)",
            boxShadow: "0 30px 80px rgba(0, 0, 0, 0.45)",
          }}
          onClick={(event) => event.stopPropagation()}
        >
          <div className="flex items-center gap-3 border-b px-4 py-3" style={{ borderColor: "var(--viz-border)" }}>
            <Search size={16} style={{ color: "var(--viz-palette-4-yellow)" }} />
            <input
              ref={inputRef}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={onInputKeyDown}
              placeholder="Buscar gráficos, tabelas e cards..."
              className="w-full bg-transparent outline-none text-sm"
              style={{ color: "var(--viz-text-primary)" }}
              aria-autocomplete="list"
              aria-controls="dashboard-command-palette-list"
              aria-activedescendant={
                suggestions[activeIndex] ? `dashboard-command-item-${suggestions[activeIndex].id}` : undefined
              }
            />
            <button
              type="button"
              className="inline-flex items-center justify-center rounded-md p-1.5"
              style={{
                border: "1px solid color-mix(in srgb, var(--viz-border), transparent 8%)",
                color: "var(--viz-text-muted)",
              }}
              onClick={() => onOpenChange(false)}
              aria-label="Fechar busca"
            >
              <X size={14} />
            </button>
          </div>

          <div
            className="max-h-[58vh] overflow-y-auto custom-scrollbar"
            id="dashboard-command-palette-list"
            role="listbox"
            ref={listRef}
          >
            {suggestions.length === 0 && (
              <div className="px-4 py-8 text-center text-sm" style={{ color: "var(--viz-text-muted)" }}>
                Nenhum resultado encontrado.
              </div>
            )}

            {suggestions.map((suggestion, index) => {
              const selected = index === activeIndex;
              return (
                <button
                  key={suggestion.id}
                  id={`dashboard-command-item-${suggestion.id}`}
                  type="button"
                  role="option"
                  aria-selected={selected}
                  data-suggestion-index={index}
                  className="w-full text-left px-4 py-3 border-b transition-colors duration-150"
                  style={{
                    borderColor: "color-mix(in srgb, var(--viz-border), transparent 16%)",
                    backgroundColor: selected
                      ? "color-mix(in srgb, var(--viz-palette-4-yellow), transparent 90%)"
                      : "transparent",
                  }}
                  onMouseEnter={() => setActiveIndex(index)}
                  onClick={() => handleSelect(suggestion)}
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div
                        className="text-sm font-semibold truncate"
                        style={{
                          color: selected
                            ? "var(--viz-palette-4-yellow)"
                            : "var(--viz-text-primary)",
                        }}
                      >
                        {suggestion.title}
                      </div>
                      <div className="text-xs truncate mt-1" style={{ color: "var(--viz-text-muted)" }}>
                        {suggestion.sectionPath || "Sem seção"}
                      </div>
                    </div>
                    <div className="text-[10px] uppercase font-mono" style={{ color: "var(--viz-text-muted)" }}>
                      {suggestion.tabId}/{suggestion.viewMode}
                    </div>
                  </div>
                  {suggestion.snippet ? (
                    <div
                      className="text-xs mt-1.5 overflow-hidden"
                      style={{ color: "var(--viz-text-secondary)", maxHeight: "2.6em" }}
                    >
                      {suggestion.snippet}
                    </div>
                  ) : null}
                </button>
              );
            })}
          </div>

          <div
            className="px-4 py-2 text-[10px] font-mono border-t flex items-center justify-between"
            style={{ borderColor: "var(--viz-border)", color: "var(--viz-text-muted)" }}
          >
            <span>↑/↓ navegar • Enter abrir • Esc fechar</span>
            <span>Ctrl+K</span>
          </div>
        </div>
      </div>
    </div>
  );
};
