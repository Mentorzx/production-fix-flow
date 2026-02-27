/**
 * Provide PortalTooltip module functionality for the HPO dashboard.
 */

import React, { useState, useRef, useCallback, useEffect } from "react";
import { createPortal } from "react-dom";

const TOOLTIP_OPEN_EVENT = "pff:portal-tooltip-open";
let tooltipInstanceSeq = 0;

/**
 * Unified portal-based tooltip with optional interactive mode.
 *
 * Auto-positions above or below trigger based on available viewport space.
 * Clamps horizontally to stay within viewport bounds.
 *
 * @param {Object} props
 * @param {React.ReactNode} props.content - Tooltip body
 * @param {React.ReactNode} props.children - Trigger element
 * @param {string} [props.className] - CSS class for trigger wrapper
 * @param {boolean} [props.interactive] - When true, tooltip stays open while hovered (links, copy, etc.)
 */
export const PortalTooltip = ({
  content,
  children,
  className = "inline-block",
  interactive = false,
}) => {
  const instanceIdRef = useRef(`pff-tooltip-${++tooltipInstanceSeq}`);
  const [visible, setVisible] = useState(false);
  const [coords, setCoords] = useState({ top: 0, left: 0 });
  const [placement, setPlacement] = useState("above");
  const triggerRef = useRef(null);
  const tooltipRef = useRef(null);
  const triggerRectRef = useRef(null);
  const isOverTrigger = useRef(false);
  const isOverTooltip = useRef(false);
  const hideTimerRef = useRef(0);

  const computePlacement = useCallback((rect, tooltipWidth = 288, tooltipHeight = 280) => {
    const margin = 8;
    const halfWidth = Math.max(48, tooltipWidth / 2);
    const centerX = rect.left + rect.width / 2;
    const left = Math.max(
      halfWidth + margin,
      Math.min(centerX, window.innerWidth - halfWidth - margin)
    );
    const maxUsableHeight = Math.max(120, window.innerHeight - margin * 2);
    if (tooltipHeight >= maxUsableHeight) {
      return { placement: "below", top: margin, left };
    }

    const availableAbove = rect.top - margin;
    const availableBelow = window.innerHeight - rect.bottom - margin;
    const placeBelow = availableAbove < tooltipHeight && availableBelow >= tooltipHeight;
    const placementNext = placeBelow ? "below" : "above";
    let top = placeBelow ? rect.bottom + 8 : rect.top - 8;

    if (placementNext === "above") {
      const topEdge = top - tooltipHeight;
      if (topEdge < margin) {
        if (availableBelow > availableAbove) {
          top = rect.bottom + 8;
          return { placement: "below", top, left };
        }
        top = tooltipHeight + margin;
      }
    } else if (top + tooltipHeight > window.innerHeight - margin) {
      if (availableAbove > availableBelow) {
        top = rect.top - 8;
        return { placement: "above", top, left };
      }
      top = Math.max(margin, window.innerHeight - margin - tooltipHeight);
    }

    return { placement: placementNext, top, left };
  }, []);

  const checkHide = () => {
    if (!isOverTrigger.current && !isOverTooltip.current) {
      setVisible(false);
    }
  };

  const clearHideTimer = () => {
    if (!hideTimerRef.current) return;
    clearTimeout(hideTimerRef.current);
    hideTimerRef.current = 0;
  };

  const hideImmediately = useCallback(() => {
    clearHideTimer();
    isOverTrigger.current = false;
    isOverTooltip.current = false;
    setVisible(false);
  }, []);

  const scheduleHide = useCallback(
    (delayMs = 80) => {
      clearHideTimer();
      hideTimerRef.current = window.setTimeout(() => {
        const triggerStillHovered = Boolean(triggerRef.current?.matches?.(":hover"));
        const tooltipStillHovered = interactive
          ? Boolean(tooltipRef.current?.matches?.(":hover"))
          : false;
        isOverTrigger.current = triggerStillHovered;
        isOverTooltip.current = tooltipStillHovered;
        if (!triggerStillHovered && !tooltipStillHovered) hideImmediately();
      }, delayMs);
    },
    [hideImmediately, interactive]
  );

  const show = useCallback(() => {
    clearHideTimer();
    isOverTrigger.current = true;
    isOverTooltip.current = false;
    if (triggerRef.current) {
      window.dispatchEvent(
        new CustomEvent(TOOLTIP_OPEN_EVENT, {
          detail: { id: instanceIdRef.current },
        })
      );
      const rect = triggerRef.current.getBoundingClientRect();
      triggerRectRef.current = rect;
      const next = computePlacement(rect);
      setPlacement(next.placement);
      setCoords({ top: next.top, left: next.left });
      setVisible(true);
    }
  }, [computePlacement]);

  useEffect(() => {
    if (!visible) return;
    const triggerRect = triggerRectRef.current;
    const tooltipElement = tooltipRef.current;
    if (!triggerRect || !tooltipElement) return;

    const measuredWidth = tooltipElement.offsetWidth || 288;
    const measuredHeight = tooltipElement.offsetHeight || 280;
    const next = computePlacement(triggerRect, measuredWidth, measuredHeight);

    setPlacement((prev) => (prev === next.placement ? prev : next.placement));
    setCoords((prev) => {
      const sameTop = Math.abs(prev.top - next.top) <= 1;
      const sameLeft = Math.abs(prev.left - next.left) <= 1;
      if (sameTop && sameLeft) return prev;
      return { top: next.top, left: next.left };
    });
  }, [visible, computePlacement]);

  const hideTrigger = () => {
    scheduleHide(interactive ? 90 : 45);
  };

  useEffect(() => {
    const closeWhenAnotherOpens = (event) => {
      const openerId = event?.detail?.id;
      if (!openerId || openerId === instanceIdRef.current) return;
      hideImmediately();
    };

    window.addEventListener(TOOLTIP_OPEN_EVENT, closeWhenAnotherOpens);
    return () => {
      window.removeEventListener(TOOLTIP_OPEN_EVENT, closeWhenAnotherOpens);
    };
  }, [hideImmediately]);

  useEffect(() => {
    if (!visible) return undefined;

    const syncHoverState = (eventTarget) => {
      const triggerEl = triggerRef.current;
      if (!triggerEl) {
        hideImmediately();
        return false;
      }

      const overTrigger =
        eventTarget instanceof Node ? triggerEl.contains(eventTarget) : triggerEl.matches(":hover");
      const tooltipEl = tooltipRef.current;
      const overTooltip =
        interactive && tooltipEl
          ? eventTarget instanceof Node
            ? tooltipEl.contains(eventTarget)
            : tooltipEl.matches(":hover")
          : false;

      isOverTrigger.current = Boolean(overTrigger) || Boolean(triggerEl.matches?.(":hover"));
      isOverTooltip.current =
        interactive && (Boolean(overTooltip) || Boolean(tooltipEl?.matches?.(":hover")));

      return isOverTrigger.current || isOverTooltip.current;
    };

    const handlePointerMove = (event) => {
      const stillHovering = syncHoverState(event.target);
      if (stillHovering) {
        clearHideTimer();
      } else {
        scheduleHide(35);
      }
    };

    const handleGlobalHide = () => {
      hideImmediately();
    };

    const handleEscape = (event) => {
      if (event.key === "Escape") hideImmediately();
    };

    const handleVisibility = () => {
      if (document.hidden) hideImmediately();
    };

    window.addEventListener("pointermove", handlePointerMove, true);
    window.addEventListener("scroll", handleGlobalHide, true);
    window.addEventListener("blur", handleGlobalHide);
    window.addEventListener("resize", handleGlobalHide);
    document.addEventListener("keydown", handleEscape, true);
    document.addEventListener("visibilitychange", handleVisibility);

    return () => {
      window.removeEventListener("pointermove", handlePointerMove, true);
      window.removeEventListener("scroll", handleGlobalHide, true);
      window.removeEventListener("blur", handleGlobalHide);
      window.removeEventListener("resize", handleGlobalHide);
      document.removeEventListener("keydown", handleEscape, true);
      document.removeEventListener("visibilitychange", handleVisibility);
      clearHideTimer();
    };
  }, [interactive, scheduleHide, hideImmediately, visible]);

  useEffect(
    () => () => {
      clearHideTimer();
    },
    []
  );

  const transform = placement === "above" ? "translate(-50%, -100%)" : "translate(-50%, 0)";

  return (
    <React.Fragment>
      <div
        ref={triggerRef}
        data-pff-tooltip-trigger="1"
        data-pff-tooltip-id={instanceIdRef.current}
        onMouseEnter={() => {
          clearHideTimer();
          show();
        }}
        onMouseLeave={() => {
          hideTrigger();
        }}
        className={className}
        style={className?.includes("contents") ? { display: "contents" } : undefined}
      >
        {children}
      </div>
      {visible &&
        createPortal(
          <div
            ref={tooltipRef}
            data-pff-tooltip-root="1"
            data-pff-tooltip-id={instanceIdRef.current}
            onMouseEnter={
              interactive
                ? () => {
                    clearHideTimer();
                    isOverTooltip.current = true;
                  }
                : undefined
            }
            onMouseLeave={
              interactive
                ? () => {
                    isOverTooltip.current = false;
                    clearHideTimer();
                    hideTimerRef.current = window.setTimeout(checkHide, 100);
                  }
                : undefined
            }
            className={`fixed z-[99999] animate-in fade-in zoom-in-95 duration-100 ${interactive ? "pointer-events-auto" : "pointer-events-none"}`}
            style={{
              top: coords.top,
              left: coords.left,
              transform,
              maxHeight: "calc(100vh - 16px)",
              overflowY: "auto",
              filter: interactive ? "drop-shadow(0 4px 20px rgba(0,0,0,0.5))" : undefined,
            }}
          >
            {content}
          </div>,
          document.body
        )}
    </React.Fragment>
  );
};
