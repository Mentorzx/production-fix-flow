import React, { useState, useRef, useCallback } from "react";
import { createPortal } from "react-dom";

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
  interactive = true,
}) => {
  const [visible, setVisible] = useState(false);
  const [coords, setCoords] = useState({ top: 0, left: 0 });
  const [placement, setPlacement] = useState("above");
  const triggerRef = useRef(null);
  const tooltipRef = useRef(null);
  const isOverTrigger = useRef(false);
  const isOverTooltip = useRef(false);

  const checkHide = () => {
    if (!isOverTrigger.current && !isOverTooltip.current) {
      setVisible(false);
    }
  };

  const show = useCallback(() => {
    isOverTrigger.current = true;
    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      const tooltipHeight = 280;
      const tooltipHalfWidth = 144;
      const spaceAbove = rect.top;
      const centerX = rect.left + rect.width / 2;
      const clampedLeft = Math.max(
        tooltipHalfWidth + 8,
        Math.min(centerX, window.innerWidth - tooltipHalfWidth - 8)
      );

      if (spaceAbove < tooltipHeight + 16) {
        setPlacement("below");
        setCoords({ top: rect.bottom + 8, left: clampedLeft });
      } else {
        setPlacement("above");
        setCoords({ top: rect.top - 8, left: clampedLeft });
      }
      setVisible(true);
    }
  }, []);

  const hideTrigger = () => {
    isOverTrigger.current = false;
    if (interactive) {
      setTimeout(checkHide, 100);
    } else {
      setVisible(false);
    }
  };

  const transform = placement === "above" ? "translate(-50%, -100%)" : "translate(-50%, 0)";

  return (
    <React.Fragment>
      <div
        ref={triggerRef}
        onMouseEnter={show}
        onMouseLeave={hideTrigger}
        className={className}
        style={className?.includes("contents") ? { display: "contents" } : undefined}
      >
        {children}
      </div>
      {visible &&
        createPortal(
          <div
            ref={tooltipRef}
            onMouseEnter={
              interactive
                ? () => {
                    isOverTooltip.current = true;
                  }
                : undefined
            }
            onMouseLeave={
              interactive
                ? () => {
                    isOverTooltip.current = false;
                    setTimeout(checkHide, 100);
                  }
                : undefined
            }
            className={`fixed z-[99999] animate-in fade-in zoom-in-95 duration-100 ${interactive ? "pointer-events-auto" : "pointer-events-none"}`}
            style={{
              top: coords.top,
              left: coords.left,
              transform,
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
