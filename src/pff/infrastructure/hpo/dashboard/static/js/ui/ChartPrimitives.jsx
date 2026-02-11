import React from "react";
import { Tooltip, CartesianGrid } from "recharts";
import { Theme } from "./Theme.js";
import { useState, useEffect, useRef } from "react";

/**
 * Chart-level primitives: colors adapter, default grid/tooltip, responsive container.
 */

export const colors = {
  bg: Theme.ui.background,
  card: Theme.ui.surface,
  border: Theme.ui.border,
  text: Theme.ui.text.secondary,
  textHigh: Theme.ui.text.primary,
  primary: Theme.semantic.primary,
  orange: Theme.palette.hotOrange,
  success: Theme.semantic.success,
  lime: Theme.palette.lime,
  amber: Theme.palette.cyberYellow,
  error: Theme.semantic.error,
  warning: Theme.semantic.warning,
  grid: Theme.ui.grid,
  tooltip: Theme.ui.background,
};

export const defaultChartMargins = { top: 25, right: 15, bottom: 35, left: 50 };

export const defaultTooltipStyle = {
  backgroundColor: Theme.ui.background,
  borderColor: Theme.ui.border,
  color: Theme.ui.text.primary,
  fontSize: "11px",
  borderRadius: "8px",
  boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.5)",
};

export const DefaultCartesianGrid = React.memo((props) => (
  <CartesianGrid strokeDasharray="3 3" stroke={Theme.ui.grid} strokeOpacity={0.5} {...props} />
));

export const DefaultTooltip = React.memo((props) => {
  const { wrapperStyle, ...rest } = props;
  return (
    <Tooltip
      contentStyle={defaultTooltipStyle}
      itemStyle={{ color: Theme.ui.text.secondary }}
      cursor={{ stroke: Theme.ui.grid, strokeDasharray: "3 3" }}
      wrapperStyle={{ zIndex: 60, ...wrapperStyle }}
      {...rest}
    />
  );
});

export const DefaultTooltipCursor = { strokeDasharray: "3 3", stroke: Theme.ui.grid };

export const ChartFrame = React.memo(({ children, className = "" }) => (
  <div className={`relative p-5 w-full h-full min-h-[220px] min-w-0 ${className}`}>{children}</div>
));

export const ChartContainer = React.memo(({ children, className = "", minHeight = 200 }) => {
  const containerRef = useRef(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      if (width > 0 && height > 0) {
        setSize({ width, height });
      } else {
        setSize({ width: 0, height: 0 });
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const ready = size.width > 0 && size.height > 0;
  const child = React.Children.only(children);

  return (
    <div
      ref={containerRef}
      className={`w-full h-full min-w-0 ${className}`}
      style={{ minWidth: 0, minHeight }}
    >
      {ready &&
        React.isValidElement(child) &&
        React.cloneElement(child, { width: size.width, height: size.height })}
    </div>
  );
});
