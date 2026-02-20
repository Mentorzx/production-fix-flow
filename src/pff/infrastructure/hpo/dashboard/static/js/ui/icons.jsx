/**
 * Provide icons module functionality for the HPO dashboard.
 */

import { Theme } from "./Theme.js";

/**
 * SVG icon base component and Feather-style icon library.
 */

/** @type {React.FC<{d: React.ReactNode, className?: string, size?: number, style?: React.CSSProperties}>} */
export const IconBase = ({ d, className = "", size = 16, style = {} }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
    style={{ shapeRendering: "geometricPrecision", ...style }}
    aria-hidden="true"
    focusable="false"
  >
    {d}
  </svg>
);

/**
 * Expose activity for dashboard usage.
 */
export const Activity = (p) => <IconBase {...p} d={<path d="M22 12h-4l-3 9L9 3l-3 9H2" />} />;
/**
 * Expose refresh cw for dashboard usage.
 */
export const RefreshCw = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
        <path d="M21 3v5h-5" />
        <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
        <path d="M8 16H3v5" />
      </>
    }
  />
);
/**
 * Expose zap for dashboard usage.
 */
export const Zap = (p) => <IconBase {...p} d={<path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />} />;
/**
 * Expose clock for dashboard usage.
 */
export const Clock = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <circle cx="12" cy="12" r="10" />
        <polyline points="12 6 12 12 16 14" />
      </>
    }
  />
);
/**
 * Expose trending up for dashboard usage.
 */
export const TrendingUp = (p) => (
  <IconBase {...p} d={<polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />} />
);
/**
 * Expose layers for dashboard usage.
 */
export const Layers = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <polygon points="12 2 2 7 12 12 22 7 12 2" />
        <polyline points="2 17 12 22 22 17" />
        <polyline points="2 12 12 17 22 12" />
      </>
    }
  />
);
/**
 * Expose table icon for dashboard usage.
 */
export const TableIcon = (p) => (
  <IconBase
    {...p}
    d={
      <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18" />
    }
  />
);
/**
 * Expose sliders for dashboard usage.
 */
export const Sliders = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <line x1="4" y1="21" x2="4" y2="14" />
        <line x1="4" y1="10" x2="4" y2="3" />
        <line x1="12" y1="21" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12" y2="3" />
        <line x1="20" y1="21" x2="20" y2="16" />
        <line x1="20" y1="12" x2="20" y2="3" />
        <line x1="1" y1="14" x2="7" y2="14" />
        <line x1="9" y1="8" x2="15" y2="8" />
        <line x1="17" y1="16" x2="23" y2="16" />
      </>
    }
  />
);
/**
 * Expose settings for dashboard usage.
 */
export const Settings = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83l-.06.06a2 2 0 0 1-2.83 0l-.06-.06A1.65 1.65 0 0 0 15 19.4a1.65 1.65 0 0 0-1 1.5V21a2 2 0 0 1-4 0v-.1a1.65 1.65 0 0 0-1-1.5 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0l-.06-.06a2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.5-1H3a2 2 0 0 1 0-4h.1a1.65 1.65 0 0 0 1.5-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83l.06-.06a2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.5V3a2 2 0 0 1 4 0v.1a1.65 1.65 0 0 0 1 1.5 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0l.06.06a2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9c.19.32.86.73 1.5 1H21a2 2 0 0 1 0 4h-.1a1.65 1.65 0 0 0-1.5 1Z" />
      </>
    }
  />
);
/**
 * Expose git merge for dashboard usage.
 */
export const GitMerge = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <circle cx="18" cy="18" r="3" />
        <circle cx="6" cy="6" r="3" />
        <path d="M6 21V9a9 9 0 0 0 9 9" />
      </>
    }
  />
);
/**
 * Expose download for dashboard usage.
 */
export const Download = (p) => (
  <IconBase
    {...p}
    d={<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" />}
  />
);
/**
 * Expose target icon for dashboard usage.
 */
export const TargetIcon = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <circle cx="12" cy="12" r="10" />
        <circle cx="12" cy="12" r="6" />
        <circle cx="12" cy="12" r="2" />
      </>
    }
  />
);
/**
 * Expose cpu for dashboard usage.
 */
export const Cpu = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <rect x="4" y="4" width="16" height="16" rx="2" ry="2" />
        <rect x="9" y="9" width="6" height="6" />
        <line x1="9" y1="1" x2="9" y2="4" />
        <line x1="15" y1="1" x2="15" y2="4" />
        <line x1="9" y1="20" x2="9" y2="23" />
        <line x1="15" y1="20" x2="15" y2="23" />
        <line x1="20" y1="9" x2="23" y2="9" />
        <line x1="20" y1="12" x2="23" y2="14" />
        <line x1="1" y1="9" x2="4" y2="9" />
        <line x1="1" y1="14" x2="4" y2="14" />
      </>
    }
  />
);
/**
 * Expose microscope for dashboard usage.
 */
export const Microscope = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <path d="M6 18h8" />
        <path d="M3 22h18" />
        <path d="M14 22a7 7 0 1 0 0-14h-1" />
        <path d="M9 14h2" />
        <path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z" />
        <path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3" />
      </>
    }
  />
);
/**
 * Expose share2 for dashboard usage.
 */
export const Share2 = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <circle cx="18" cy="5" r="3" />
        <circle cx="6" cy="12" r="3" />
        <circle cx="18" cy="19" r="3" />
        <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
        <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
      </>
    }
  />
);
/**
 * Expose alert triangle for dashboard usage.
 */
export const AlertTriangle = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
        <line x1="12" y1="9" x2="12" y2="13" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </>
    }
  />
);
/**
 * Expose alert octagon for dashboard usage.
 */
export const AlertOctagon = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </>
    }
  />
);
/**
 * Expose check circle for dashboard usage.
 */
export const CheckCircle = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
        <polyline points="22 4 12 14.01 9 11.01" />
      </>
    }
  />
);
/**
 * Expose x for dashboard usage.
 */
export const X = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <line x1="18" y1="6" x2="6" y2="18" />
        <line x1="6" y1="6" x2="18" y2="18" />
      </>
    }
  />
);
/**
 * Expose chevron right for dashboard usage.
 */
export const ChevronRight = (p) => <IconBase {...p} d={<polyline points="9 18 15 12 9 6" />} />;
/**
 * Expose bell for dashboard usage.
 */
export const Bell = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <path d="M18 8a6 6 0 0 0-12 0c0 7-3 8-3 8h18s-3-1-3-8" />
        <path d="M13.73 21a2 2 0 0 1-3.46 0" />
      </>
    }
  />
);
/**
 * Expose search for dashboard usage.
 */
export const Search = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <circle cx="11" cy="11" r="8" />
        <line x1="21" y1="21" x2="16.65" y2="16.65" />
      </>
    }
  />
);
/**
 * Expose bar chart2 for dashboard usage.
 */
export const BarChart2 = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <line x1="18" y1="20" x2="18" y2="10" />
        <line x1="12" y1="20" x2="12" y2="4" />
        <line x1="6" y1="20" x2="6" y2="14" />
      </>
    }
  />
);
/**
 * Expose terminal for dashboard usage.
 */
export const Terminal = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <polyline points="4 17 10 11 4 5" />
        <line x1="12" y1="19" x2="20" y2="19" />
      </>
    }
  />
);
/**
 * Expose info for dashboard usage.
 */
export const Info = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="16" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12.01" y2="8" />
      </>
    }
  />
);
/**
 * Expose help circle for dashboard usage.
 */
export const HelpCircle = (p) => (
  <IconBase
    {...p}
    d={
      <>
        <circle cx="12" cy="12" r="10" />
        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </>
    }
  />
);

/**
 * Expose arrow up for dashboard usage.
 */
export const ArrowUp = ({ className = "", size = 10 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke={Theme.semantic.success}
    strokeWidth="3"
    className={className}
  >
    <polyline points="18 15 12 9 6 15" />
  </svg>
);
/**
 * Expose arrow down for dashboard usage.
 */
export const ArrowDown = ({ className = "", size = 10 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke={Theme.semantic.chart.loss}
    strokeWidth="3"
    className={className}
  >
    <polyline points="6 9 12 15 18 9" />
  </svg>
);
