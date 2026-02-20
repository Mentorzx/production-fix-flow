/**
 * Provide ThemeToggle module functionality for the HPO dashboard.
 */

import { useTheme } from "./ThemeContext";

/**
 * Expose theme toggle for dashboard usage.
 */
export const ThemeToggle = ({ className = "" }) => {
  const { theme, toggleTheme } = useTheme();
  const isDark = theme === "dark";

  return (
    <button
      onClick={toggleTheme}
      className={`btn-theme relative p-2 rounded-full transition-all duration-300 hover:bg-black/5 dark:hover:bg-white/10 focus:outline-none overflow-hidden group ${className}`}
      aria-label="Toggle Theme"
      aria-pressed={isDark}
      data-state={isDark ? "active" : "inactive"}
      title={`Switch to ${isDark ? "Light" : "Dark"} Mode`}
      style={{
        "--viz-icon-active": isDark ? "var(--viz-palette-4-yellow)" : "var(--viz-palette-3-orange)",
        color: isDark ? "var(--viz-text-primary)" : "var(--viz-text-muted)",
        borderColor: "var(--viz-border)",
      }}
    >
      <div className="relative w-5 h-5">
        {/* Sun Icon */}
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className={`absolute inset-0 transition-all duration-500 ease-in-out transform origin-center ${
            isDark ? "rotate-90 opacity-0 scale-50" : "rotate-0 opacity-100 scale-100"
          }`}
        >
          <circle cx="12" cy="12" r="5"></circle>
          <line x1="12" y1="1" x2="12" y2="3"></line>
          <line x1="12" y1="21" x2="12" y2="23"></line>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
          <line x1="1" y1="12" x2="3" y2="12"></line>
          <line x1="21" y1="12" x2="23" y2="12"></line>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
        </svg>

        {/* Moon Icon */}
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className={`absolute inset-0 transition-all duration-500 ease-in-out transform origin-center ${
            isDark ? "rotate-0 opacity-100 scale-100" : "-rotate-90 opacity-0 scale-50"
          }`}
        >
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
        </svg>
      </div>
    </button>
  );
};
