/**
 * Navigation helpers for command palette suggestions.
 */

const HIGHLIGHT_CLASS = "search-focus-highlight";
const HIGHLIGHT_DURATION_MS = 800;
const MAX_SCROLL_ATTEMPTS = 14;
const RETRY_DELAY_MS = 70;

const openSectionIfCollapsed = (sectionKey) => {
  if (!sectionKey) return;
  const toggle = document.querySelector(`[data-section-toggle="${sectionKey}"]`);
  if (!toggle) return;
  const expanded = toggle.getAttribute("aria-expanded") === "true";
  if (!expanded) {
    toggle.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  }
};

const findAnchor = (domId) => {
  if (!domId) return null;
  return (
    document.getElementById(domId) ||
    document.querySelector(`[data-search-id="${domId}"]`) ||
    document.querySelector(`[data-search-id="${domId.replace(/^search-/, "")}"]`)
  );
};

const applyHighlight = (target) => {
  target.classList.add(HIGHLIGHT_CLASS);
  window.setTimeout(() => {
    target.classList.remove(HIGHLIGHT_CLASS);
  }, HIGHLIGHT_DURATION_MS);
};

const focusSuggestion = (suggestion) => {
  let attempts = 0;

  const tick = () => {
    attempts += 1;
    const target = findAnchor(suggestion.domId);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
      applyHighlight(target);
      return;
    }

    if (attempts < MAX_SCROLL_ATTEMPTS) {
      window.setTimeout(tick, RETRY_DELAY_MS);
    }
  };

  tick();
};

/**
 * Navigate to a selected suggestion by changing view/tab then scrolling to anchor.
 */
export const navigateToSuggestion = (suggestion, actions) => {
  if (!suggestion || !actions) return;

  if (suggestion.viewMode && actions.setViewMode) {
    actions.setViewMode(suggestion.viewMode);
  }
  if (suggestion.tabId && actions.setActiveTab) {
    actions.setActiveTab(suggestion.tabId);
  }

  openSectionIfCollapsed(suggestion.sectionKey);
  window.requestAnimationFrame(() => {
    openSectionIfCollapsed(suggestion.sectionKey);
    focusSuggestion(suggestion);
  });
};
