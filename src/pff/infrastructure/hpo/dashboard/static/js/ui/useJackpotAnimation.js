import { useEffect } from "react";

const MAX_ANIMATED_NODES = 320;
const FRAME_MS = 56;
const DURATION_MS = 1120;

const SELECTOR = "h1,h2,h3,h4,h5,p,span,small,strong,td,th,div,button,li";
const FORCE_SCOPE_SELECTOR = "[data-jackpot-force='true']";
const SCOPED_SELECTOR = `${FORCE_SCOPE_SELECTOR} :is(${SELECTOR})`;

const hasDigits = (text) => /\d/.test(text || "");
const randomDigit = () => String(Math.floor(Math.random() * 10));

const scrambleNumberText = (target, progress) => {
  const revealThreshold = Math.max(0, Math.min(1, progress));
  let digitIndex = 0;
  return target
    .split("")
    .map((ch) => {
      if (!/\d/.test(ch)) return ch;
      digitIndex += 1;
      const localReveal = digitIndex / Math.max(1, target.replace(/\D/g, "").length);
      return revealThreshold >= localReveal ? ch : randomDigit();
    })
    .join("");
};

const isCandidate = (el) => {
  if (!el || !(el instanceof HTMLElement)) return false;
  if (el.closest("[data-jackpot-skip='true']")) return false;
  if (el.childElementCount > 0) return false;
  if (el.offsetParent === null) return false;
  const text = (el.textContent || "").trim();
  if (!text || text.length > 48) return false;
  if (!hasDigits(text)) return false;
  return true;
};

/**
 * Applies slot/jackpot-style digit animation to visible numeric nodes on view/data changes.
 */
export const useJackpotAnimation = (rootRef, triggerKey = "") => {
  useEffect(() => {
    const root = rootRef?.current;
    if (!root) return undefined;
    if (
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches
    ) {
      return undefined;
    }

    const nodes = Array.from(root.querySelectorAll(SCOPED_SELECTOR))
      .filter(isCandidate)
      .slice(0, MAX_ANIMATED_NODES);

    const timers = [];

    for (const node of nodes) {
      const target = (node.textContent || "").trim();
      const previous = node.dataset.jackpotPrev || "";
      const previousTrigger = node.dataset.jackpotPrevTrigger || "";
      const forceOnTrigger = Boolean(node.closest("[data-jackpot-force='true']"));
      const triggerChanged = previousTrigger !== String(triggerKey || "");
      if (previous === target && !(forceOnTrigger && triggerChanged)) continue;
      node.dataset.jackpotPrev = target;
      node.dataset.jackpotPrevTrigger = String(triggerKey || "");

      let elapsed = 0;
      node.style.willChange = "contents, filter, opacity";
      node.style.filter =
        "drop-shadow(0 0 5px color-mix(in srgb, var(--viz-palette-4-yellow), transparent 62%))";
      node.style.opacity = "0.985";

      const interval = setInterval(() => {
        elapsed += FRAME_MS;
        const progress = Math.min(1, elapsed / DURATION_MS);
        node.textContent = scrambleNumberText(target, progress);
        if (progress >= 1) {
          clearInterval(interval);
          node.textContent = target;
          node.style.willChange = "";
          node.style.filter = "";
          node.style.opacity = "";
        }
      }, FRAME_MS);

      timers.push(interval);
    }

    return () => {
      for (const t of timers) clearInterval(t);
    };
  }, [rootRef, triggerKey]);
};
