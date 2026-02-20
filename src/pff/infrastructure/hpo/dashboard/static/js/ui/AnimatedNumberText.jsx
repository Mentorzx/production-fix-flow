import { useEffect, useMemo, useRef, useState } from "react";

const DIGIT_REGEX = /\d/;

const hasDigits = (text) => DIGIT_REGEX.test(String(text || ""));
const randomDigit = () => String(Math.floor(Math.random() * 10));

const scrambleDigits = (target, progress) => {
  const safeProgress = Math.max(0, Math.min(1, progress));
  const targetString = String(target ?? "");
  const totalDigits = (targetString.match(/\d/g) || []).length || 1;
  let seenDigits = 0;

  return targetString
    .split("")
    .map((char) => {
      if (!DIGIT_REGEX.test(char)) return char;
      seenDigits += 1;
      const revealPoint = seenDigits / totalDigits;
      return safeProgress >= revealPoint ? char : randomDigit();
    })
    .join("");
};

/**
 * Deterministic digit-roll animation for KPI values.
 * This avoids relying on global DOM scanning.
 */
export const AnimatedNumberText = ({
  value,
  seed = "",
  className = "",
  style = undefined,
  durationMs = 900,
  frameMs = 48,
  forceOnSeed = true,
}) => {
  const target = useMemo(() => String(value ?? "—"), [value]);
  const [display, setDisplay] = useState(target);
  const prevTargetRef = useRef(target);
  const prevSeedRef = useRef(String(seed || ""));

  useEffect(() => {
    const reducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
    if (reducedMotion || !hasDigits(target)) {
      setDisplay(target);
      prevTargetRef.current = target;
      prevSeedRef.current = String(seed || "");
      return undefined;
    }

    const seedChanged = prevSeedRef.current !== String(seed || "");
    const targetChanged = prevTargetRef.current !== target;
    if (!targetChanged && !(forceOnSeed && seedChanged)) {
      return undefined;
    }

    prevTargetRef.current = target;
    prevSeedRef.current = String(seed || "");

    let elapsed = 0;
    setDisplay(target);

    const timer = setInterval(
      () => {
        elapsed += frameMs;
        const progress = Math.min(1, elapsed / Math.max(120, durationMs));
        setDisplay(scrambleDigits(target, progress));
        if (progress >= 1) {
          clearInterval(timer);
          setDisplay(target);
        }
      },
      Math.max(16, frameMs)
    );

    return () => clearInterval(timer);
  }, [target, seed, durationMs, frameMs, forceOnSeed]);

  return (
    <span className={className} style={style} data-jackpot-target="true" data-jackpot-skip="true">
      {display}
    </span>
  );
};
