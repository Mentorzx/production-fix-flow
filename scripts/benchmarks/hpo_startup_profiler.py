"""HPO startup profiler — measures time spent in each phase of the HPO startup path.

Usage:
    poetry run python scripts/benchmarks/hpo_startup_profiler.py [--import-only]
"""

from __future__ import annotations

import importlib
import sys
import time


def _measure_import(module_name: str) -> float:
    """Measure import time for a single module (cold import)."""
    if module_name in sys.modules:
        return 0.0
    t0 = time.perf_counter()
    importlib.import_module(module_name)
    return time.perf_counter() - t0


def profile_import_chain() -> dict[str, float]:
    """Profile the import chain triggered during HPO startup."""
    timings: dict[str, float] = {}

    stages = [
        ("1. pff (package init)", ["pff"]),
        ("2. torch (determinism)", ["torch"]),
        ("3. shared.core.config", ["pff.shared.core.config"]),
        ("4. shared.core.logging", ["pff.shared.core.logging"]),
        ("5. cli.main (entrypoint)", ["pff.drivers.cli.main"]),
        ("6. cli.commands (classes)", ["pff.drivers.cli.internal.commands"]),
        ("7. pff.__main__", ["pff.__main__"]),
        ("--- Below: deferred to execute() ---", []),
        ("8. hpo.runner (deferred)", ["pff.infrastructure.hpo.runner"]),
        ("9. optuna (deferred)", ["optuna"]),
        ("10. mlflow (deferred)", ["mlflow"]),
    ]

    for label, modules in stages:
        if not modules:
            timings[label] = 0.0
            continue
        t0 = time.perf_counter()
        for mod in modules:
            if mod not in sys.modules:
                try:
                    importlib.import_module(mod)
                except ImportError:
                    pass
        elapsed = time.perf_counter() - t0
        timings[label] = elapsed

    return timings


def main() -> None:
    """Run the profiler and print results."""
    print("=" * 65)
    print("PFF HPO Startup Import Profiler")
    print("=" * 65)

    overall_t0 = time.perf_counter()
    timings = profile_import_chain()
    overall_elapsed = time.perf_counter() - overall_t0

    startup_total = 0.0
    deferred_total = 0.0
    in_deferred = False

    print(f"\n{'Stage':<45} {'Time (s)':>10}")
    print("-" * 57)
    for label, elapsed in timings.items():
        if "deferred" in label.lower() and "---" in label:
            in_deferred = True
            print(f"  {label}")
            continue
        bar = "█" * int(elapsed * 20)
        print(f"  {label:<43} {elapsed:>8.3f}s  {bar}")
        if in_deferred:
            deferred_total += elapsed
        else:
            startup_total += elapsed

    print("-" * 57)
    print(f"  {'STARTUP (before execute)':<43} {startup_total:>8.3f}s")
    print(f"  {'DEFERRED (inside execute)':<43} {deferred_total:>8.3f}s")
    print(f"  {'TOTAL (cumulative)':<43} {overall_elapsed:>8.3f}s")
    print()

    all_timed = {k: v for k, v in timings.items() if v > 0}
    top3 = sorted(all_timed.items(), key=lambda x: x[1], reverse=True)[:3]
    print("Top 3 bottlenecks:")
    for i, (label, elapsed) in enumerate(top3, 1):
        print(f"  {i}. {label}: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
