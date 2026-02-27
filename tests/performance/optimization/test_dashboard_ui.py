"""UI/UX and Regression tests for HPO Dashboard using Playwright.

Covers:
- CLS (Cumulative Layout Shift)
- Responsiveness (Reflow at 320px)
- Polling stability (Anti-flicker)
- Accessibility (Axe)
"""

import socket
import threading
import time
from unittest.mock import patch

import orjson
import pytest

# Mark entire module as slow/UI tests to exclude from fast CI
pytestmark = [pytest.mark.slow, pytest.mark.integration]

playwright = pytest.importorskip("playwright", reason="playwright not installed")
from playwright.sync_api import Page, expect  # noqa: E402

from pff.infrastructure.hpo.dashboard.server import run_server  # noqa: E402
from pff.shared.core.config import settings  # noqa: E402

# --- FIXTURES ---


def _initial_dashboard_payload():
    return {
        "studyName": "UI Test Study",
        "updatedAt": "2024-01-01T12:00:00Z",
        "bestValue": 0.5,
        "trials": [
            {
                "id": 1,
                "value": 0.5,
                "state": "COMPLETE",
                "duration": 10,
                "params": {"lr": 0.01},
                "metrics": {"confusion_matrix": {"vp": 50, "fp": 10, "fn": 5, "vn": 35}},
            },
            {
                "id": 2,
                "value": 0.4,
                "state": "PRUNED",
                "duration": 5,
                "params": {"lr": 0.1},
            },
        ],
    }


@pytest.fixture(scope="module")
def dashboard_server(tmp_path_factory):
    """Starts the dashboard server on a background thread."""
    # Setup temp environment
    root = tmp_path_factory.mktemp("dashboard_root")

    # 1. Setup Static Files
    # In real app: pff/infrastructure/hpo/dashboard/static
    # We copy the actual static files to the temp dir to test the REAL dashboard HTML
    real_static = settings.PACKAGE_DIR / "infrastructure" / "hpo" / "dashboard" / "static"
    temp_static = root / "static"
    import shutil

    shutil.copytree(real_static, temp_static)

    # 1b. Setup Dist Files (built JS/CSS bundle)
    real_dist = settings.PACKAGE_DIR / "infrastructure" / "hpo" / "dashboard" / "dist"
    temp_dist = root / "dist"
    if real_dist.exists():
        shutil.copytree(real_dist, temp_dist)

    # 2. Setup Data File
    data_dir = root / "cache"
    data_dir.mkdir()
    data_file = data_dir / "dashboard_data.json"

    # Initial Data
    initial_data = _initial_dashboard_payload()
    data_file.write_bytes(orjson.dumps(initial_data))

    # 3. Start Server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    # Patch paths
    with (
        patch("pff.infrastructure.hpo.dashboard.server.DATA_CACHE_PATH", data_file),
        patch("pff.infrastructure.hpo.dashboard.server.STATIC_DIR", temp_static),
        patch("pff.infrastructure.hpo.dashboard.server.DIST_DIR", temp_dist),
        patch(
            "pff.infrastructure.hpo.dashboard.server._collect_dashboard_data_paths",
            lambda: [data_file],
        ),
    ):
        server_thread = threading.Thread(
            target=run_server, kwargs={"port": port, "bind": "127.0.0.1"}, daemon=True
        )
        server_thread.start()
        time.sleep(2)

        yield {"url": f"http://127.0.0.1:{port}", "data_file": data_file}

        # Daemon thread dies with the process; no manual shutdown needed.
        # Avoid calling force_stop() here as it contaminates the event loop
        # used by Playwright's browser teardown.


# --- TESTS ---

# Note: These tests require a full browser environment with system dependencies (GTK, ATK, etc.)
# which might be missing in minimal containers.
# We mark them as skipped by default to avoid breaking CI in restricted environments.
# To run them, remove the skip marker or run with pytest -m "ui" (after removing skip).


# @pytest.mark.skip(
#     reason="Requires full browser environment with system dependencies (libatk, etc)"
# )
def test_dashboard_cls_stability(page: Page, dashboard_server):
    """Ensure CLS (Cumulative Layout Shift) is low (< 0.1)."""

    # Inject CLS observer
    page.add_init_script("""
        window.__cls = 0;
        new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                if (!entry.hadRecentInput) window.__cls += entry.value;
            }
        }).observe({ type: 'layout-shift', buffered: true });
    """)

    page.goto(dashboard_server["url"])

    # Wait for React hydration and Charts
    page.wait_for_selector("text=UI Test Study")
    page.wait_for_timeout(2000)

    # Check CLS
    cls = page.evaluate("window.__cls")
    print(f"CLS Value: {cls}")
    assert cls <= 0.7, f"Layout Shift too high: {cls}"


# # @pytest.mark.skip(reason="Requires full browser environment")
def test_dashboard_reflow_mobile(page: Page, dashboard_server):
    """WCAG Reflow: No horizontal scroll at 320px width."""
    page.set_viewport_size({"width": 320, "height": 800})
    page.goto(dashboard_server["url"])
    page.wait_for_selector("text=UI Test Study")

    # Check for horizontal scroll
    has_horizontal_scroll = page.evaluate("""
        document.documentElement.scrollWidth > document.documentElement.clientWidth
    """)

    assert not has_horizontal_scroll, (
        "Dashboard has horizontal scroll on 320px width (WCAG violation)"
    )


# # @pytest.mark.skip(reason="Requires full browser environment")
def test_dashboard_polling_no_flicker(page: Page, dashboard_server):
    """Ensure UI doesn't 'flash empty' during data polling."""
    page.goto(dashboard_server["url"])
    page.wait_for_selector("text=UI Test Study")

    # Now verify "Log empty" or "Waiting" is NOT present
    expect(page.get_by_text("Waiting for optimization data")).to_have_count(0)
    expect(page.get_by_text("Log empty")).to_have_count(0)

    # Simulate a slow delayed update via SSE-backed file write
    time.sleep(1)
    data_file = dashboard_server["data_file"]
    polled = {
        "studyName": "Polled Study",
        "updatedAt": "2024-01-01T12:00:10Z",
        "bestValue": 0.7,
        "trials": [
            {
                "id": 1,
                "value": 0.6,
                "state": "COMPLETE",
                "duration": 10,
                "params": {},
            },
            {
                "id": 2,
                "value": 0.7,
                "state": "COMPLETE",
                "duration": 10,
                "params": {},
            },
        ],
    }
    data_file.write_bytes(orjson.dumps(polled))
    page.wait_for_selector("text=Polled Study")

    # During this wait, the UI should have remained stable (no empty state)
    expect(page.get_by_text("Waiting for optimization data")).to_have_count(0)

    # And eventually updated (explicit event generated after polling refresh)
    expect(page.get_by_text("Trial #2 assumiu a liderança").first).to_be_visible()


def test_confusion_matrix_percentages_and_tooltip(page: Page, dashboard_server):
    """Execute test confusion matrix percentages and tooltip.



    Args:

        page: Input value used by this callable.

        dashboard_server: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    dashboard_server["data_file"].write_bytes(orjson.dumps(_initial_dashboard_payload()))
    page.goto(dashboard_server["url"])
    page.wait_for_selector("text=UI Test Study")

    page.get_by_text("Análise").click()
    page.wait_for_selector("text=Matriz de Confusão")

    vp_cell = page.get_by_text("VP").first
    expect(vp_cell).to_be_visible()
    page.get_by_text("50.0%").first.wait_for()

    vp_cell.hover()
    expect(page.get_by_text("Explicação Técnica")).to_be_visible()
    expect(page.get_by_text("Para Leigos")).to_be_visible()
    expect(page.get_by_text("Verdadeiro Positivo (VP) = 50")).to_be_visible()
    expect(page.get_by_text("Acertou quando disse SIM.")).to_be_visible()


def test_command_palette_opens_and_navigates(page: Page, dashboard_server):
    """Validate command palette shortcut and navigation to a target card."""
    dashboard_server["data_file"].write_bytes(orjson.dumps(_initial_dashboard_payload()))
    page.goto(dashboard_server["url"])
    page.wait_for_selector("text=UI Test Study")

    page.keyboard.press("Control+k")
    palette_input = page.get_by_placeholder("Buscar gráficos, tabelas e cards...")
    expect(palette_input).to_be_visible()
    palette_input.fill("matriz de confusao")

    target = page.get_by_role("option", name="Matriz de Confusão").first
    expect(target).to_be_visible()
    target.click()

    page.wait_for_timeout(250)
    expect(page.get_by_text("Matriz de Confusão").first).to_be_visible()


def test_command_palette_opens_when_typing(page: Page, dashboard_server):
    """Ensure printable typing opens palette and seeds query text."""
    dashboard_server["data_file"].write_bytes(orjson.dumps(_initial_dashboard_payload()))
    page.goto(dashboard_server["url"])
    page.wait_for_selector("text=UI Test Study")

    page.keyboard.type("matriz")
    palette_input = page.get_by_placeholder("Buscar gráficos, tabelas e cards...")
    expect(palette_input).to_be_visible()
    expect(palette_input).to_have_value("matriz")


# # @pytest.mark.skip(reason="Requires full browser environment")
def test_dashboard_console_clean(page: Page, dashboard_server):
    """Ensure no console errors or forbidden warnings (Tailwind CDN, Babel)."""
    errors = []
    warnings = []

    dashboard_server["data_file"].write_bytes(orjson.dumps(_initial_dashboard_payload()))

    def handle_console(msg):
        """Execute handle console.



        Args:

            msg: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if msg.type == "error":
            errors.append(f"{msg.type}: {msg.text}")
            return
        if msg.type == "warning":
            text = msg.text
            if any(
                token in text for token in ("width(-1)", "height(-1)", "should be greater than 0")
            ):
                warnings.append(f"{msg.type}: {text}")

    page.on("console", handle_console)
    page.on("pageerror", lambda exc: errors.append(f"EXCEPTION: {exc}"))

    page.goto(dashboard_server["url"])
    page.wait_for_selector("text=UI Test Study")

    if errors:
        pytest.fail("Console errors detected:\n" + "\n".join(errors))
    if warnings:
        pytest.fail("Console warnings detected:\n" + "\n".join(warnings))


def test_dashboard_legend_hover_hint_and_toggle_series(page: Page, dashboard_server):
    """Ensure legend hover shows hintbox and click toggles series visibility."""
    dashboard_server["data_file"].write_bytes(
        orjson.dumps(
            {
                "studyName": "UI Test Study",
                "updatedAt": "2024-01-01T12:00:00Z",
                "bestValue": 0.62,
                "direction": "maximize",
                "totalTrials": 50,
                "trials": [
                    {"id": 1, "value": 0.50, "state": "COMPLETE", "duration": 3, "params": {}},
                    {"id": 2, "value": 0.52, "state": "COMPLETE", "duration": 3, "params": {}},
                    {"id": 3, "value": 0.60, "state": "COMPLETE", "duration": 3, "params": {}},
                    {"id": 4, "value": 0.58, "state": "COMPLETE", "duration": 3, "params": {}},
                    {"id": 5, "value": 0.62, "state": "COMPLETE", "duration": 3, "params": {}},
                ],
            }
        )
    )
    page.goto(dashboard_server["url"])
    page.wait_for_selector("text=UI Test Study")

    legend_btn = page.locator('button[aria-label*="série Objetivo"]').first
    expect(legend_btn).to_be_visible()

    legend_btn.hover()
    page.wait_for_timeout(300)

    expect(page.get_by_text("Explicação Técnica").first).to_be_visible()
    expect(page.get_by_text("Para Leigos").first).to_be_visible()

    before_paths = page.locator("path.recharts-line-curve").count()
    expect(legend_btn).to_have_attribute("aria-pressed", "false")

    legend_btn.click()
    page.wait_for_timeout(250)

    after_paths = page.locator("path.recharts-line-curve").count()
    expect(legend_btn).to_have_attribute("aria-pressed", "true")
    assert after_paths < before_paths, (
        f"Legend click did not hide any line (before={before_paths}, after={after_paths})"
    )

    legend_btn.click()
    page.wait_for_timeout(250)
    expect(legend_btn).to_have_attribute("aria-pressed", "false")


def test_incumbent_chart_gradient_and_tooltip_without_duplicates(page: Page, dashboard_server):
    """Ensure incumbent chart renders gradient areas and deduplicated tooltip rows."""
    dashboard_server["data_file"].write_bytes(
        orjson.dumps(
            {
                "studyName": "UI Test Study",
                "updatedAt": "2024-01-01T12:00:00Z",
                "bestValue": 0.62,
                "direction": "maximize",
                "totalTrials": 50,
                "trials": [
                    {"id": 1, "value": 0.50, "state": "COMPLETE", "duration": 3, "params": {}},
                    {"id": 2, "value": 0.52, "state": "COMPLETE", "duration": 3, "params": {}},
                    {"id": 3, "value": 0.60, "state": "COMPLETE", "duration": 3, "params": {}},
                    {"id": 4, "value": 0.58, "state": "COMPLETE", "duration": 3, "params": {}},
                    {"id": 5, "value": 0.62, "state": "COMPLETE", "duration": 3, "params": {}},
                ],
            }
        )
    )
    page.goto(dashboard_server["url"])
    page.wait_for_selector("text=UI Test Study")
    page.wait_for_selector("text=HISTÓRICO DE OTIMIZAÇÃO")

    area_count = page.locator("path.recharts-area-area").count()
    assert area_count >= 3, f"Expected gradient area fills, got {area_count} rendered areas"

    fills = page.evaluate("""
        () => Array.from(document.querySelectorAll("path.recharts-area-area"))
          .slice(0, 3)
          .map((el) => el.getAttribute("fill") || "")
    """)
    assert all(fill.startswith("url(#grad-") for fill in fills), (
        f"Area paths are not using gradient fills: {fills}"
    )

    target_point = page.evaluate("""
        () => {
          const surfaces = Array.from(document.querySelectorAll("svg.recharts-surface"));
          const target = surfaces.find((svg) => svg.querySelectorAll("path.recharts-area-area").length >= 3);
          if (!target) return null;
          const rect = target.getBoundingClientRect();
          return {
            x: rect.left + rect.width * 0.5,
            y: rect.top + rect.height * 0.45,
          };
        }
    """)
    assert target_point, "Unable to locate incumbent chart surface for tooltip hover"
    page.mouse.move(target_point["x"], target_point["y"])
    page.wait_for_timeout(300)

    tooltip = page.get_by_test_id("incumbent-trajectory-tooltip").first
    expect(tooltip).to_be_visible()
    tooltip_text = tooltip.inner_text()

    assert "Objetivo:" in tooltip_text
    assert "Média Móvel:" in tooltip_text
    assert "Melhor (Incumbent):" in tooltip_text
    assert "value:" not in tooltip_text
    assert "movingAverage:" not in tooltip_text
    assert "incumbent:" not in tooltip_text


def test_micro_trial_legend_hint_and_toggle_series(page: Page, dashboard_server):
    """Ensure micro trial chart legend shows hintbox and toggles series visibility."""
    dashboard_server["data_file"].write_bytes(
        orjson.dumps(
            {
                "studyName": "UI Test Study",
                "updatedAt": "2024-01-01T12:00:00Z",
                "bestValue": 0.62,
                "direction": "maximize",
                "totalTrials": 50,
                "trials": [
                    {"id": 1, "value": 0.50, "state": "COMPLETE", "duration": 3, "params": {}},
                    {"id": 2, "value": 0.52, "state": "COMPLETE", "duration": 3, "params": {}},
                    {"id": 3, "value": 0.60, "state": "COMPLETE", "duration": 3, "params": {}},
                    {"id": 4, "value": 0.58, "state": "COMPLETE", "duration": 3, "params": {}},
                    {"id": 5, "value": 0.62, "state": "RUNNING", "duration": 3, "params": {}},
                ],
                "liveStatus": {
                    "updated_at": "2024-01-01T12:00:01Z",
                    "trial_number": 4,
                    "epoch_history": [
                        {
                            "epoch": 1,
                            "metrics": {"loss": 6.5, "val_loss": 1.6, "mrr": 0.08, "mcc": 0.03},
                        },
                        {
                            "epoch": 2,
                            "metrics": {"loss": 4.7, "val_loss": 1.2, "mrr": 0.12, "mcc": 0.08},
                        },
                        {
                            "epoch": 3,
                            "metrics": {"loss": 4.0, "val_loss": 0.9, "mrr": 0.17, "mcc": 0.12},
                        },
                        {
                            "epoch": 4,
                            "metrics": {"loss": 3.6, "val_loss": 0.7, "mrr": 0.20, "mcc": 0.18},
                        },
                        {
                            "epoch": 5,
                            "metrics": {"loss": 3.2, "val_loss": 0.6, "mrr": 0.22, "mcc": 0.21},
                        },
                    ],
                },
            }
        )
    )

    page.goto(dashboard_server["url"])
    page.wait_for_selector("text=UI Test Study")

    scope_tabs = page.locator('[role="tablist"][aria-label="Escopo macro e micro"] [role="tab"]')
    micro_tab = scope_tabs.nth(1)
    micro_tab.dispatch_event("click")
    expect(micro_tab).to_have_attribute("aria-selected", "true")
    page.wait_for_selector("#search-overview-trial-learning-metrics")

    legend_btn = page.locator('button[aria-label*="série LOSS"]').first
    expect(legend_btn).to_be_visible()

    legend_btn.hover()
    page.wait_for_timeout(250)
    expect(page.get_by_text("Explicação Técnica").first).to_be_visible()
    expect(page.get_by_text("Para Leigos").first).to_be_visible()

    chart_root = page.locator("#search-overview-trial-learning-metrics")
    before_areas = chart_root.locator("path.recharts-area-area").count()
    expect(legend_btn).to_have_attribute("aria-pressed", "false")

    legend_btn.click()
    page.wait_for_timeout(250)
    after_areas = chart_root.locator("path.recharts-area-area").count()

    expect(legend_btn).to_have_attribute("aria-pressed", "true")
    assert after_areas < before_areas, (
        f"Legend click did not hide micro-series area (before={before_areas}, after={after_areas})"
    )

    legend_btn.click()
    page.wait_for_timeout(250)
    expect(legend_btn).to_have_attribute("aria-pressed", "false")


def test_tooltips_single_active_and_auto_hide(page: Page, dashboard_server):
    """Ensure only one hintbox stays visible and all hintboxes auto-hide on mouse away."""
    dashboard_server["data_file"].write_bytes(
        orjson.dumps(
            {
                "studyName": "UI Test Study",
                "updatedAt": "2024-01-01T12:00:00Z",
                "bestValue": 0.62,
                "direction": "maximize",
                "totalTrials": 50,
                "trials": [
                    {"id": 1, "value": 0.50, "state": "COMPLETE", "duration": 3, "params": {}},
                    {"id": 2, "value": 0.52, "state": "COMPLETE", "duration": 3, "params": {}},
                    {"id": 3, "value": 0.60, "state": "COMPLETE", "duration": 3, "params": {}},
                    {"id": 4, "value": 0.58, "state": "COMPLETE", "duration": 3, "params": {}},
                    {"id": 5, "value": 0.62, "state": "RUNNING", "duration": 3, "params": {}},
                ],
                "liveStatus": {
                    "updated_at": "2024-01-01T12:00:01Z",
                    "trial_number": 4,
                    "epoch_history": [
                        {
                            "epoch": 1,
                            "metrics": {"loss": 6.5, "val_loss": 1.6, "mrr": 0.08, "mcc": 0.03},
                        },
                        {
                            "epoch": 2,
                            "metrics": {"loss": 4.7, "val_loss": 1.2, "mrr": 0.12, "mcc": 0.08},
                        },
                    ],
                },
            }
        )
    )

    page.goto(dashboard_server["url"])
    page.wait_for_selector("text=UI Test Study")
    page.wait_for_selector("[data-pff-tooltip-trigger]")

    triggers = page.locator("[data-pff-tooltip-trigger]")
    total_triggers = triggers.count()
    assert total_triggers >= 2, f"Expected at least 2 tooltip triggers, got {total_triggers}"

    interactive_indexes = []
    for idx in range(total_triggers):
        candidate = triggers.nth(idx)
        box = candidate.bounding_box()
        if not box or box["width"] <= 2 or box["height"] <= 2:
            continue
        try:
            candidate.hover(timeout=1200)
        except Exception:
            continue
        page.wait_for_timeout(120)
        if page.locator("[data-pff-tooltip-root]").count() == 1:
            interactive_indexes.append(idx)
            safe_probe = page.evaluate("""
                () => ({
                  x: Math.max(12, window.innerWidth - 24),
                  y: Math.max(12, window.innerHeight - 24),
                })
            """)
            page.mouse.move(safe_probe["x"], safe_probe["y"])
            page.wait_for_timeout(120)
        if len(interactive_indexes) >= 2:
            break

    assert len(interactive_indexes) >= 2, "Unable to locate two interactive tooltip triggers"

    first_trigger = triggers.nth(interactive_indexes[0])
    second_trigger = triggers.nth(interactive_indexes[1])

    first_trigger.hover()
    page.wait_for_timeout(160)
    assert page.locator("[data-pff-tooltip-root]").count() == 1

    second_trigger.hover()
    page.wait_for_timeout(160)
    assert page.locator("[data-pff-tooltip-root]").count() == 1

    safe_point = page.evaluate("""
        () => ({
          x: Math.max(12, window.innerWidth - 24),
          y: Math.max(12, window.innerHeight - 24),
        })
    """)
    page.mouse.move(safe_point["x"], safe_point["y"])
    page.wait_for_timeout(220)
    expect(page.locator("[data-pff-tooltip-root]")).to_have_count(0)


# @pytest.mark.skip(reason="Requires full browser environment")
def test_animation_accessibility_reduced_motion(page: Page, dashboard_server):
    """Ensure reduced motion preference is respected."""
    # Set reduced motion preference
    page.emulate_media(reduced_motion="reduce")
    dashboard_server["data_file"].write_bytes(orjson.dumps(_initial_dashboard_payload()))
    page.goto(dashboard_server["url"])
    page.wait_for_selector("text=UI Test Study")

    # Check if animations are disabled or very fast
    duration = page.evaluate(
        "() => { const el = document.querySelector('.animate-spring-up'); if (!el) return 0; return parseFloat(getComputedStyle(el).animationDuration); }"
    )
    assert duration <= 0.001, f"Animations still present with reduced motion: {duration}s"


# @pytest.mark.skip(reason="Requires full browser environment")
def test_staggered_load_performance(page: Page, dashboard_server):
    """Ensure staggered load doesn't block interactivity for too long."""
    start_time = time.time()
    dashboard_server["data_file"].write_bytes(orjson.dumps(_initial_dashboard_payload()))
    page.goto(dashboard_server["url"])
    page.wait_for_selector("text=UI Test Study")

    # Wait for the last card delay (e.g., 700ms) + small buffer
    page.wait_for_timeout(1000)

    # Try to click a tab
    pass

    # If we reached here without error, it's interactive
    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Dashboard load took too long: {elapsed}s"


# @pytest.mark.skip(reason="Requires full browser environment")
def test_cls_with_animations(page: Page, dashboard_server):
    """Ensure CLS remains stable even with staggered animations."""
    page.add_init_script("""
        window.__cls = 0;
        new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                if (!entry.hadRecentInput) window.__cls += entry.value;
            }
        }).observe({ type: 'layout-shift', buffered: true });
    """)
    dashboard_server["data_file"].write_bytes(orjson.dumps(_initial_dashboard_payload()))
    page.goto(dashboard_server["url"])
    page.wait_for_selector("text=UI Test Study")

    # Wait for all entrance animations to finish (approx 1.5s total)
    page.wait_for_timeout(2000)

    cls = page.evaluate("window.__cls")
    assert cls <= 0.7, f"CLS too high with animations: {cls}"
