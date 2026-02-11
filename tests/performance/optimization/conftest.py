import pytest

# ─── Playwright Fixture Overrides ─────────────────────────────────────
# Override pytest-playwright's session-scoped fixtures to module scope.
# Reason: sync_playwright() starts a background event loop thread that
# contaminates the process-global asyncio state. With session scope, this
# thread stays alive for the entire test session, causing all subsequent
# pytest-asyncio tests to fail with:
#   "Runner.run() cannot be called from a running event loop"
# By scoping to module, pw.stop() runs when test_dashboard_ui.py finishes,
# cleaning up the event loop BEFORE other test modules execute.


@pytest.fixture(scope="module")
def playwright():
    """Module-scoped Playwright instance (overrides session-scoped default)."""
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    yield pw
    pw.stop()


@pytest.fixture(scope="module")
def browser_type(playwright, browser_name):
    """Module-scoped browser type (overrides session-scoped default)."""
    return getattr(playwright, browser_name)


@pytest.fixture(scope="module")
def browser_type_launch_args():
    """Module-scoped launch args (overrides session-scoped default)."""
    return {
        "headless": True,
        "args": [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
        ],
    }


@pytest.fixture(scope="module")
def launch_browser(browser_type_launch_args, browser_type):
    """Module-scoped browser launcher (overrides session-scoped default)."""

    def launch(**kwargs):
        launch_options = {**browser_type_launch_args, **kwargs}
        return browser_type.launch(**launch_options)

    return launch


@pytest.fixture(scope="module")
def browser(launch_browser):
    """Module-scoped browser instance (overrides session-scoped default)."""
    browser = launch_browser()
    yield browser
    browser.close()


@pytest.fixture(scope="module")
def browser_context_args():
    """Module-scoped browser context args (overrides session-scoped default)."""
    return {
        "ignore_https_errors": True,
        "viewport": {"width": 1280, "height": 720},
    }
