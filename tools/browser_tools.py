import re
from typing import Optional
try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

class BrowserTools:
    def __init__(self, log_fn):
        self.log = log_fn
        self._playwright = None
        self._browser = None
        self._page = None

    def _ensure_browser(self) -> bool:
        if sync_playwright is None:
            return False
        if self._playwright is None:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=False)
            self._page = self._browser.new_page()
        return True

    def shutdown(self, force=False):
        """Shutdown browser. If force=False (default), keep it open for user.
        Only close if force=True or app is closing."""
        if not force:
            # Keep browser open for user to continue working
            return
        try:
            if self._browser:
                self._browser.close()
        except Exception: pass
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception: pass
        self._browser = None
        self._page = None
        self._playwright = None

    def browser_open(self, url: str) -> str:
        if not self._ensure_browser(): return "Playwright is not installed."
        try:
            self._page.goto(url, wait_until="domcontentloaded")
            return f"Browser opened: {url}"
        except Exception as e: return f"browser_open failed: {e}"

    def browser_click(self, selector: str) -> str:
        if not self._ensure_browser(): return "Playwright is not installed."
        try:
            self._page.click(selector, timeout=15000)
            return f"Clicked: {selector}"
        except Exception as e: return f"browser_click failed: {e}"

    def browser_type(self, selector: str, text: str) -> str:
        if not self._ensure_browser(): return "Playwright is not installed."
        try:
            self._page.fill(selector, text, timeout=15000)
            return f"Typed into {selector}"
        except Exception as e: return f"browser_type failed: {e}"

    def browser_read(self, selector: str = "body") -> str:
        if not self._ensure_browser(): return "Playwright is not installed."
        try:
            content = self._page.locator(selector).inner_text(timeout=15000)
            content = re.sub(r"\n{3,}", "\n\n", content).strip()
            if len(content) > 12000: content = content[:12000] + "\n\n[TRUNCATED]"
            return content
        except Exception as e: return f"browser_read failed: {e}"
