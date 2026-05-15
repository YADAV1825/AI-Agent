"""
Real Browser Tools — Connect to the user's actual Chrome/Brave/Edge
using Playwright's persistent context for full DOM access.

Instead of the fragile CDP remote-debugging approach, this uses
playwright.chromium.launch_persistent_context() which:
  - Directly launches Chrome with the user's profile data
  - Gives instant DOM access (no port probing needed)
  - Skips the profile picker automatically
  - Works with the user's cookies, logins, extensions, etc.
"""

import os
import time
import subprocess
import re
from typing import Optional

try:
    from playwright.sync_api import sync_playwright, BrowserContext, Page
except ImportError:
    sync_playwright = None

# --- Browser executable discovery (Windows) ---
_CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
]

_BRAVE_PATHS = [
    r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
]

_EDGE_PATHS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def _find_browser_exe(browser_name: str = "chrome") -> Optional[str]:
    name = browser_name.lower()
    if "brave" in name:
        paths = _BRAVE_PATHS
    elif "edge" in name:
        paths = _EDGE_PATHS
    else:
        paths = _CHROME_PATHS
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def _get_user_data_dir(browser_name: str) -> str:
    name = browser_name.lower()
    if "brave" in name:
        return os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\User Data")
    elif "edge" in name:
        return os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\User Data")
    else:
        return os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data")


class RealBrowserTools:
    """Connect to the user's real Chrome/Brave/Edge via Playwright persistent context."""

    def __init__(self, log_fn, stop_check=None):
        self.log = log_fn
        self.stop_check = stop_check
        self._playwright = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    # ================================================================ lifecycle

    def chrome_connect(self, browser_name: str = "chrome") -> str:
        """Open the user's real Chrome with their profile and get DOM access.

        Uses Playwright's launch_persistent_context which:
        - Launches Chrome with the user's actual profile (cookies, logins, etc.)
        - Skips the profile picker via --profile-directory=Default
        - Gives instant DOM access — no CDP port needed
        """
        if sync_playwright is None:
            return "chrome_connect failed: Playwright is not installed. Run: pip install playwright && playwright install chromium"

        # If already connected, reuse
        if self._context is not None:
            try:
                pages = self._context.pages
                if pages:
                    return f"Already connected to {browser_name}. {len(pages)} tab(s) open."
            except Exception:
                self._cleanup()

        exe = _find_browser_exe(browser_name)
        if not exe:
            return f"chrome_connect failed: Could not find {browser_name} on this system."

        user_data = _get_user_data_dir(browser_name)
        exe_name = os.path.basename(exe)

        # Kill existing browser instances (required — Chrome locks its profile)
        self.log(f"[Chrome] Closing existing {exe_name}...")
        try:
            subprocess.run(
                f'taskkill /F /IM "{exe_name}" /T',
                shell=True, capture_output=True, timeout=10,
            )
        except Exception:
            pass
        time.sleep(2)

        # Launch Chrome with the user's profile via Playwright
        self.log(f"[Chrome] Opening {browser_name} with your profile...")
        try:
            self._playwright = sync_playwright().start()
            self._context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=user_data,
                executable_path=exe,
                headless=False,
                no_viewport=True,           # don't constrain to a viewport
                args=[
                    "--profile-directory=Default",  # skip profile picker, use first profile
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-blink-features=AutomationControlled",  # hide automation flag
                    "--start-maximized",
                ],
                ignore_default_args=["--enable-automation"],  # remove "Chrome is controlled" bar
            )

            pages = self._context.pages
            if pages:
                self._page = pages[0]
            else:
                self._page = self._context.new_page()

            n_tabs = len(self._context.pages)
            self.log(f"[Chrome] ✅ Connected! {n_tabs} tab(s) open.")
            return f"Connected to {browser_name} with your profile. {n_tabs} tab(s) open. DOM access is ready."

        except Exception as e:
            self._cleanup()
            return f"chrome_connect failed: {e}"

    def _cleanup(self):
        """Clean up Playwright resources."""
        try:
            if self._context:
                self._context.close()
        except Exception:
            pass
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._context = None
        self._page = None
        self._playwright = None

    def shutdown(self):
        """Disconnect and close the browser."""
        self._cleanup()

    def _ensure_connected(self) -> Optional[str]:
        if self._context is None or self._page is None:
            return "Not connected to a real browser. Use chrome_connect first."
        return None

    def _refresh_page(self):
        """Re-acquire the active page if tabs changed."""
        if self._context:
            pages = self._context.pages
            if pages and self._page not in pages:
                self._page = pages[-1]

    # ================================================================ tabs

    def chrome_tabs(self) -> str:
        """List all open tabs."""
        err = self._ensure_connected()
        if err:
            return err
        try:
            self._refresh_page()
            pages = self._context.pages
            lines = []
            for i, p in enumerate(pages):
                marker = " ← active" if p == self._page else ""
                title = p.title()[:60] or "(loading...)"
                lines.append(f"  [{i}] {title} | {p.url[:80]}{marker}")
            return f"Open tabs ({len(pages)}):\n" + "\n".join(lines)
        except Exception as e:
            return f"chrome_tabs failed: {e}"

    def chrome_tab(self, index: int) -> str:
        """Switch to a tab by index."""
        err = self._ensure_connected()
        if err:
            return err
        try:
            pages = self._context.pages
            if index < 0 or index >= len(pages):
                return f"chrome_tab failed: index {index} out of range (0-{len(pages)-1})"
            self._page = pages[index]
            self._page.bring_to_front()
            return f"Switched to tab [{index}]: {self._page.title()[:60]} | {self._page.url[:80]}"
        except Exception as e:
            return f"chrome_tab failed: {e}"

    # ================================================================ navigation

    def chrome_navigate(self, url: str) -> str:
        """Navigate the active tab to a URL."""
        err = self._ensure_connected()
        if err:
            return err
        try:
            self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(1.5)
            return f"Navigated to: {self._page.url} | Title: {self._page.title()}"
        except Exception as e:
            return f"chrome_navigate failed: {e}"

    # ================================================================ DOM

    def chrome_dom(self, selector: str = "body", max_depth: int = 4) -> str:
        """Get a simplified DOM tree of the active page.

        Returns tag names, IDs, classes, aria-labels, roles, placeholders,
        and direct text content. Max 200 nodes.
        """
        err = self._ensure_connected()
        if err:
            return err
        try:
            js = """
            (args) => {
                const [rootSelector, maxDepth] = args;
                const root = document.querySelector(rootSelector) || document.body;
                let output = [];
                let count = 0;
                const MAX_NODES = 200;

                function walk(el, depth) {
                    if (count >= MAX_NODES || depth > maxDepth) return;
                    count++;

                    const tag = el.tagName.toLowerCase();
                    if (['script','style','noscript','svg','path','meta','link','head'].includes(tag)) return;

                    let attrs = [];
                    if (el.id) attrs.push('id="' + el.id + '"');
                    if (el.className && typeof el.className === 'string') {
                        const cls = el.className.trim().split(/\\s+/).slice(0, 3).join(' ');
                        if (cls) attrs.push('class="' + cls + '"');
                    }
                    if (el.getAttribute('role')) attrs.push('role="' + el.getAttribute('role') + '"');
                    if (el.getAttribute('aria-label')) attrs.push('aria-label="' + el.getAttribute('aria-label') + '"');
                    if (el.getAttribute('placeholder')) attrs.push('placeholder="' + el.getAttribute('placeholder') + '"');
                    if (el.getAttribute('type')) attrs.push('type="' + el.getAttribute('type') + '"');
                    if (el.getAttribute('name')) attrs.push('name="' + el.getAttribute('name') + '"');
                    if (el.getAttribute('href')) attrs.push('href="' + el.getAttribute('href').substring(0,60) + '"');
                    if (el.getAttribute('data-tooltip')) attrs.push('tooltip="' + el.getAttribute('data-tooltip') + '"');
                    if (el.getAttribute('contenteditable')) attrs.push('contenteditable="' + el.getAttribute('contenteditable') + '"');
                    if (el.getAttribute('tabindex')) attrs.push('tabindex="' + el.getAttribute('tabindex') + '"');

                    let text = '';
                    for (const child of el.childNodes) {
                        if (child.nodeType === 3) {
                            const t = child.textContent.trim();
                            if (t) text += t + ' ';
                        }
                    }
                    text = text.trim().substring(0, 80);

                    const indent = '  '.repeat(depth);
                    let line = indent + '<' + tag;
                    if (attrs.length) line += ' ' + attrs.join(' ');
                    line += '>';
                    if (text) line += ' "' + text + '"';
                    output.push(line);

                    for (const child of el.children) {
                        walk(child, depth + 1);
                    }
                }

                walk(root, 0);
                return output.join('\\n');
            }
            """
            result = self._page.evaluate(js, [selector, max_depth])
            if not result:
                return f"chrome_dom: No elements found for '{selector}'"

            if len(result) > 10000:
                result = result[:10000] + "\n\n[TRUNCATED — use a more specific selector]"

            return f"DOM of '{selector}' on {self._page.url}:\n{result}"

        except Exception as e:
            return f"chrome_dom failed: {e}"

    def chrome_read(self, selector: str = "body") -> str:
        """Read the text content of a DOM element."""
        err = self._ensure_connected()
        if err:
            return err
        try:
            text = self._page.locator(selector).inner_text(timeout=10000)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) > 8000:
                text = text[:8000] + "\n\n[TRUNCATED]"
            return text
        except Exception as e:
            return f"chrome_read failed: {e}"

    def chrome_click(self, selector: str) -> str:
        """Click a DOM element by CSS selector."""
        err = self._ensure_connected()
        if err:
            return err
        try:
            self._page.click(selector, timeout=10000)
            time.sleep(0.5)
            return f"Clicked: {selector}"
        except Exception as e:
            return f"chrome_click failed: {e}"

    def chrome_type(self, selector: str, text: str) -> str:
        """Type text into a DOM element.

        Uses fill() for <input>/<textarea>, falls back to
        click + keyboard.type() for contenteditable divs (e.g. Gmail body).
        """
        err = self._ensure_connected()
        if err:
            return err
        try:
            self._page.fill(selector, text, timeout=10000)
            return f"Typed '{text[:50]}' into {selector}"
        except Exception:
            try:
                self._page.click(selector, timeout=5000)
                time.sleep(0.2)
                self._page.keyboard.type(text, delay=20)
                return f"Typed '{text[:50]}' into {selector} (keyboard fallback)"
            except Exception as e2:
                return f"chrome_type failed: {e2}"

    def chrome_js(self, code: str) -> str:
        """Execute JavaScript in the active tab."""
        err = self._ensure_connected()
        if err:
            return err
        try:
            result = self._page.evaluate(code)
            text = str(result)
            if len(text) > 6000:
                text = text[:6000] + "\n[TRUNCATED]"
            return text
        except Exception as e:
            return f"chrome_js failed: {e}"

    def chrome_wait(self, selector: str, timeout: int = 15) -> str:
        """Wait for a DOM element to appear."""
        err = self._ensure_connected()
        if err:
            return err
        try:
            self._page.wait_for_selector(selector, timeout=timeout * 1000)
            return f"Element found: {selector}"
        except Exception as e:
            return f"chrome_wait: '{selector}' not found within {timeout}s: {e}"
