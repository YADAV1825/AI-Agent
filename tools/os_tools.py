import os
import sys
import time
import subprocess
import webbrowser
import requests
import re
import pyautogui
from PIL import ImageGrab
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

class OSTools:
    def __init__(self, log_fn, stop_check=None):
        self.log = log_fn
        self.stop_check = stop_check

    # Modifier keys that need keyDown/keyUp for proper combo behavior
    _MODIFIERS = {"alt", "ctrl", "shift", "win", "winleft", "winright", "command"}

    # Map common aliases to pyautogui key names
    _KEY_ALIASES = {
        "windows": "win", "cmd": "win", "control": "ctrl",
        "return": "enter", "esc": "escape", "del": "delete",
        "pageup": "pageup", "pagedown": "pagedown",
        "arrowup": "up", "arrowdown": "down",
        "arrowleft": "left", "arrowright": "right",
    }

    def press_key(self, key: str) -> str:
        """Press a key or key combination.

        Supports:
          - Single key:   "enter", "tab", "space", "f5"
          - Simple combo:  "ctrl+l", "ctrl+shift+esc"
          - Modifier hold: "alt+tab" (holds alt, presses tab, releases alt)
          - Repeated:      "alt+tab+tab" (holds alt, presses tab twice)
          - Win combos:    "win+tab", "win+d"
        """
        try:
            raw = key.strip()
            parts = [self._normalize_key(k) for k in raw.split("+")]

            if len(parts) == 1:
                k = parts[0]
                if k.endswith("_down"):
                    base_key = k[:-5]
                    pyautogui.keyDown(base_key)
                    return f"Held down key: {base_key}"
                elif k.endswith("_up"):
                    base_key = k[:-3]
                    pyautogui.keyUp(base_key)
                    return f"Released key: {base_key}"
                else:
                    # Single key press
                    pyautogui.press(k)
                    if k in ("enter", "win", "winleft"):
                        time.sleep(1.0)
                    return f"Pressed key: {raw}"

            # Separate modifiers from regular keys
            modifiers = []
            regular_keys = []
            for p in parts:
                if p in self._MODIFIERS:
                    if p not in modifiers:  # avoid duplicate modifiers
                        modifiers.append(p)
                else:
                    regular_keys.append(p)

            if not regular_keys:
                # All parts are modifiers (weird but handle it)
                pyautogui.hotkey(*parts)
                return f"Pressed key combo: {raw}"

            # Hold modifiers → press each regular key in sequence → release modifiers
            # This handles: alt+tab (hold alt, press tab, release alt)
            #               alt+tab+tab (hold alt, press tab twice, release alt)
            #               ctrl+shift+esc (hold ctrl+shift, press esc, release)
            for mod in modifiers:
                pyautogui.keyDown(mod)
                time.sleep(0.05)

            for rk in regular_keys:
                pyautogui.press(rk)
                time.sleep(0.15)  # brief pause between repeated keys

            # Small pause before releasing modifiers (Windows needs this for
            # win+tab and alt+tab to register properly)
            time.sleep(0.3)

            for mod in reversed(modifiers):
                pyautogui.keyUp(mod)
                time.sleep(0.05)

            # Extra settle time for system-level shortcuts
            if any(m in ("win", "winleft", "winright") for m in modifiers):
                time.sleep(1.0)
            elif any(m == "alt" for m in modifiers):
                time.sleep(0.5)

            return f"Pressed key combo: {raw}"

        except Exception as e:
            return f"press_key failed: {e}"

    def _normalize_key(self, key: str) -> str:
        """Normalize a key name to what pyautogui expects."""
        k = key.strip().lower()
        return self._KEY_ALIASES.get(k, k)

    def type_text(self, text: str) -> str:
        try:
            pyautogui.write(text, interval=0.01)
            return f"Typed text: {text}"
        except Exception as e:
            return f"type_text failed: {e}"

    def take_screenshot(self) -> str:
        try:
            time.sleep(1.5)
            path = os.path.abspath("temp_screenshot.jpg")
            img = ImageGrab.grab()
            # Compress or resize if needed, but saving as JPEG works well
            img.convert("RGB").save(path, quality=85)
            return f"[SCREENSHOT_TAKEN: {path}]"
        except Exception as e:
            return f"take_screenshot failed: {e}"

    def open_app(self, app_name: str) -> str:
        app = (app_name or "").strip().lower()
        if not app: return "open_app: missing app name"
        try:
            if os.name == "nt":
                if app in {"chrome", "google chrome"}:
                    try:
                        subprocess.Popen("start chrome --guest", shell=True)
                        return f"Opened app: {app_name} (guest mode)"
                    except Exception:
                        subprocess.Popen("start chrome", shell=True)
                        return f"Opened app: {app_name}"
                elif app in {"vscode", "vs code", "visual studio code"}:
                    subprocess.Popen("code", shell=True)
                elif app in {"notepad"}:
                    subprocess.Popen("notepad", shell=True)
                else:
                    subprocess.Popen(app_name, shell=True)
            elif sys_platform := sys.platform:
                if "darwin" in sys_platform:
                    subprocess.Popen(["open", "-a", app_name])
                else:
                    subprocess.Popen([app_name])
            return f"Opened app: {app_name}"
        except Exception as e:
            return f"open_app failed: {e}"

    def open_url(self, url: str) -> str:
        try:
            webbrowser.open(url)
            return f"Opened URL: {url}"
        except Exception as e:
            return f"open_url failed: {e}"

    def search_web(self, query: str) -> str:
        try:
            url = "https://www.google.com/search?q=" + requests.utils.quote(query)
            webbrowser.open(url)
            return f"Searched web for: {query}"
        except Exception as e:
            return f"search_web failed: {e}"

    def read_url(self, url: str, max_chars: int = 12000) -> str:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            r = requests.get(url, headers=headers, timeout=25)
            r.raise_for_status()
            text = r.text
            if BeautifulSoup is not None:
                soup = BeautifulSoup(text, "html.parser")
                for tag in soup(["script", "style", "noscript"]): tag.decompose()
                text = "\n".join([ln.strip() for ln in soup.get_text(separator="\n").splitlines() if ln.strip()])
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) > max_chars: text = text[:max_chars] + "\n\n[TRUNCATED]"
            return text
        except Exception as e:
            return f"read_url failed: {e}"

    def shell(self, command: str, confirm_fn=None) -> str:
        cmd_preview = command[:150] + ("..." if len(command) > 150 else "")
        message = f"Execute shell command?\n\nCommand:\n{cmd_preview}"
        if confirm_fn and not confirm_fn("Confirm Shell Execution", message): 
            return "Shell command cancelled by user"
        try:
            proc = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            deadline = time.time() + 90
            while proc.poll() is None:
                if self.stop_check and self.stop_check():
                    proc.kill()
                    raise InterruptedError("Stop requested during subprocess execution")
                if time.time() > deadline:
                    proc.kill()
                    raise subprocess.TimeoutExpired(command, 90)
                time.sleep(0.25)
            out = (proc.stdout.read() if proc.stdout else "").strip()
            err = (proc.stderr.read() if proc.stderr else "").strip()
            rc = proc.returncode
            if rc != 0: return f"shell failed ({rc}):\n{err or out}"
            return out or "(no output)"
        except InterruptedError as e: return str(e)
        except Exception as e: return f"shell failed: {e}"
