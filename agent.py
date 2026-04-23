"""
Desktop AI Agent — single-file end-to-end version for Lightning AI + gpt-oss-120b

What this does:
- GUI for Lightning API key / teamspace / model / task input
- Planner → JSON action loop
- Tool execution:
  - open_app
  - open_url
  - search_web
  - read_url
  - browser_open / browser_click / browser_type / browser_read (optional Playwright)
  - write_file / append_file / list_dir
  - run_python
  - shell
- Stop button
- Log window
- Max-step safety limit
- Works as a local desktop agent on Windows (best), with graceful fallbacks on macOS/Linux

Install:
    pip install requests
Optional:
    pip install beautifulsoup4 playwright
    playwright install

Run:
    python desktop_agent_lightning.py
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import tkinter as tk
from dotenv import load_dotenv
from tkinter import ttk, messagebox, filedialog

load_dotenv()

API_KEY = os.getenv("LIGHTNING_API_KEY")
TEAMSPACE = os.getenv("LIGHTNING_TEAMSPACE")
MODEL = os.getenv("MODEL_NAME", "lightning-ai/gpt-oss-120b")

# -----------------------------
# Optional deps
# -----------------------------
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None

try:
    from playwright.sync_api import sync_playwright  # type: ignore
except Exception:
    sync_playwright = None


# =============================================================================
# Lightning API Client
# =============================================================================

LIGHTNING_CHAT_URL = "https://lightning.ai/api/v1/chat/completions"
DEFAULT_MODEL = "lightning-ai/gpt-oss-120b"


def _extract_first_json_block(text: str) -> Any:
    """
    Best-effort JSON extraction.
    Handles:
    - plain JSON
    - code fences
    - assistant text with JSON embedded
    """
    if not text:
        raise ValueError("Empty model response")

    cleaned = text.strip()

    # Remove code fences if present
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.IGNORECASE)

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Extract first object/array-looking block
    start_candidates = [m.start() for m in re.finditer(r"[\{\[]", cleaned)]
    end_candidates = [m.start() for m in re.finditer(r"[\}\]]", cleaned)]
    for s in start_candidates:
        for e in reversed(end_candidates):
            if e > s:
                chunk = cleaned[s : e + 1]
                try:
                    return json.loads(chunk)
                except Exception:
                    continue

    raise ValueError(f"Could not parse JSON from model response:\n{text}")


@dataclass
class LightningClient:
    api_key: str
    teamspace: str
    model: str = DEFAULT_MODEL
    base_url: str = LIGHTNING_CHAT_URL

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
        timeout: int = 120,
    ) -> str:
        """
        Calls Lightning AI chat completions and returns text.
        Supports both `choices` and `candidates` shaped responses.
        """
        if not self.api_key.strip():
            raise ValueError("Lightning API key is empty")
        if not self.teamspace.strip():
            raise ValueError("Lightning teamspace/project path is empty")

        headers = {
            "Authorization": f"Bearer {self.api_key}/{self.teamspace}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = requests.post(self.base_url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            return msg.get("content", "") or ""

        if "candidates" in data and data["candidates"]:
            parts = data["candidates"][0].get("content", {}).get("parts", [])
            if parts and isinstance(parts, list):
                return parts[0].get("text", "") or ""

        return json.dumps(data, ensure_ascii=False, indent=2)


# =============================================================================
# Agent State
# =============================================================================

SYSTEM_PROMPT = """
You are a desktop AI agent controlling the user's computer.

You must output ONLY valid JSON.

Available actions:
- open_app
- open_url
- search_web
- read_url
- browser_open
- browser_click
- browser_type
- browser_read
- write_file
- append_file
- list_dir
- run_python
- shell
- done

Output schema:
{
  "action": "open_app",
  "input": "chrome",
  "path": "",
  "selector": "",
  "text": "",
  "code": "",
  "reason": "short reason"
}

Rules:
- Return exactly one action per turn.
- Use "done" when the task is finished.
- Prefer safe, small steps.
- If you need to inspect a web page, use read_url or browser_read.
- If you need DOM control and browser tools are available, use browser_open/browser_click/browser_type/browser_read.
- If the task is ambiguous, make the smallest useful next step.
- Never write extra commentary outside JSON.
"""


@dataclass
class AgentStep:
    turn: int
    action: str
    raw: str
    parsed: Dict[str, Any]
    result: str


@dataclass
class AgentState:
    task: str = ""
    steps: List[AgentStep] = field(default_factory=list)
    running: bool = False
    stop_requested: bool = False


# =============================================================================
# Tooling
# =============================================================================

class ToolKit:
    def __init__(self, log_fn, gui_root=None):
        self.log = log_fn
        self.gui_root = gui_root
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

    def shutdown(self):
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._browser = None
        self._page = None
        self._playwright = None

    def _find_edge_on_windows(self) -> Optional[str]:
        """Search for Edge in Windows Start menu or Program Files."""
        try:
            # Try common Edge installation paths
            edge_paths = [
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            ]
            for path in edge_paths:
                if Path(path).exists():
                    return path
            
            # Try Windows Start search
            result = subprocess.run(
                "where msedge",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')[0]
        except Exception:
            pass
        return None

    def open_app(self, app_name: str) -> str:
        app = (app_name or "").strip().lower()
        if not app:
            return "open_app: missing app name"

        try:
            if os.name == "nt":
                if app in {"chrome", "google chrome"}:
                    # Try Chrome with guest mode
                    try:
                        subprocess.Popen("start chrome --guest", shell=True)
                        self.log("Opened Chrome in guest mode")
                        return f"Opened app: {app_name} (guest mode)"
                    except Exception:
                        # Try without guest mode
                        try:
                            subprocess.Popen("start chrome", shell=True)
                            self.log("Chrome opened (fallback)")
                            return f"Opened app: {app_name}"
                        except Exception:
                            # Try Edge as fallback
                            self.log("Chrome not found, trying Edge...")
                            edge_path = self._find_edge_on_windows()
                            if edge_path:
                                subprocess.Popen([edge_path, "--guest"])
                                self.log("Opened Edge in guest mode")
                                return f"Opened Edge (guest mode) as Chrome fallback"
                            else:
                                return "Chrome and Edge not found. Please install one of them."
                elif app in {"edge", "microsoft edge"}:
                    edge_path = self._find_edge_on_windows()
                    if edge_path:
                        subprocess.Popen([edge_path, "--guest"])
                        return f"Opened app: {app_name} (guest mode)"
                    else:
                        return "Edge not found on system"
                elif app in {"vscode", "vs code", "visual studio code"}:
                    subprocess.Popen("code", shell=True)
                elif app in {"notepad"}:
                    subprocess.Popen("notepad", shell=True)
                else:
                    subprocess.Popen(app_name, shell=True)
            elif sys_platform := sys.platform:
                if "darwin" in sys_platform:
                    if app in {"chrome", "google chrome"}:
                        subprocess.Popen(["open", "-a", "Google Chrome", "--args", "--guest"])
                    elif app in {"edge", "microsoft edge"}:
                        subprocess.Popen(["open", "-a", "Microsoft Edge", "--args", "--guest"])
                    elif app in {"vscode", "vs code", "visual studio code"}:
                        subprocess.Popen(["open", "-a", "Visual Studio Code"])
                    else:
                        subprocess.Popen(["open", "-a", app_name])
                else:
                    # Linux / other unix
                    if app in {"chrome", "google chrome"}:
                        subprocess.Popen(["google-chrome", "--guest"])
                    elif app in {"edge", "microsoft edge"}:
                        subprocess.Popen(["microsoft-edge", "--guest"])
                    elif app in {"vscode", "vs code", "visual studio code"}:
                        subprocess.Popen(["code"])
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
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            }
            r = requests.get(url, headers=headers, timeout=25)
            r.raise_for_status()
            text = r.text

            if BeautifulSoup is not None:
                soup = BeautifulSoup(text, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                text = soup.get_text(separator="\n")
                lines = [ln.strip() for ln in text.splitlines()]
                text = "\n".join([ln for ln in lines if ln])

            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[TRUNCATED]"
            return text
        except Exception as e:
            return f"read_url failed: {e}"

    def browser_open(self, url: str) -> str:
        if not self._ensure_browser():
            return "Playwright is not installed. Run: pip install playwright && playwright install"
        try:
            self._page.goto(url, wait_until="domcontentloaded")
            return f"Browser opened: {url}"
        except Exception as e:
            return f"browser_open failed: {e}"

    def browser_click(self, selector: str) -> str:
        if not self._ensure_browser():
            return "Playwright is not installed."
        try:
            self._page.click(selector, timeout=15000)
            return f"Clicked: {selector}"
        except Exception as e:
            return f"browser_click failed: {e}"

    def browser_type(self, selector: str, text: str) -> str:
        if not self._ensure_browser():
            return "Playwright is not installed."
        try:
            self._page.fill(selector, text, timeout=15000)
            return f"Typed into {selector}"
        except Exception as e:
            return f"browser_type failed: {e}"

    def browser_read(self, selector: str = "body") -> str:
        if not self._ensure_browser():
            return "Playwright is not installed."
        try:
            content = self._page.locator(selector).inner_text(timeout=15000)
            content = re.sub(r"\n{3,}", "\n\n", content).strip()
            if len(content) > 12000:
                content = content[:12000] + "\n\n[TRUNCATED]"
            return content
        except Exception as e:
            return f"browser_read failed: {e}"

    def list_dir(self, path: str) -> str:
        try:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return f"Path not found: {p}"
            if p.is_file():
                return f"FILE: {p.name}\n{p.read_text(encoding='utf-8', errors='ignore')[:4000]}"
            items = []
            for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                suffix = "/" if item.is_dir() else ""
                items.append(item.name + suffix)
            return "\n".join(items)
        except Exception as e:
            return f"list_dir failed: {e}"

    def write_file(self, path: str, content: str) -> str:
        try:
            p = Path(path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content or "", encoding="utf-8")
            return f"Wrote file: {p}"
        except Exception as e:
            return f"write_file failed: {e}"

    def append_file(self, path: str, content: str) -> str:
        try:
            p = Path(path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(content or "")
            return f"Appended file: {p}"
        except Exception as e:
            return f"append_file failed: {e}"

    def _show_confirmation_dialog(self, title: str, message: str) -> bool:
        """Show a confirmation dialog to the user. Returns True if user accepts."""
        if self.gui_root is None:
            return True  # Auto-accept if no GUI available
        try:
            result = messagebox.askyesno(title, message)
            return result
        except Exception:
            return True

    def run_python(self, code: str) -> str:
        """
        Runs code in a separate Python process.
        Safer than exec in-process, but still powerful.
        Requires user confirmation for security.
        """
        # Show confirmation dialog
        code_preview = code[:200] + ("..." if len(code) > 200 else "")
        message = f"Execute Python code?\n\nCode:\n{code_preview}"
        
        if not self._show_confirmation_dialog("Confirm Python Execution", message):
            return "Python execution cancelled by user"
        
        try:
            with tempfile.TemporaryDirectory() as td:
                script = Path(td) / "snippet.py"
                script.write_text(code or "", encoding="utf-8")
                proc = subprocess.run(
                    [sys.executable, str(script)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                out = (proc.stdout or "").strip()
                err = (proc.stderr or "").strip()
                if proc.returncode != 0:
                    return f"run_python failed ({proc.returncode}):\n{err or out}"
                return out or "(no output)"
        except Exception as e:
            return f"run_python failed: {e}"

    def shell(self, command: str) -> str:
        """Execute shell command with user confirmation for security."""
        # Show confirmation dialog
        cmd_preview = command[:150] + ("..." if len(command) > 150 else "")
        message = f"Execute shell command?\n\nCommand:\n{cmd_preview}"
        
        if not self._show_confirmation_dialog("Confirm Shell Execution", message):
            return "Shell command cancelled by user"
        
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=90,
            )
            out = (proc.stdout or "").strip()
            err = (proc.stderr or "").strip()
            if proc.returncode != 0:
                return f"shell failed ({proc.returncode}):\n{err or out}"
            return out or "(no output)"
        except Exception as e:
            return f"shell failed: {e}"

    def execute(self, action: Dict[str, Any]) -> str:
        act = (action.get("action") or "").strip()
        inp = str(action.get("input") or "").strip()
        path = str(action.get("path") or "").strip()
        selector = str(action.get("selector") or "").strip()
        text = str(action.get("text") or "")
        code = str(action.get("code") or "")

        if act == "open_app":
            return self.open_app(inp)
        if act == "open_url":
            return self.open_url(inp)
        if act == "search_web":
            return self.search_web(inp)
        if act == "read_url":
            return self.read_url(inp)
        if act == "browser_open":
            return self.browser_open(inp)
        if act == "browser_click":
            return self.browser_click(selector or inp)
        if act == "browser_type":
            return self.browser_type(selector or inp, text or inp)
        if act == "browser_read":
            return self.browser_read(selector or inp or "body")
        if act == "write_file":
            return self.write_file(path or inp, text or inp)
        if act == "append_file":
            return self.append_file(path or inp, text or inp)
        if act == "list_dir":
            return self.list_dir(path or inp or ".")
        if act == "run_python":
            return self.run_python(code or inp)
        if act == "shell":
            return self.shell(inp)
        if act == "done":
            return "done"
        return f"Unknown action: {act}"


# =============================================================================
# Agent Core
# =============================================================================

class DesktopAgent:
    def __init__(self, client: LightningClient, toolkit: ToolKit, log_fn, gui_root=None):
        self.client = client
        self.tools = toolkit
        self.log = log_fn
        self.gui_root = gui_root

    def _make_messages(self, task: str, history: List[AgentStep]) -> List[Dict[str, str]]:
        compact_history = []
        for s in history[-8:]:
            compact_history.append(
                {
                    "turn": s.turn,
                    "action": s.action,
                    "result": s.result[:1200],
                }
            )

        user_content = f"""
TASK:
{task}

HISTORY:
{json.dumps(compact_history, ensure_ascii=False, indent=2)}

Decide the next best single action.
Return ONLY JSON.
"""

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content.strip()},
        ]

    def _call_planner(self, task: str, history: List[AgentStep]) -> Dict[str, Any]:
        messages = self._make_messages(task, history)
        raw = self.client.chat(messages, temperature=0.2, max_tokens=1200)
        parsed = _extract_first_json_block(raw)

        if not isinstance(parsed, dict):
            raise ValueError(f"Planner did not return a JSON object:\n{raw}")

        parsed.setdefault("action", "done")
        parsed.setdefault("input", "")
        parsed.setdefault("path", "")
        parsed.setdefault("selector", "")
        parsed.setdefault("text", "")
        parsed.setdefault("code", "")
        parsed.setdefault("reason", "")
        return parsed, raw

    def run_task(self, state: AgentState, max_steps: int = 8):
        state.running = True
        state.stop_requested = False
        history: List[AgentStep] = []

        self.log(f"\n=== Task ===\n{state.task}\n")
        self.log(f"Model: {self.client.model}")
        self.log(f"Max steps: {max_steps}")
        self.log("-" * 70)

        try:
            for turn in range(1, max_steps + 1):
                if state.stop_requested:
                    self.log("Stop requested. Halting agent.")
                    break

                self.log(f"\n[Step {turn}] Planning...")
                parsed, raw = self._call_planner(state.task, history)
                action = parsed.get("action", "done")

                self.log(f"RAW:\n{raw}")
                self.log(f"PARSED:\n{json.dumps(parsed, indent=2, ensure_ascii=False)}")

                if action == "done":
                    self.log("Planner returned done. Finished.")
                    history.append(AgentStep(turn, action, raw, parsed, "done"))
                    break

                result = self.tools.execute(parsed)
                self.log(f"RESULT:\n{result}")

                step = AgentStep(
                    turn=turn,
                    action=action,
                    raw=raw,
                    parsed=parsed,
                    result=result,
                )
                history.append(step)
                state.steps.append(step)

                # Feed result back into the task context for the next turn
                # by appending a short summary to the task itself.
                state.task = (
                    state.task
                    + "\n\nPrevious step result:\n"
                    + result[:1500]
                    + "\n\nContinue from here."
                )

                time.sleep(0.3)

        except Exception as e:
            self.log(f"\n[ERROR] {e}")
            self.log(traceback.format_exc())

        finally:
            state.running = False
            self.tools.shutdown()
            self.log("\n=== Agent stopped ===\n")


# =============================================================================
# GUI
# =============================================================================

class AgentApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Desktop AI Agent — Lightning + GPT-OSS")
        self.root.geometry("1180x860")

        self.state = AgentState()
        self.agent_thread: Optional[threading.Thread] = None

        self._build_ui()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        # Top config frame
        cfg = ttk.LabelFrame(self.root, text="Lightning AI Configuration")
        cfg.grid(row=0, column=0, sticky="ew", padx=12, pady=10)
        cfg.columnconfigure(1, weight=1)
        cfg.columnconfigure(3, weight=1)

        ttk.Label(cfg, text="API Key").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        self.api_key_var = tk.StringVar(value=os.getenv("LIGHTNING_API_KEY", ""))
        ttk.Entry(cfg, textvariable=self.api_key_var, show="*", width=42).grid(
            row=0, column=1, sticky="ew", padx=8, pady=8
        )

        ttk.Label(cfg, text="Teamspace / Project path").grid(row=0, column=2, sticky="w", padx=8, pady=8)
        self.teamspace_var = tk.StringVar(value=os.getenv("LIGHTNING_TEAMSPACE", ""))
        ttk.Entry(cfg, textvariable=self.teamspace_var, width=42).grid(
            row=0, column=3, sticky="ew", padx=8, pady=8
        )

        ttk.Label(cfg, text="Model").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        ttk.Entry(cfg, textvariable=self.model_var, width=42).grid(
            row=1, column=1, sticky="ew", padx=8, pady=8
        )

        ttk.Label(cfg, text="Max Steps").grid(row=1, column=2, sticky="w", padx=8, pady=8)
        self.steps_var = tk.IntVar(value=8)
        ttk.Spinbox(cfg, from_=1, to=20, textvariable=self.steps_var, width=10).grid(
            row=1, column=3, sticky="w", padx=8, pady=8
        )

        # Task frame
        task_frame = ttk.LabelFrame(self.root, text="Task")
        task_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
        task_frame.columnconfigure(0, weight=1)

        self.task_text = tk.Text(task_frame, height=6, wrap="word")
        self.task_text.grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        preset_frame = ttk.Frame(task_frame)
        preset_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        for i in range(5):
            preset_frame.columnconfigure(i, weight=1)

        presets = [
            ("Open Chrome", "open chrome"),
            ("Open VS Code", "open vscode"),
            ("Search Web", "search for latest AI agent patterns"),
            ("Open Localhost", "open http://127.0.0.1:3000 and inspect the page"),
            ("Write File", "create a file named notes.txt with a short hello message"),
        ]
        for idx, (label, value) in enumerate(presets):
            ttk.Button(
                preset_frame,
                text=label,
                command=lambda v=value: self._set_task(v),
            ).grid(row=0, column=idx, sticky="ew", padx=4)

        # Buttons
        btns = ttk.Frame(self.root)
        btns.grid(row=2, column=0, sticky="ew", padx=12, pady=6)
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        btns.columnconfigure(2, weight=1)
        btns.columnconfigure(3, weight=1)

        ttk.Button(btns, text="Start Agent", command=self.start_agent).grid(row=0, column=0, sticky="ew", padx=4)
        ttk.Button(btns, text="Stop", command=self.stop_agent).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(btns, text="Load Task from File", command=self.load_task_file).grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Button(btns, text="Clear Log", command=self.clear_log).grid(row=0, column=3, sticky="ew", padx=4)

        # Main split
        body = ttk.Frame(self.root)
        body.grid(row=3, column=0, sticky="nsew", padx=12, pady=8)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Left: log
        left = ttk.LabelFrame(body, text="Agent Log")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.log_text = tk.Text(left, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Right: help / live instructions
        right = ttk.LabelFrame(body, text="How it works")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        help_text = (
            "1) Put your Lightning API key and teamspace/project path.\n\n"
            "2) Type a task in the box above.\n\n"
            "3) The model returns one JSON action per step.\n\n"
            "4) The tool executes it locally.\n\n"
            "5) The result is fed back into the next step.\n\n"
            "Supported actions:\n"
            "- open_app\n- open_url\n- search_web\n- read_url\n"
            "- browser_open / browser_click / browser_type / browser_read\n"
            "- write_file / append_file / list_dir\n"
            "- run_python / shell\n"
            "- done\n\n"
            "Playwright browser control is optional. If not installed, the browser tools will tell you what to install.\n"
        )
        self.help_label = tk.Label(right, text=help_text, justify="left", anchor="nw", wraplength=500)
        self.help_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.root.rowconfigure(3, weight=1)

    def _set_task(self, value: str):
        self.task_text.delete("1.0", "end")
        self.task_text.insert("1.0", value)

    def log(self, text: str):
        def _append():
            self.log_text.insert("end", text + "\n")
            self.log_text.see("end")
        self.root.after(0, _append)

    def clear_log(self):
        self.log_text.delete("1.0", "end")

    def load_task_file(self):
        path = filedialog.askopenfilename(
            title="Select a text file with a task",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            content = Path(path).read_text(encoding="utf-8", errors="ignore")
            self._set_task(content)
            self.log(f"Loaded task from: {path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def stop_agent(self):
        self.state.stop_requested = True
        self.log("Stop requested...")

    def start_agent(self):
        if self.state.running:
            messagebox.showinfo("Agent running", "The agent is already running.")
            return

        api_key = self.api_key_var.get().strip()
        teamspace = self.teamspace_var.get().strip()
        model = self.model_var.get().strip() or DEFAULT_MODEL
        task = self.task_text.get("1.0", "end").strip()
        max_steps = int(self.steps_var.get())

        if not api_key:
            messagebox.showerror("Missing API key", "Please enter your Lightning API key.")
            return

        if not teamspace:
            messagebox.showerror(
                "Missing teamspace",
                "Please enter your Lightning teamspace / project path.\n"
                "Example: your-org/your-project"
            )
            return

        if not task:
            messagebox.showerror("Missing task", "Type a task first.")
            return

        self.clear_log()
        self.state = AgentState(task=task)
        self.state.running = True

        client = LightningClient(api_key=api_key, teamspace=teamspace, model=model)
        toolkit = ToolKit(self.log, gui_root=self.root)
        agent = DesktopAgent(client=client, toolkit=toolkit, log_fn=self.log, gui_root=self.root)

        def worker():
            try:
                agent.run_task(self.state, max_steps=max_steps)
            except Exception as e:
                self.log(f"[THREAD ERROR] {e}")
                self.log(traceback.format_exc())
            finally:
                self.state.running = False

        self.agent_thread = threading.Thread(target=worker, daemon=True)
        self.agent_thread.start()
        self.log("Agent started.")

    def run(self):
        self.root.mainloop()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    app = AgentApp()
    app.run()