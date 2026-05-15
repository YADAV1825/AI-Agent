import tkinter as tk
from tkinter import messagebox
import threading
import os
import traceback
from typing import Optional
from datetime import datetime
from pathlib import Path

from core.state import AgentState
from core.llm import LightningClient
from core.agent import DesktopAgent
from tools.toolkit import ToolKit

# --- Voice Integration (graceful fallback) ---
_VOICE_AVAILABLE = False
try:
    import speech_recognition as sr
    _VOICE_AVAILABLE = True
except ImportError:
    pass


class AgentApp:
    """Compact PiP Voice-First floating widget for the Desktop AI Agent."""

    # --- Theme constants ---
    BG = "#1a1a2e"
    BG_ENTRY = "#16213e"
    BG_LOG = "#0d1117"
    FG = "#e0e0e0"
    FG_DIM = "#6e7681"
    ACCENT = "#0f3460"
    ACCENT_HOVER = "#533483"
    MIC_COLOR = "#e94560"
    MIC_HOVER = "#ff6b6b"
    STOP_COLOR = "#c0392b"
    STOP_HOVER = "#e74c3c"
    LOG_BTN_COLOR = "#2d4059"
    LOG_BTN_HOVER = "#3b5278"
    STATUS_FG = "#8d99ae"

    # Persistent log directory
    LOGS_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent / "logs"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Opus Agent")
        self.root.geometry("400x190")
        self.root.resizable(False, False)
        self.root.attributes('-topmost', True)
        self.root.configure(bg=self.BG)

        # Remove default window icon padding for a cleaner look
        try:
            self.root.iconbitmap(default="")
        except Exception:
            pass

        self.state = AgentState()
        self.agent_thread: Optional[threading.Thread] = None
        self.toolkit: Optional[ToolKit] = None
        self._listening = False

        # --- Log buffer ---
        self._log_lines = []        # list of (timestamp, text) tuples
        self._log_window = None     # Toplevel reference
        self._log_text_widget = None  # Text widget inside log window
        self._current_log_file = None  # file handle for persistent log

        # Ensure logs directory exists
        self.LOGS_DIR.mkdir(exist_ok=True)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # --- Title bar row ---
        title_frame = tk.Frame(self.root, bg=self.BG)
        title_frame.pack(fill="x", padx=14, pady=(10, 4))

        tk.Label(
            title_frame, text="🤖 Opus Agent", font=("Segoe UI Semibold", 12),
            bg=self.BG, fg=self.FG
        ).pack(side="left")

        self.status_label = tk.Label(
            title_frame, text="● Idle", font=("Segoe UI", 9),
            bg=self.BG, fg=self.STATUS_FG
        )
        self.status_label.pack(side="right")

        # --- Input row ---
        input_frame = tk.Frame(self.root, bg=self.BG)
        input_frame.pack(fill="x", padx=14, pady=6)

        # Mic button
        self.mic_btn = tk.Button(
            input_frame, text="🎤", font=("Segoe UI", 14),
            bg=self.MIC_COLOR, fg="white", relief="flat",
            activebackground=self.MIC_HOVER, activeforeground="white",
            width=3, cursor="hand2",
            command=self._start_listening
        )
        self.mic_btn.pack(side="left", padx=(0, 6))

        if not _VOICE_AVAILABLE:
            self.mic_btn.configure(state="disabled", bg="#555555", cursor="arrow")

        # Hover effects for mic
        if _VOICE_AVAILABLE:
            self.mic_btn.bind("<Enter>", lambda e: self.mic_btn.configure(bg=self.MIC_HOVER))
            self.mic_btn.bind("<Leave>", lambda e: self.mic_btn.configure(
                bg=self.MIC_COLOR if not self._listening else "#ff9800"
            ))

        # Text entry
        self.task_var = tk.StringVar()
        self.task_entry = tk.Entry(
            input_frame, textvariable=self.task_var,
            font=("Segoe UI", 11), bg=self.BG_ENTRY, fg=self.FG,
            insertbackground=self.FG, relief="flat",
            highlightthickness=1, highlightcolor=self.ACCENT_HOVER,
            highlightbackground=self.ACCENT
        )
        self.task_entry.pack(side="left", fill="x", expand=True, ipady=6)
        self.task_entry.bind("<Return>", lambda e: self.start_agent())

        # --- Button row ---
        btn_frame = tk.Frame(self.root, bg=self.BG)
        btn_frame.pack(fill="x", padx=14, pady=(4, 4))

        self.start_btn = tk.Button(
            btn_frame, text="▶  Start", font=("Segoe UI Semibold", 10),
            bg=self.ACCENT, fg=self.FG, relief="flat",
            activebackground=self.ACCENT_HOVER, activeforeground="white",
            cursor="hand2", command=self.start_agent
        )
        self.start_btn.pack(side="left", fill="x", expand=True, ipady=4, padx=(0, 3))
        self.start_btn.bind("<Enter>", lambda e: self.start_btn.configure(bg=self.ACCENT_HOVER))
        self.start_btn.bind("<Leave>", lambda e: self.start_btn.configure(bg=self.ACCENT))

        # Logs button
        self.log_btn = tk.Button(
            btn_frame, text="📋 Logs", font=("Segoe UI Semibold", 10),
            bg=self.LOG_BTN_COLOR, fg=self.FG, relief="flat",
            activebackground=self.LOG_BTN_HOVER, activeforeground="white",
            cursor="hand2", command=self._toggle_log_window
        )
        self.log_btn.pack(side="left", fill="x", expand=True, ipady=4, padx=(3, 3))
        self.log_btn.bind("<Enter>", lambda e: self.log_btn.configure(bg=self.LOG_BTN_HOVER))
        self.log_btn.bind("<Leave>", lambda e: self.log_btn.configure(bg=self.LOG_BTN_COLOR))

        self.stop_btn = tk.Button(
            btn_frame, text="⏹  Stop", font=("Segoe UI Semibold", 10),
            bg=self.STOP_COLOR, fg="white", relief="flat",
            activebackground=self.STOP_HOVER, activeforeground="white",
            cursor="hand2", command=self.stop_agent
        )
        self.stop_btn.pack(side="right", fill="x", expand=True, ipady=4, padx=(3, 0))
        self.stop_btn.bind("<Enter>", lambda e: self.stop_btn.configure(bg=self.STOP_HOVER))
        self.stop_btn.bind("<Leave>", lambda e: self.stop_btn.configure(bg=self.STOP_COLOR))

        # --- Footer hint ---
        hint = "Voice unavailable — install SpeechRecognition + PyAudio" if not _VOICE_AVAILABLE else "Press 🎤 or type a command and hit Enter"
        tk.Label(
            self.root, text=hint, font=("Segoe UI", 8),
            bg=self.BG, fg="#555"
        ).pack(side="bottom", pady=(0, 6))

    # ------------------------------------------------- Focus Management
    def yield_focus(self):
        """Minimize the PiP widget so it doesn't steal focus from the target window.
        Called automatically before press_key/type_text/take_screenshot."""
        def _do():
            try:
                self.root.iconify()
                # Also minimize log window if open
                if self._log_window and self._log_window.winfo_exists():
                    self._log_window.iconify()
            except Exception:
                pass
        self.root.after(0, _do)
        import time
        time.sleep(0.3)  # give OS time to process the minimize

    def reclaim_focus(self):
        """Restore the PiP widget after the action completes."""
        def _do():
            try:
                self.root.deiconify()
                self.root.attributes('-topmost', True)
                if self._log_window and self._log_window.winfo_exists():
                    self._log_window.deiconify()
                    self._log_window.attributes('-topmost', True)
            except Exception:
                pass
        self.root.after(0, _do)

    # ---------------------------------------------------------- Log Window
    def _toggle_log_window(self):
        """Open the log viewer window, or bring it to front if already open."""
        if self._log_window is not None and self._log_window.winfo_exists():
            self._log_window.lift()
            self._log_window.focus_force()
            return

        self._log_window = tk.Toplevel(self.root)
        self._log_window.title("Opus Agent — Logs")
        self._log_window.geometry("720x480")
        self._log_window.configure(bg=self.BG)
        self._log_window.attributes('-topmost', True)

        # --- Top bar with controls ---
        top_bar = tk.Frame(self._log_window, bg=self.BG)
        top_bar.pack(fill="x", padx=10, pady=(8, 4))

        tk.Label(
            top_bar, text="📋 Agent Logs", font=("Segoe UI Semibold", 12),
            bg=self.BG, fg=self.FG
        ).pack(side="left")

        # Clear button
        clear_btn = tk.Button(
            top_bar, text="🗑 Clear", font=("Segoe UI", 9),
            bg=self.STOP_COLOR, fg="white", relief="flat",
            activebackground=self.STOP_HOVER, cursor="hand2",
            command=self._clear_logs
        )
        clear_btn.pack(side="right", padx=(4, 0))
        clear_btn.bind("<Enter>", lambda e: clear_btn.configure(bg=self.STOP_HOVER))
        clear_btn.bind("<Leave>", lambda e: clear_btn.configure(bg=self.STOP_COLOR))

        # Copy All button
        copy_btn = tk.Button(
            top_bar, text="📄 Copy All", font=("Segoe UI", 9),
            bg=self.LOG_BTN_COLOR, fg=self.FG, relief="flat",
            activebackground=self.LOG_BTN_HOVER, cursor="hand2",
            command=self._copy_all_logs
        )
        copy_btn.pack(side="right", padx=(4, 0))
        copy_btn.bind("<Enter>", lambda e: copy_btn.configure(bg=self.LOG_BTN_HOVER))
        copy_btn.bind("<Leave>", lambda e: copy_btn.configure(bg=self.LOG_BTN_COLOR))

        # History button — browse past runs
        history_btn = tk.Button(
            top_bar, text="📂 History", font=("Segoe UI", 9),
            bg=self.ACCENT, fg=self.FG, relief="flat",
            activebackground=self.ACCENT_HOVER, cursor="hand2",
            command=self._show_history
        )
        history_btn.pack(side="right", padx=(4, 0))
        history_btn.bind("<Enter>", lambda e: history_btn.configure(bg=self.ACCENT_HOVER))
        history_btn.bind("<Leave>", lambda e: history_btn.configure(bg=self.ACCENT))

        # --- Log text area with scrollbar ---
        log_frame = tk.Frame(self._log_window, bg=self.BG_LOG)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self._log_text_widget = tk.Text(
            log_frame, wrap="word",
            bg=self.BG_LOG, fg="#c9d1d9",
            font=("Cascadia Code", 10),
            relief="flat", borderwidth=0,
            insertbackground=self.FG,
            selectbackground=self.ACCENT_HOVER,
            selectforeground="white",
            padx=12, pady=10
        )
        self._log_text_widget.grid(row=0, column=0, sticky="nsew")

        scrollbar = tk.Scrollbar(
            log_frame, orient="vertical",
            command=self._log_text_widget.yview,
            bg=self.BG, troughcolor=self.BG_LOG,
            activebackground=self.ACCENT_HOVER
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._log_text_widget.configure(yscrollcommand=scrollbar.set)

        # Configure text tags for colored log entries
        self._log_text_widget.tag_configure("timestamp", foreground="#6e7681", font=("Cascadia Code", 9))
        self._log_text_widget.tag_configure("action", foreground="#58a6ff", font=("Cascadia Code", 10, "bold"))
        self._log_text_widget.tag_configure("result", foreground="#7ee787")
        self._log_text_widget.tag_configure("error", foreground="#f85149")
        self._log_text_widget.tag_configure("voice", foreground="#d2a8ff")
        self._log_text_widget.tag_configure("status", foreground="#79c0ff")
        self._log_text_widget.tag_configure("separator", foreground="#30363d")

        # Populate with existing log buffer
        for ts, line in self._log_lines:
            self._append_to_log_widget(ts, line)

        # Allow user to select/copy but not edit
        self._log_text_widget.configure(state="disabled")

        # Handle log window close
        self._log_window.protocol("WM_DELETE_WINDOW", self._on_log_window_close)

    def _on_log_window_close(self):
        """Clean up log window reference on close."""
        if self._log_window:
            self._log_window.destroy()
            self._log_window = None
            self._log_text_widget = None

    def _append_to_log_widget(self, timestamp: str, text: str):
        """Append a single log line to the log text widget with syntax coloring."""
        if self._log_text_widget is None:
            return

        self._log_text_widget.configure(state="normal")

        # Determine tag based on content
        tag = "normal"
        if text.startswith("ACTION:"):
            tag = "action"
        elif text.startswith("RESULT:"):
            tag = "result"
        elif "[ERROR]" in text or "[THREAD ERROR]" in text or "Error" in text:
            tag = "error"
        elif "[Voice]" in text:
            tag = "voice"
        elif text.startswith("[Agent]") or text.startswith("[App]") or text.startswith("[Step"):
            tag = "status"
        elif text.startswith("---") or text.startswith("==="):
            tag = "separator"

        self._log_text_widget.insert("end", f"{timestamp}  ", "timestamp")
        self._log_text_widget.insert("end", f"{text}\n", tag)
        self._log_text_widget.see("end")
        self._log_text_widget.configure(state="disabled")

    def _copy_all_logs(self):
        """Copy all log content to clipboard."""
        all_text = "\n".join(f"[{ts}] {line}" for ts, line in self._log_lines)
        self.root.clipboard_clear()
        self.root.clipboard_append(all_text)
        # Brief visual feedback
        self._update_status("Logs copied!", "#7ee787")
        self.root.after(1500, lambda: self._update_status(
            "Idle" if not self.state.running else f"Running Step {len(self.state.steps)}",
            self.STATUS_FG if not self.state.running else self.ACCENT_HOVER
        ))

    def _clear_logs(self):
        """Clear the log buffer and the log widget."""
        self._log_lines.clear()
        if self._log_text_widget:
            self._log_text_widget.configure(state="normal")
            self._log_text_widget.delete("1.0", "end")
            self._log_text_widget.configure(state="disabled")

    def _show_history(self):
        """Show a list of past log files the user can load."""
        log_files = sorted(self.LOGS_DIR.glob("run_*.log"), reverse=True)
        if not log_files:
            messagebox.showinfo("No History", "No past runs found in the logs/ folder.")
            return

        # Create a selection dialog
        hist_win = tk.Toplevel(self._log_window or self.root)
        hist_win.title("Past Runs")
        hist_win.geometry("420x350")
        hist_win.configure(bg=self.BG)
        hist_win.attributes('-topmost', True)

        tk.Label(
            hist_win, text="📂 Select a past run to view:",
            font=("Segoe UI Semibold", 11), bg=self.BG, fg=self.FG
        ).pack(padx=10, pady=(10, 6))

        listbox_frame = tk.Frame(hist_win, bg=self.BG)
        listbox_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        listbox = tk.Listbox(
            listbox_frame, bg=self.BG_LOG, fg="#c9d1d9",
            font=("Cascadia Code", 10), selectbackground=self.ACCENT_HOVER,
            selectforeground="white", relief="flat", borderwidth=0
        )
        listbox.pack(side="left", fill="both", expand=True)

        sb = tk.Scrollbar(listbox_frame, orient="vertical", command=listbox.yview)
        sb.pack(side="right", fill="y")
        listbox.configure(yscrollcommand=sb.set)

        for f in log_files:
            # Show friendly name: run_2026-04-28_17-07-32.log → 2026-04-28 17:07:32
            name = f.stem.replace("run_", "").replace("_", " ", 1).replace("-", ":", 2)
            listbox.insert("end", f"  {name}")

        def _load_selected():
            sel = listbox.curselection()
            if not sel:
                return
            chosen_file = log_files[sel[0]]
            try:
                content = chosen_file.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                messagebox.showerror("Error", f"Could not read file: {e}")
                return

            # Show in a new viewer window
            viewer = tk.Toplevel(hist_win)
            viewer.title(f"Log: {chosen_file.name}")
            viewer.geometry("720x480")
            viewer.configure(bg=self.BG)
            viewer.attributes('-topmost', True)

            txt = tk.Text(
                viewer, wrap="word", bg=self.BG_LOG, fg="#c9d1d9",
                font=("Cascadia Code", 10), relief="flat",
                padx=12, pady=10
            )
            txt.pack(fill="both", expand=True, padx=10, pady=10)
            txt.insert("1.0", content)
            txt.configure(state="disabled")

        load_btn = tk.Button(
            hist_win, text="Open Selected", font=("Segoe UI Semibold", 10),
            bg=self.ACCENT, fg=self.FG, relief="flat",
            activebackground=self.ACCENT_HOVER, cursor="hand2",
            command=_load_selected
        )
        load_btn.pack(pady=(0, 10))

    # -------------------------------------------------------------- Voice
    def _start_listening(self):
        if not _VOICE_AVAILABLE or self._listening:
            return

        self._listening = True
        self.mic_btn.configure(bg="#ff9800", text="🔴")
        self._update_status("Listening...", "#ff9800")

        def _listen_worker():
            recognizer = sr.Recognizer()
            text = ""
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    self.log("[Voice] Listening...")
                    audio = recognizer.listen(source, timeout=8, phrase_time_limit=15)

                self._update_status("Processing speech...", "#ff9800")
                text = recognizer.recognize_google(audio)
                self.log(f"[Voice] Recognized: {text}")

            except sr.WaitTimeoutError:
                self.log("[Voice] No speech detected (timeout).")
                self._update_status("No speech detected", self.STATUS_FG)
            except sr.UnknownValueError:
                self.log("[Voice] Could not understand audio.")
                self._update_status("Couldn't understand", self.STATUS_FG)
            except sr.RequestError as e:
                self.log(f"[Voice] Recognition service error: {e}")
                self._update_status("Voice error", self.STOP_COLOR)
            except Exception as e:
                self.log(f"[Voice] Error: {e}")
                self._update_status("Voice error", self.STOP_COLOR)
            finally:
                self._listening = False
                self.root.after(0, lambda: self.mic_btn.configure(bg=self.MIC_COLOR, text="🎤"))

            if text:
                # Populate entry and auto-start
                self.root.after(0, lambda t=text: self._voice_submit(t))

        threading.Thread(target=_listen_worker, daemon=True).start()

    def _voice_submit(self, text: str):
        self.task_var.set(text)
        self.start_agent(task=text)

    # -------------------------------------------------------------- Status
    def _update_status(self, text: str, color: str = None):
        def _do():
            self.status_label.configure(text=f"● {text}", fg=color or self.STATUS_FG)
        self.root.after(0, _do)

    # ----------------------------------------------------------------- Log
    def log(self, text: str):
        """Log to console + internal buffer + live log window + persistent file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(text)

        # Store in buffer
        self._log_lines.append((timestamp, text))

        # Write to persistent log file
        if self._current_log_file is not None:
            try:
                self._current_log_file.write(f"[{timestamp}] {text}\n")
                self._current_log_file.flush()
            except Exception:
                pass

        # If log window is open, append live
        if self._log_text_widget is not None:
            self.root.after(0, lambda ts=timestamp, t=text: self._append_to_log_widget(ts, t))

    # ------------------------------------------------------------- Agent
    def stop_agent(self):
        self.state.stop_requested = True
        self._update_status("Stopping...", self.STOP_COLOR)
        self.log("[Agent] Stop requested.")

    def start_agent(self, task: str = None):
        if self.state.running:
            messagebox.showinfo("Agent running", "The agent is already running.")
            return

        task = task or self.task_var.get().strip()
        if not task:
            messagebox.showwarning("No task", "Please speak or type a task first.")
            return

        # Load credentials from env vars (via LightningClient defaults)
        client = LightningClient()

        # Quick validation
        if not client.api_key.strip():
            messagebox.showerror("Missing API Key", "Set LIGHTNING_API_KEY in your .env file.")
            return
        if not client.teamspace.strip():
            messagebox.showerror("Missing Teamspace", "Set LIGHTNING_TEAMSPACE in your .env file.")
            return

        self.state = AgentState(task=task)
        self.state.running = True
        self._update_status("Starting...", self.ACCENT_HOVER)

        # Immediately return focus to the window the user was using before clicking the PiP
        self._drop_focus()

        # Start a new persistent log file for this run
        self._start_log_file(task)

        self.toolkit = ToolKit(self.log, gui_root=self.root, pip_app=self)
        agent = DesktopAgent(client=client, toolkit=self.toolkit, log_fn=self.log, gui_root=self.root)

        step_counter = {"n": 0}  # mutable reference for status updates

        def _status_callback(step_num: int):
            """Called by the agent each step to update the widget."""
            step_counter["n"] = step_num
            self._update_status(f"Running Step {step_num}", self.ACCENT_HOVER)

        def worker():
            try:
                agent.run_task(self.state, step_callback=_status_callback)
            except Exception as e:
                self.log(f"[THREAD ERROR] {e}")
                self.log(traceback.format_exc())
            finally:
                self.state.running = False
                self._update_status("Idle", self.STATUS_FG)

        self.agent_thread = threading.Thread(target=worker, daemon=True)
        self.agent_thread.start()
        self.log(f"[Agent] Started task: {task}")

    def _drop_focus(self):
        """Simulate Alt+Tab to return focus to the previously active window."""
        import pyautogui
        import time
        try:
            pyautogui.keyUp("enter")
            time.sleep(0.1)
            pyautogui.hotkey("alt", "tab")
            time.sleep(0.2)
        except Exception as e:
            self.log(f"[Warning] Could not drop focus: {e}")

    # --------------------------------------------------------- Log File
    def _start_log_file(self, task: str):
        """Create a new timestamped log file for this run."""
        # Close previous log file if open
        if self._current_log_file is not None:
            try:
                self._current_log_file.close()
            except Exception:
                pass

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = self.LOGS_DIR / f"run_{ts}.log"
        try:
            self._current_log_file = open(log_path, "w", encoding="utf-8")
            self._current_log_file.write(f"=== Task: {task} ===\n")
            self._current_log_file.write(f"=== Started: {datetime.now().isoformat()} ===\n\n")
            self._current_log_file.flush()
        except Exception as e:
            print(f"[Warning] Could not create log file: {e}")
            self._current_log_file = None

    # ------------------------------------------------------------ Cleanup
    def on_closing(self):
        self.log("[App] Closing... shutting down browser.")
        if self._current_log_file:
            try:
                self._current_log_file.close()
            except Exception:
                pass
        if self._log_window and self._log_window.winfo_exists():
            self._log_window.destroy()
        if self.toolkit:
            self.toolkit.shutdown(force=True)
        self.root.destroy()

    def run(self):
        self.root.mainloop()
