import json
import time
import traceback
import base64
import re
import threading
from typing import Any, Callable, Dict, List, Optional
from tkinter import messagebox

from core.state import AgentState, AgentStep
from core.llm import OllamaClient
from core.prompts import SYSTEM_PROMPT
from utils.parsers import extract_first_json_block
# Note: ToolKit will be imported at runtime or passed in constructor

SAFETY_THRESHOLD = 40  # Ask user every N steps

# Keywords that indicate a reflex or screen-context command (skip plan.txt)
_REFLEX_PREFIXES = (
    "press ", "hit ", "type ", "click ", "scroll ", "open ",
    "take a screenshot", "take screenshot", "close ", "switch ",
    "minimize", "maximize", "alt tab", "enter", "escape",
)
_SCREEN_CONTEXT_KEYWORDS = (
    "you see", "you are seeing", "on the screen", "on screen",
    "that is showing", "what's on", "what is on", "visible",
    "that you see", "currently on", "looking at", "shown on",
    "access dom", "compose a mail", "compose an email", "compose mail",
)


class DesktopAgent:
    def __init__(self, client: OllamaClient, toolkit, log_fn, gui_root=None):
        self.client = client
        self.tools = toolkit
        self.log = log_fn
        self.gui_root = gui_root

    @staticmethod
    def _is_direct_command(task: str) -> bool:
        """Check if a task is a reflex or screen-context command (skip plan.txt)."""
        t = task.strip().lower()
        # Check reflex prefixes
        for prefix in _REFLEX_PREFIXES:
            if t.startswith(prefix):
                return True
        # Check screen-context keywords
        for kw in _SCREEN_CONTEXT_KEYWORDS:
            if kw in t:
                return True
        return False

    def _make_messages(self, task: str, history: List[AgentStep], enforce_plan: bool = False, latest_result: str = "") -> List[Dict[str, str]]:
        compact_history = []
        for s in history[-8:]:
            compact_history.append({
                "turn": s.turn,
                "action": s.action,
                "result": s.result[:1200]
            })

        user_content = f"TASK:\n{task}\n\nHISTORY:\n{json.dumps(compact_history, ensure_ascii=False, indent=2)}\n\n"
        if enforce_plan:
            user_content += (
                "THIS IS YOUR FIRST STEP. You MUST create plan.txt now. "
                "Output: {\"action\": \"write_file\", \"path\": \"plan.txt\", "
                "\"text\": \"<your numbered step-by-step plan>\", "
                "\"reason\": \"Creating execution plan\"}\n\n"
            )
        user_content += "Decide the next best single action. Return ONLY JSON. If the task is complete, return {\"action\": \"done\"}."
        
        user_msg = {"role": "user", "content": user_content.strip()}
        
        if latest_result:
            match = re.search(r"\[SCREENSHOT_TAKEN:\s*(.+?)\]", latest_result)
            if match:
                image_path = match.group(1)
                try:
                    with open(image_path, "rb") as img_file:
                        b64_image = base64.b64encode(img_file.read()).decode('utf-8')
                        user_msg["images"] = [b64_image]
                        user_msg["content"] += "\n\nLook at the screenshot and decide the next action."
                except Exception as e:
                    user_msg["content"] += f"\n[Failed to load image: {e}]"

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            user_msg
        ]

    def _call_planner(self, task: str, history: List[AgentStep], enforce_plan: bool = False, latest_result: str = "") -> Dict[str, Any]:
        messages = self._make_messages(task, history, enforce_plan=enforce_plan, latest_result=latest_result)
        raw = self.client.chat(messages, temperature=0.2)
        parsed = extract_first_json_block(raw)

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

    def _ask_continue(self, turn: int) -> bool:
        """Thread-safe 40-step confirmation dialog via Tk main thread.
        
        Returns True if user wants to continue, False to stop.
        If no GUI is available (headless), auto-continues.
        """
        if self.gui_root is None:
            return True

        result = threading.Event()
        answer = [True]  # mutable container for the dialog result

        def _show_dialog():
            try:
                resp = messagebox.askyesno(
                    "Continue?",
                    f"I have reached {turn} steps.\nShould I keep going?",
                    parent=self.gui_root
                )
                answer[0] = resp
            except Exception:
                answer[0] = False
            finally:
                result.set()

        self.gui_root.after(0, _show_dialog)
        result.wait(timeout=120)  # Wait up to 2 min for user response
        return answer[0]

    def run_task(self, state: AgentState, step_callback: Optional[Callable[[int], None]] = None):
        """Run the agent in an open-ended loop until done or stopped.
        
        Args:
            state: The AgentState with the task and stop flag.
            step_callback: Optional callable(step_num) invoked each step for UI updates.
        """
        state.running = True
        state.stop_requested = False
        history: List[AgentStep] = []
        original_task = state.task  # preserve original task text
        latest_result = ""
        turn = 0

        self.log(f"\n=== Task ===\n{state.task}\n")
        self.log(f"Model: {self.client.model}")
        self.log(f"Mode: Continuous (safety check every {SAFETY_THRESHOLD} steps)")
        self.log("-" * 70)

        try:
            while not state.stop_requested:
                turn += 1

                # Notify UI of current step
                if step_callback:
                    step_callback(turn)

                # --- 40-step safety check ---
                if turn > 1 and turn % SAFETY_THRESHOLD == 0:
                    self.log(f"\n⚠️  Reached {turn} steps — asking user to confirm continuation...")
                    if not self._ask_continue(turn):
                        self.log(f"\n>>> User chose to stop at step {turn}.")
                        break
                    self.log(f"✅ User confirmed — continuing past step {turn}.")

                # Enforce planning on step 1 — unless it's a reflex/screen-context command
                is_direct = self._is_direct_command(original_task)
                enforce_plan = (turn == 1 and not is_direct)
                label = 'Creating plan...' if enforce_plan else ('Executing...' if turn == 1 and is_direct else 'Planning...')
                self.log(f"\n[Step {turn}] {label}")

                try:
                    parsed, raw = self._call_planner(state.task, history, enforce_plan=enforce_plan, latest_result=latest_result)
                except InterruptedError:
                    self.log("\n>>> Stop requested during model call. Halting.")
                    break

                action = parsed.get("action", "done")

                self.log(f"ACTION: {action}")
                self.log(f"REASON: {parsed.get('reason', 'N/A')}")
                self.log(f"VISION ANALYSIS: {parsed.get('vision_analysis', 'None')}")

                # --- EARLY STOPPING: clean break on "done" ---
                if action == "done":
                    reason = parsed.get('reason', 'Task complete.')
                    self.log(f"\n✅ Agent finished: {reason}")
                    self.log(f"   Completed in {turn} steps.")
                    history.append(AgentStep(turn, action, raw, parsed, "done"))
                    break

                # Execute the tool action
                try:
                    result = self.tools.execute(parsed)
                except InterruptedError:
                    self.log("\n>>> Stop requested during tool execution. Halting.")
                    break

                self.log(f"RESULT:\n{result}")

                step = AgentStep(turn=turn, action=action, raw=raw, parsed=parsed, result=result)
                history.append(step)
                state.steps.append(step)

                # Append result context (don't bloat task string endlessly)
                state.task = original_task + f"\n\nPrevious step ({turn}) result:\n{result[:1500]}\n\nContinue from here."
                latest_result = result
                time.sleep(0.3)

        except Exception as e:
            self.log(f"\n[ERROR] {e}")
            self.log(traceback.format_exc())

        finally:
            state.running = False
            # Keep browser open so user can continue interacting with it
            # Don't call shutdown() - browser persists after task completes
            self.log("\n=== Agent stopped ===")
