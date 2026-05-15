from typing import Any, Dict
from tkinter import messagebox

from tools.file_tools import FileTools
from tools.os_tools import OSTools
from tools.code_tools import CodeTools


class ToolKit:
    def __init__(self, log_fn, gui_root=None, stop_check=None, pip_app=None):
        self.log = log_fn
        self.gui_root = gui_root
        self.stop_check = stop_check
        self.pip_app = pip_app  # reference to AgentApp for focus management
        
        # Initialize modules
        self.fs = FileTools(log_fn)
        self.os = OSTools(log_fn, stop_check=stop_check)
        self.code = CodeTools(log_fn, stop_check=stop_check)

    def shutdown(self, force=False):
        pass

    def _show_confirmation_dialog(self, title: str, message: str) -> bool:
        if self.gui_root is None: return True
        try: return messagebox.askyesno(title, message)
        except Exception: return True

    def execute(self, action: Dict[str, Any]) -> str:
        act = (action.get("action") or "").strip()
        inp = str(action.get("input") or "").strip()
        path = str(action.get("path") or "").strip()
        text = str(action.get("text") or "")
        code = str(action.get("code") or "")

        try:
            result = self._dispatch(act, inp, path, text, code)
        finally:
            pass

        return result

    def _dispatch(self, act, inp, path, text, code) -> str:
        """Route the action to the correct tool."""
        # --- OS Tools ---
        if act == "open_app": return self.os.open_app(inp)
        if act == "open_url": return self.os.open_url(inp)
        if act == "search_web": return self.os.search_web(inp)
        if act == "read_url": return self.os.read_url(inp)
        if act == "press_key": return self.os.press_key(inp)
        if act == "type_text": return self.os.type_text(text or inp)
        if act == "take_screenshot": return self.os.take_screenshot()

        # --- File Tools ---
        if act == "read_file": return self.fs.read_file(path or inp)
        if act == "write_file": return self.fs.write_file(path or inp, text or inp)
        if act == "append_file": return self.fs.append_file(path or inp, text or inp)
        if act == "list_dir": return self.fs.list_dir(path or inp or ".")

        # --- Code Tools ---
        if act == "run_python": return self.code.run_python(code or inp, confirm_fn=self._show_confirmation_dialog)
        if act == "shell": return self.os.shell(inp, confirm_fn=self._show_confirmation_dialog)

        if act == "done": return "done"
        
        return f"Unknown action: {act}"

