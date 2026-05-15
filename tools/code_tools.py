import sys
import time
import subprocess
import tempfile
from pathlib import Path

class CodeTools:
    def __init__(self, log_fn, stop_check=None):
        self.log = log_fn
        self.stop_check = stop_check

    def run_python(self, code: str, confirm_fn=None) -> str:
        code_preview = code[:200] + ("..." if len(code) > 200 else "")
        message = f"Execute Python code?\n\nCode:\n{code_preview}"
        if confirm_fn and not confirm_fn("Confirm Python Execution", message): 
            return "Python execution cancelled by user"
        try:
            with tempfile.TemporaryDirectory() as td:
                script = Path(td) / "snippet.py"
                script.write_text(code or "", encoding="utf-8")
                
                cmd = [sys.executable, str(script)]
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                deadline = time.time() + 60
                while proc.poll() is None:
                    if self.stop_check and self.stop_check():
                        proc.kill()
                        raise InterruptedError("Stop requested during Python execution")
                    if time.time() > deadline:
                        proc.kill()
                        raise subprocess.TimeoutExpired(cmd, 60)
                    time.sleep(0.25)
                out = (proc.stdout.read() if proc.stdout else "").strip()
                err = (proc.stderr.read() if proc.stderr else "").strip()
                rc = proc.returncode
                if rc != 0: return f"run_python failed ({rc}):\n{err or out}"
                return out or "(no output)"
        except InterruptedError as e: return str(e)
        except Exception as e: return f"run_python failed: {e}"
