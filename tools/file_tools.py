from pathlib import Path

class FileTools:
    def __init__(self, log_fn):
        self.log = log_fn

    def list_dir(self, path: str) -> str:
        try:
            p = Path(path).expanduser().resolve()
            if not p.exists(): return f"Path not found: {p}"
            if p.is_file(): return f"FILE: {p.name}\n{p.read_text(encoding='utf-8', errors='ignore')[:4000]}"
            items = [item.name + ("/" if item.is_dir() else "") for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))]
            return "\n".join(items)
        except Exception as e: return f"list_dir failed: {e}"

    def write_file(self, path: str, content: str) -> str:
        try:
            p = Path(path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content or "", encoding="utf-8")
            return f"Wrote file: {p}"
        except Exception as e: return f"write_file failed: {e}"

    def read_file(self, path: str, max_chars: int = 8000) -> str:
        """Read a file's contents (used for plan.txt review)."""
        try:
            p = Path(path).expanduser().resolve()
            if not p.exists(): return f"File not found: {p}"
            if not p.is_file(): return f"Not a file: {p}"
            content = p.read_text(encoding="utf-8", errors="ignore")
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[TRUNCATED]"
            return content
        except Exception as e: return f"read_file failed: {e}"

    def append_file(self, path: str, content: str) -> str:
        try:
            p = Path(path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f: f.write(content or "")
            return f"Appended file: {p}"
        except Exception as e: return f"append_file failed: {e}"
