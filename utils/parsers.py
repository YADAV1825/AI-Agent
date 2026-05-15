import json
import re
from typing import Any

def extract_first_json_block(text: str) -> Any:
    """Best-effort JSON extraction."""
    if not text:
        raise ValueError("Empty model response")

    cleaned = text.strip()
    
    # Remove code fences if present
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.IGNORECASE)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

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
