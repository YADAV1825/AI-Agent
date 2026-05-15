from dataclasses import dataclass, field
from typing import Any, Dict, List

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
