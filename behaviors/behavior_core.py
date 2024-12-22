"Behavior Core"
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class BehaviorCore:
    "Behavior Core"
    Name: str
    Description: str
    AutoHead: bool
    Execute : Callable[[bool, Any], bool]
    IsReady : Callable[[Any], bool]
