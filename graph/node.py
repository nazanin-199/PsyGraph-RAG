from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Node:
    id: str
    label: str
    type: str
    embedding: list
    degree: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
