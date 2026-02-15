"""Node definition."""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class Node:
    """Represents a node in the knowledge graph."""
    id: str
    label: str
    type: str
    embedding: Optional[List[float]] = None
    degree: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
