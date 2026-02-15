"""Edge definition."""
from dataclasses import dataclass


@dataclass
class Edge:
    """Represents an edge in the knowledge graph."""
    source: str
    relation: str
    target: str
    confidence: float
