from dataclasses import dataclass


@dataclass
class Edge:
    source: str
    relation: str
    target: str
    confidence: float
