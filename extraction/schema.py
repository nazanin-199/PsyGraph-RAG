"""
Schema definitions for entity extraction.
"""
from pydantic import BaseModel, Field
from typing import List


class ExtractedEntities(BaseModel):
    """Structured entities extracted from text."""
    symptoms: List[str] = Field(default_factory=list)
    disorders: List[str] = Field(default_factory=list)
    therapies: List[str] = Field(default_factory=list)
    emotions: List[str] = Field(default_factory=list)
    cognitive_patterns: List[str] = Field(default_factory=list)


class ExtractedRelation(BaseModel):
    """Represents a relation between entities."""
    source: str
    relation: str
    target: str
    confidence: float = 1.0


class ExtractionResult(BaseModel):
    """Complete extraction result with entities and relations."""
    entities: ExtractedEntities
    relations: List[ExtractedRelation]
