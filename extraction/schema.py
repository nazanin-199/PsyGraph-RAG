from pydantic import BaseModel
from typing import List, Optional


class ExtractedEntities(BaseModel):
    problems: List[str] = []
    symptoms: List[str] = []
    concepts: List[str] = []
    advice: List[str] = []
    context_factors: List[str] = []


class ExtractedRelation(BaseModel):
    source: str
    relation: str
    target: str
    confidence: float


class ExtractionResult(BaseModel):
    entities: ExtractedEntities
    relations: List[ExtractedRelation]
