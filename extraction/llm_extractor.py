"""
LLM-based entity extraction with robust error handling.
"""
from typing import Optional
import json
import re
import logging
from openai import OpenAI, APIError

from .schema import ExtractionResult, ExtractedEntities
from exceptions import ExtractionError

logger = logging.getLogger(__name__)


EXTRACTION_PROMPT = """You are an expert in psychology and mental health. Extract structured information from the given text.

Extract:
1. Symptoms (anxiety, depression, insomnia, etc.)
2. Disorders (PTSD, OCD, bipolar disorder, etc.)
3. Therapies (CBT, meditation, medication, etc.)
4. Emotions (fear, sadness, joy, etc.)
5. Cognitive patterns (negative thinking, rumination, etc.)

Also identify relations between entities (e.g., "CBT treats anxiety", "stress causes insomnia").

Return ONLY valid JSON in this exact format:
{
  "entities": {
    "symptoms": [],
    "disorders": [],
    "therapies": [],
    "emotions": [],
    "cognitive_patterns": []
  },
  "relations": [
    {
      "source": "entity1",
      "relation": "treats|causes|related_to|symptom_of",
      "target": "entity2",
      "confidence": 0.9
    }
  ]
}

Do not include any text before or after the JSON."""


class LLMExtractor:
    """
    Extracts structured entities and relations from text using LLM.
    
    Features:
    - Robust JSON parsing (handles markdown code blocks)
    - Graceful fallback on errors
    - Input validation
    - Timeout handling
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize LLM extractor.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key
            timeout: Request timeout in seconds
        """
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1" )
        self.model = model
        self.timeout = timeout
        self._extraction_count = 0
        self._error_count = 0
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extract entities and relations with robust JSON parsing.
        
        Returns empty result on failure rather than crashing pipeline.
        
        Args:
            text: Input text to analyze
            
        Returns:
            ExtractionResult with entities and relations
        """
        if not text or len(text.strip()) < 10:
            logger.warning("Text too short for extraction")
            return self._empty_result()
        
        # Truncate very long texts
        if len(text) > 4000:
            logger.debug(f"Truncating text from {len(text)} to 4000 chars")
            text = text[:4000]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,  # Deterministic
                timeout=self.timeout
            )
            
            raw_output = response.choices[0].message.content
            
            # Strip markdown code blocks
            json_str = self._clean_json_output(raw_output)
            
            # Parse and validate
            try:
                result = ExtractionResult.model_validate_json(json_str)
            except Exception as parse_error:
                logger.error(f"JSON parsing failed: {parse_error}")
                logger.debug(f"Raw output: {raw_output[:500]}")
                return self._empty_result()
            
            # Sanity check: at least some entities or relations
            total_entities = sum(len(v) for v in result.entities.dict().values())
            if total_entities == 0 and len(result.relations) == 0:
                logger.warning(f"Extraction returned empty result for text: {text[:100]}")
            
            self._extraction_count += 1
            return result
            
        except APIError as e:
            self._error_count += 1
            logger.error(f"LLM API error during extraction: {e}")
            return self._empty_result()
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Extraction failed: {e}")
            return self._empty_result()
    
    def _clean_json_output(self, raw: str) -> str:
        """Remove markdown formatting and extract JSON."""
        # Remove markdown code blocks
        cleaned = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE | re.DOTALL)
        cleaned = re.sub(r'^```\s*|\s*```$', '', cleaned, flags=re.MULTILINE | re.DOTALL)
        return cleaned.strip()
    
    def _empty_result(self) -> ExtractionResult:
        """Return empty but valid result to continue pipeline."""
        return ExtractionResult(
            entities=ExtractedEntities(
                symptoms=[],
                disorders=[],
                therapies=[],
                emotions=[],
                cognitive_patterns=[]
            ),
            relations=[]
        )
    
    def get_stats(self) -> dict:
        """Return extraction statistics."""
        return {
            'total_extractions': self._extraction_count,
            'total_errors': self._error_count,
            'error_rate': self._error_count / max(self._extraction_count + self._error_count, 1)
        }
