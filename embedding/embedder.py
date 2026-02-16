"""
Enhanced embedding module with error handling, retries, and validation.
"""
from typing import List, Optional
import time
import logging
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
import numpy as np

from exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class Embedder:
    """
    Generates text embeddings with robust error handling.
    
    Features:
    - Exponential backoff for rate limits
    - Dimension validation
    - Batch processing support
    - Comprehensive error handling
    
    Example:
        >>> embedder = Embedder("text-embedding-3-small", api_key="sk-...")
        >>> embedding = embedder.embed("anxiety")
        >>> len(embedding)
        1536
    """
    
    def __init__(
        self, 
        model_name: str, 
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize embedder with configuration.
        
        Args:
            model_name: OpenAI embedding model name
            api_key: OpenAI API key (defaults to env var)
            max_retries: Number of retry attempts for failed requests
            timeout: Request timeout in seconds
        """
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1" )
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        self._expected_dim: Optional[int] = None
        self._request_count = 0
        self._error_count = 0
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            ValueError: If text is empty or dimension mismatch occurs
            EmbeddingError: If API fails after max retries
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty or whitespace-only text")
        
        # Truncate to model's max length (OpenAI: 8191 tokens ≈ 32k chars)
        text = text.strip()[:32000]
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=text,
                    timeout=self.timeout
                )
                
                embedding = response.data[0].embedding
                self._request_count += 1
                
                # Validate dimension consistency
                if self._expected_dim is None:
                    self._expected_dim = len(embedding)
                    logger.info(f"Initialized embedding dimension: {self._expected_dim}")
                elif len(embedding) != self._expected_dim:
                    error_msg = (
                        f"Embedding dimension mismatch: expected {self._expected_dim}, "
                        f"got {len(embedding)}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                return embedding
                
            except RateLimitError as e:
                self._error_count += 1
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + (0.1 * attempt)
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}). "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise EmbeddingError(
                        f"Rate limit exceeded after {self.max_retries} attempts: {e}"
                    ) from e
                    
            except APIConnectionError as e:
                self._error_count += 1
                if attempt < self.max_retries - 1:
                    wait_time = 1.0 * (attempt + 1)
                    logger.warning(
                        f"Connection error (attempt {attempt + 1}/{self.max_retries}). "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise EmbeddingError(
                        f"Connection failed after {self.max_retries} attempts: {e}"
                    ) from e
                    
            except APIError as e:
                self._error_count += 1
                logger.error(f"OpenAI API error: {e}")
                raise EmbeddingError(f"Embedding generation failed: {e}") from e
                
            except Exception as e:
                self._error_count += 1
                logger.error(f"Unexpected error during embedding: {e}")
                raise EmbeddingError(f"Unexpected embedding error: {e}") from e
    
    def embed_batch(
        self, 
        texts: List[str], 
        batch_size: int = 100,
        skip_errors: bool = True
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        OpenAI allows up to 2048 inputs per request. We batch for better performance.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts per API call (max 2048)
            skip_errors: If True, return None for failed embeddings; if False, raise
            
        Returns:
            List of embeddings (same length as input). Failed items are None if skip_errors=True
            
        Raises:
            EmbeddingError: If skip_errors=False and any embedding fails
        """
        if not texts:
            return []
        
        if batch_size > 2048:
            logger.warning(f"Batch size {batch_size} exceeds OpenAI limit (2048). Using 2048.")
            batch_size = 2048
        
        all_embeddings: List[Optional[List[float]]] = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Clean and truncate texts
            batch = [t.strip()[:32000] if t and t.strip() else "" for t in batch]
            
            # Filter out empty strings but track indices
            valid_indices = [idx for idx, t in enumerate(batch) if t]
            valid_texts = [batch[idx] for idx in valid_indices]
            
            if not valid_texts:
                all_embeddings.extend([None] * len(batch))
                continue
            
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=valid_texts,
                    timeout=self.timeout
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                self._request_count += 1
                
                # Validate dimensions
                for emb in batch_embeddings:
                    if self._expected_dim is None:
                        self._expected_dim = len(emb)
                    elif len(emb) != self._expected_dim:
                        raise ValueError(
                            f"Dimension mismatch in batch: expected {self._expected_dim}, "
                            f"got {len(emb)}"
                        )
                
                # Map results back to original indices
                result_batch: List[Optional[List[float]]] = [None] * len(batch)
                for valid_idx, embedding in zip(valid_indices, batch_embeddings):
                    result_batch[valid_idx] = embedding
                
                all_embeddings.extend(result_batch)
                
            except Exception as e:
                self._error_count += 1
                logger.error(f"Batch embedding failed for batch {i//batch_size + 1}: {e}")
                
                if skip_errors:
                    all_embeddings.extend([None] * len(batch))
                else:
                    raise EmbeddingError(f"Batch embedding failed: {e}") from e
        
        return all_embeddings
    
    def get_stats(self) -> dict:
        """Return embedding statistics for monitoring."""
        return {
            'total_requests': self._request_count,
            'total_errors': self._error_count,
            'error_rate': self._error_count / max(self._request_count, 1),
            'expected_dimension': self._expected_dim
        }
