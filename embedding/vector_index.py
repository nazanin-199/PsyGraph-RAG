"""
FAISS vector similarity search with validation.
"""
import faiss
import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Optional
import logging
import pickle
from pathlib import Path

from exceptions import VectorIndexError

logger = logging.getLogger(__name__)


class FaissIndex:
    """
    FAISS-based vector similarity search with safety checks.
    
    Features:
    - Dimension validation
    - Zero vector detection
    - Normalized cosine similarity (inner product on unit vectors)
    - GPU support (optional)
    - Save/load functionality
    
    Example:
        >>> index = FaissIndex(1536)
        >>> index.add("node_1", np.random.rand(1536).astype('float32'))
        >>> results = index.search(query_vector, top_k=5)
        >>> [(node_id, score), ...]
    """
    
    def __init__(self, embedding_dim: int, use_gpu: bool = False):
        """
        Initialize FAISS index.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            use_gpu: If True and GPU available, use GPU acceleration
            
        Raises:
            ValueError: If embedding_dim is invalid
        """
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        
        self.embedding_dim = embedding_dim
        
        # Create inner product index (for cosine similarity on normalized vectors)
        cpu_index = faiss.IndexFlatIP(embedding_dim)
        
        # Move to GPU if requested and available
        if use_gpu and faiss.get_num_gpus() > 0:
            try:
                self.index = faiss.index_cpu_to_all_gpus(cpu_index)
                logger.info(f"Initialized FAISS index on {faiss.get_num_gpus()} GPU(s)")
            except Exception as e:
                logger.warning(f"GPU initialization failed, using CPU: {e}")
                self.index = cpu_index
        else:
            self.index = cpu_index
        
        self.node_ids: List[str] = []
        self._add_count = 0
    
    def add(self, node_id: str, embedding: npt.NDArray[np.float32]) -> None:
        """
        Add normalized embedding to index.
        
        Args:
            node_id: Unique identifier for the vector
            embedding: Embedding vector (will be normalized)
            
        Raises:
            ValueError: If embedding is invalid
            VectorIndexError: If FAISS operation fails
        """
        if not node_id:
            raise ValueError("node_id cannot be empty")
        
        # Convert to float32 if needed
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        
        # Validate dimension
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embedding.shape[0]}"
            )
        
        # Check for zero vector
        norm = np.linalg.norm(embedding)
        if norm < 1e-9:
            raise ValueError(
                f"Cannot add zero vector for node '{node_id}'. "
                f"Norm: {norm:.2e}"
            )
        
        # Normalize to unit length for cosine similarity
        normalized = embedding / norm
        
        # Validate normalization (should be ~1.0)
        new_norm = np.linalg.norm(normalized)
        if not (0.99 < new_norm < 1.01):
            logger.warning(
                f"Normalization issue for node '{node_id}': "
                f"norm after normalization = {new_norm}"
            )
        
        try:
            # Add to FAISS index (expects 2D array)
            self.index.add(normalized.reshape(1, -1))
            self.node_ids.append(node_id)
            self._add_count += 1
            
            # Log progress every 1000 additions
            if self._add_count % 1000 == 0:
                logger.info(f"Added {self._add_count} vectors to index")
                
        except Exception as e:
            logger.error(f"FAISS add operation failed for node '{node_id}': {e}")
            raise VectorIndexError(f"Failed to add vector to index: {e}") from e
    
    def search(
        self, 
        query_embedding: npt.NDArray[np.float32], 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search for most similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of (node_id, similarity_score) tuples, sorted by score descending
            
        Raises:
            ValueError: If query is invalid
            VectorIndexError: If search fails
        """
        if len(self.node_ids) == 0:
            logger.warning("Search called on empty index")
            return []
        
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        
        # Limit top_k to index size
        top_k = min(top_k, len(self.node_ids))
        
        # Convert to float32
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        # Validate dimension
        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query dimension mismatch: expected {self.embedding_dim}, "
                f"got {query_embedding.shape[0]}"
            )
        
        # Normalize query
        norm = np.linalg.norm(query_embedding)
        if norm < 1e-9:
            raise ValueError("Cannot search with zero query vector")
        
        normalized = query_embedding / norm
        
        try:
            # Search (expects 2D array)
            scores, indices = self.index.search(normalized.reshape(1, -1), top_k)
            
            # Build results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                # FAISS returns -1 for invalid indices
                if idx < 0 or idx >= len(self.node_ids):
                    logger.warning(f"Invalid index {idx} returned by FAISS")
                    continue
                
                results.append((self.node_ids[idx], float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            raise VectorIndexError(f"Search operation failed: {e}") from e
    
    def save(self, filepath: str) -> None:
        """
        Save index to disk.
        
        Args:
            filepath: Path to save index (without extension)
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.index")
            
            # Save node IDs and metadata
            metadata = {
                'node_ids': self.node_ids,
                'embedding_dim': self.embedding_dim,
                'add_count': self._add_count
            }
            
            with open(f"{filepath}.meta", 'wb') as f:
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(
                f"Saved FAISS index with {len(self.node_ids)} vectors to {filepath}"
            )
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise VectorIndexError(f"Save operation failed: {e}") from e
    
    def load(self, filepath: str) -> None:
        """
        Load index from disk.
        
        Args:
            filepath: Path to load index from (without extension)
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.index")
            
            # Load metadata
            with open(f"{filepath}.meta", 'rb') as f:
                metadata = pickle.load(f)
            
            self.node_ids = metadata['node_ids']
            self.embedding_dim = metadata['embedding_dim']
            self._add_count = metadata['add_count']
            
            # Validate consistency
            if self.index.ntotal != len(self.node_ids):
                raise VectorIndexError(
                    f"Index size mismatch: FAISS has {self.index.ntotal} vectors "
                    f"but node_ids has {len(self.node_ids)} entries"
                )
            
            logger.info(
                f"Loaded FAISS index with {len(self.node_ids)} vectors from {filepath}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise VectorIndexError(f"Load operation failed: {e}") from e
    
    def get_stats(self) -> dict:
        """Return index statistics for monitoring."""
        return {
            'dimension': self.embedding_dim,
            'num_vectors': len(self.node_ids),
            'index_type': type(self.index).__name__,
            'is_trained': self.index.is_trained,
            'total_adds': self._add_count
        }
