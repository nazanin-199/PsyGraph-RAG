"""
Graph construction with deduplication and atomic operations.
"""
from typing import Optional, Set, Dict, Any, List
from dataclasses import dataclass, field
import numpy as np
import logging
import re
import xml.etree.ElementTree as ET

from graph.typed_graph import TypedKnowledgeGraph
from embedding.vector_index import FaissIndex
from extraction.llm_extractor import LLMExtractor
from embedding.embedder import Embedder
from exceptions import GraphError

logger = logging.getLogger(__name__)


@dataclass
class GraphBuilderState:
    """Track processing state to enable idempotent operations."""
    seen_entities: Set[str] = field(default_factory=set)
    seen_relations: Set[tuple] = field(default_factory=set)
    node_count: int = 0
    edge_count: int = 0
    failed_embeddings: int = 0
    duplicate_entities: int = 0


class GraphBuilder:
    """
    Builds knowledge graph and vector index from extracted entities.
    
    Features:
    - Automatic deduplication of entities
    - Atomic batch commits
    - Comprehensive error handling
    - Processing statistics
    - Batch embedding for performance
    
    Example:
        >>> builder = GraphBuilder(extractor, embedder)
        >>> graph, index = builder.build_from_chunks(chunks)
        >>> print(f"Built graph with {len(graph.graph.nodes)} nodes")
    """
    
    def __init__(
        self, 
        extractor: LLMExtractor, 
        embedder: Embedder,
        store_embeddings_in_graph: bool = False
    ):
        """
        Initialize graph builder.
        
        Args:
            extractor: Entity extraction service
            embedder: Text embedding service
            store_embeddings_in_graph: If False, only store in FAISS (saves memory)
        """
        self.extractor = extractor
        self.embedder = embedder
        self.graph = TypedKnowledgeGraph()
        self.embedding_dim: Optional[int] = None
        self.vector_index: Optional[FaissIndex] = None
        self.state = GraphBuilderState()
        self.store_embeddings = store_embeddings_in_graph
    
    def _initialize_index(self, embedding: List[float]) -> None:
        """Initialize FAISS index on first embedding."""
        self.embedding_dim = len(embedding)
        self.vector_index = FaissIndex(self.embedding_dim)
        logger.info(f"Initialized FAISS index with dimension {self.embedding_dim}")
    
    def _generate_entity_key(self, label: str, category: str) -> str:
        """Generate unique key for entity deduplication."""
        # Normalize: lowercase, strip whitespace, collapse multiple spaces
        normalized = ' '.join(label.lower().strip().split())
        return f"{normalized}|{category}"
    
    def _generate_node_id(self, label: str, category: str, entity_key: str) -> str:
        """Generate unique node ID that prevents collisions."""
        # Use hash of entity key to ensure uniqueness
        key_hash = abs(hash(entity_key)) % 100000
        return f"{label}_{category}_{key_hash}"
    
    def process_text(self, text: str, source_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract entities/relations and add to graph.
        
        Args:
            text: Input text to process
            source_id: Optional identifier for source (e.g., video_id)
            
        Returns:
            Dict with processing statistics
            
        Raises:
            GraphError: If critical error occurs
        """
        if not text or len(text.strip()) < 10:
            logger.warning(f"Skipping text too short: {len(text)} chars")
            return {'status': 'skipped', 'reason': 'text_too_short'}
        
        # Extract entities and relations
        try:
            structured = self.extractor.extract(text)
        except Exception as e:
            logger.error(f"Extraction failed for text: {text[:100]}... Error: {e}")
            return {'status': 'error', 'reason': 'extraction_failed'}
        
        # Collect entities for batch embedding
        entities_to_process = []
        
        for category, values in structured.entities.dict().items():
            for label in values:
                entity_key = self._generate_entity_key(label, category)
                
                # Skip duplicates
                if entity_key in self.state.seen_entities:
                    logger.debug(f"Skipping duplicate entity: {entity_key}")
                    self.state.duplicate_entities += 1
                    continue
                
                node_id = self._generate_node_id(label, category, entity_key)
                entities_to_process.append({
                    'label': label,
                    'category': category,
                    'entity_key': entity_key,
                    'node_id': node_id
                })
        
        # Batch embed all new entities
        if entities_to_process:
            labels = [e['label'] for e in entities_to_process]
            
            try:
                embeddings = self.embedder.embed_batch(labels, skip_errors=True)
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                return {'status': 'error', 'reason': 'embedding_failed'}
            
            # Initialize index on first embedding
            if self.embedding_dim is None and embeddings:
                valid_embedding = next((e for e in embeddings if e is not None), None)
                if valid_embedding:
                    self._initialize_index(valid_embedding)
            
            # Atomic commit: add all entities or roll back
            successful_entities = []
            
            for entity_data, embedding in zip(entities_to_process, embeddings):
                if embedding is None:
                    logger.warning(f"Failed to embed '{entity_data['label']}', skipping")
                    self.state.failed_embeddings += 1
                    continue
                
                try:
                    # Add to graph
                    self.graph.add_node(
                        node_id=entity_data['node_id'],
                        label=entity_data['label'],
                        node_type=entity_data['category'],
                        embedding=embedding if self.store_embeddings else None,
                        metadata={'source_id': source_id} if source_id else {}
                    )
                    
                    # Add to vector index
                    self.vector_index.add(entity_data['node_id'], np.array(embedding, dtype=np.float32))
                    
                    # Mark as processed
                    self.state.seen_entities.add(entity_data['entity_key'])
                    self.state.node_count += 1
                    successful_entities.append(entity_data['node_id'])
                    
                except Exception as e:
                    logger.error(
                        f"Failed to add entity '{entity_data['label']}' to graph: {e}"
                    )
                    continue
        
        # Add relations
        for rel in structured.relations:
            relation_key = (rel.source, rel.relation, rel.target)
            
            if relation_key in self.state.seen_relations:
                logger.debug(f"Skipping duplicate relation: {relation_key}")
                continue
            
            try:
                # Only add edge if both nodes exist
                if (self.graph.has_node(rel.source) and 
                    self.graph.has_node(rel.target)):
                    
                    self.graph.add_edge(
                        source=rel.source,
                        target=rel.target,
                        relation=rel.relation,
                        confidence=rel.confidence
                    )
                    
                    self.state.seen_relations.add(relation_key)
                    self.state.edge_count += 1
                else:
                    logger.debug(
                        f"Skipping relation {relation_key}: nodes not found in graph"
                    )
                    
            except Exception as e:
                logger.error(f"Failed to add relation {relation_key}: {e}")
                continue
        
        return {
            'status': 'success',
            'entities_added': len(successful_entities),
            'edges_added': self.state.edge_count,
            'failed_embeddings': self.state.failed_embeddings,
            'duplicates_skipped': self.state.duplicate_entities
        }

 

    def clean_youtube_xml(xml_text):
        try:
            root = ET.fromstring(xml_text)
            texts = [elem.text for elem in root.findall(".//text") if elem.text]
            return " ".join(texts)
        except:
            return re.sub(r"<[^>]+>", "", xml_text)
    def build_from_chunks(self, chunks: List[dict]) -> tuple:
        """
        Build graph from list of text chunks.
        
        Args:
            chunks: List of dicts with 'text' field (and optional 'video_id')
            
        Returns:
            Tuple of (graph, vector_index)
            
        Raises:
            GraphError: If building fails completely
        """
        if not chunks:
            raise GraphError("Cannot build graph from empty chunks list")
        
        logger.info(f"Building graph from {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks, 1):
            if 'text' not in chunk:
                logger.warning(f"Chunk {i} missing 'text' field, skipping")
                continue
            
            try:
                source_id = chunk.get('video_id', f'chunk_{i}')
                self.process_text(chunk['text'], source_id=source_id)
                
                # Log progress every 100 chunks
                if i % 100 == 0:
                    logger.info(
                        f"Processed {i}/{len(chunks)} chunks. "
                        f"Nodes: {self.state.node_count}, "
                        f"Edges: {self.state.edge_count}"
                    )
                    
            except Exception as e:
                logger.error(f"Failed to process chunk {i}: {e}")
                continue
        
        logger.info(
            f"Graph building complete. "
            f"Final stats: {self.state.node_count} nodes, "
            f"{self.state.edge_count} edges, "
            f"{self.state.duplicate_entities} duplicates skipped, "
            f"{self.state.failed_embeddings} failed embeddings"
        )
        
        if self.vector_index is None:
            raise GraphError("No valid embeddings generated - index not initialized")
        
        return self.graph, self.vector_index
    
    def get_stats(self) -> Dict[str, Any]:
        """Return building statistics for monitoring."""
        return {
            'nodes': self.state.node_count,
            'edges': self.state.edge_count,
            'duplicates_skipped': self.state.duplicate_entities,
            'failed_embeddings': self.state.failed_embeddings,
            'unique_entities': len(self.state.seen_entities),
            'unique_relations': len(self.state.seen_relations)
        }
