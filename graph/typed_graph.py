"""
Type-safe knowledge graph with consistent API.
"""
import networkx as nx
from typing import List, Dict, Any, Optional
import logging
import pickle
from pathlib import Path

from exceptions import GraphError

logger = logging.getLogger(__name__)


class TypedKnowledgeGraph:
    """
    Knowledge graph with typed nodes and relations.
    
    Features:
    - Consistent API (no direct NetworkX access)
    - Type safety and validation
    - Relation metadata support
    - Subgraph extraction
    - Save/load functionality
    
    Example:
        >>> graph = TypedKnowledgeGraph()
        >>> graph.add_node("node_1", "Anxiety", "symptom", embedding=[...])
        >>> graph.add_edge("node_1", "node_2", "causes", confidence=0.9)
        >>> neighbors = graph.get_neighbors("node_1")
    """
    
    def __init__(self):
        """Initialize empty knowledge graph."""
        self.graph = nx.MultiDiGraph()
        self._node_count = 0
        self._edge_count = 0
    
    def add_node(
        self, 
        node_id: str, 
        label: str, 
        node_type: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add node to graph with type and attributes.
        
        Args:
            node_id: Unique node identifier
            label: Human-readable label
            node_type: Node category (e.g., "symptom", "disorder", "therapy")
            embedding: Optional embedding vector
            metadata: Optional additional attributes
            
        Raises:
            ValueError: If required fields are missing
        """
        if not node_id or not label or not node_type:
            raise ValueError("node_id, label, and node_type are required")
        
        # Check if node already exists
        if self.has_node(node_id):
            logger.debug(f"Node {node_id} already exists, updating attributes")
        else:
            self._node_count += 1
        
        self.graph.add_node(
            node_id,
            label=label,
            type=node_type,
            embedding=embedding,
            metadata=metadata or {},
            degree=self.graph.degree(node_id) if self.has_node(node_id) else 0
        )
    
    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add directed edge with relation type.
        
        Args:
            source: Source node ID
            target: Target node ID
            relation: Relation type (e.g., "causes", "treats", "related_to")
            confidence: Confidence score (0.0 to 1.0)
            metadata: Optional additional attributes
            
        Raises:
            GraphError: If nodes don't exist
        """
        if not self.has_node(source):
            raise GraphError(f"Source node '{source}' not found in graph")
        
        if not self.has_node(target):
            raise GraphError(f"Target node '{target}' not found in graph")
        
        if not 0.0 <= confidence <= 1.0:
            logger.warning(f"Confidence {confidence} outside [0,1], clipping")
            confidence = max(0.0, min(1.0, confidence))
        
        self.graph.add_edge(
            source,
            target,
            relation=relation,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        # Update node degrees
        self.graph.nodes[source]['degree'] = self.graph.degree(source)
        self.graph.nodes[target]['degree'] = self.graph.degree(target)
        
        self._edge_count += 1
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists in graph."""
        return self.graph.has_node(node_id)
    
    def get_node_data(self, node_id: str) -> Dict[str, Any]:
        """
        Get node attributes.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Dict of node attributes
            
        Raises:
            GraphError: If node doesn't exist
        """
        if not self.has_node(node_id):
            raise GraphError(f"Node '{node_id}' not found")
        
        return dict(self.graph.nodes[node_id])
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """
        Get all neighbors (successors and predecessors).
        
        Args:
            node_id: Node identifier
            
        Returns:
            List of neighbor node IDs
            
        Raises:
            GraphError: If node doesn't exist
        """
        if not self.has_node(node_id):
            raise GraphError(f"Node '{node_id}' not found")
        
        # Get both outgoing and incoming neighbors
        successors = set(self.graph.successors(node_id))
        predecessors = set(self.graph.predecessors(node_id))
        
        return list(successors | predecessors)
    
    def get_edges(self, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get edges, optionally filtered by source node.
        
        Args:
            source: If provided, only return edges from this node
            
        Returns:
            List of edge dicts with source, target, relation, confidence
        """
        edges = []
        
        if source:
            if not self.has_node(source):
                raise GraphError(f"Node '{source}' not found")
            edge_iter = self.graph.out_edges(source, data=True)
        else:
            edge_iter = self.graph.edges(data=True)
        
        for src, tgt, data in edge_iter:
            edges.append({
                'source': src,
                'target': tgt,
                'relation': data.get('relation', 'unknown'),
                'confidence': data.get('confidence', 1.0),
                'metadata': data.get('metadata', {})
            })
        
        return edges
    
    def get_subgraph(self, node_ids: List[str]) -> 'TypedKnowledgeGraph':
        """
        Extract subgraph containing specified nodes.
        
        Args:
            node_ids: List of node IDs to include
            
        Returns:
            New TypedKnowledgeGraph with only specified nodes
        """
        subgraph = TypedKnowledgeGraph()
        
        # Add nodes
        for node_id in node_ids:
            if self.has_node(node_id):
                node_data = self.get_node_data(node_id)
                subgraph.add_node(
                    node_id=node_id,
                    label=node_data['label'],
                    node_type=node_data['type'],
                    embedding=node_data.get('embedding'),
                    metadata=node_data.get('metadata', {})
                )
        
        # Add edges between included nodes
        for edge in self.get_edges():
            if edge['source'] in node_ids and edge['target'] in node_ids:
                subgraph.add_edge(
                    source=edge['source'],
                    target=edge['target'],
                    relation=edge['relation'],
                    confidence=edge['confidence'],
                    metadata=edge.get('metadata', {})
                )
        
        return subgraph
    
    def save(self, filepath: str) -> None:
        """Save graph to disk using pickle."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {
                'graph': self.graph,
                'node_count': self._node_count,
                'edge_count': self._edge_count
            }
            
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(
                f"Saved graph with {len(self.graph.nodes)} nodes "
                f"and {len(self.graph.edges)} edges to {filepath}"
            )
        except Exception as e:
            raise GraphError(f"Failed to save graph: {e}") from e
    
    def load(self, filepath: str) -> None:
        """Load graph from disk."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.graph = data['graph']
            self._node_count = data.get('node_count', len(self.graph.nodes))
            self._edge_count = data.get('edge_count', len(self.graph.edges))
            
            logger.info(
                f"Loaded graph with {self._node_count} nodes "
                f"and {self._edge_count} edges from {filepath}"
            )
        except Exception as e:
            raise GraphError(f"Failed to load graph: {e}") from e
    
    def get_stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        return {
            'num_nodes': len(self.graph.nodes),
            'num_edges': len(self.graph.edges),
            'node_types': self._get_node_type_counts(),
            'relation_types': self._get_relation_type_counts(),
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(len(self.graph.nodes), 1),
            'is_directed': self.graph.is_directed(),
            'is_multigraph': self.graph.is_multigraph()
        }
    
    def _get_node_type_counts(self) -> Dict[str, int]:
        """Count nodes by type."""
        type_counts = {}
        for _, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts
    
    def _get_relation_type_counts(self) -> Dict[str, int]:
        """Count edges by relation type."""
        rel_counts = {}
        for _, _, data in self.graph.edges(data=True):
            relation = data.get('relation', 'unknown')
            rel_counts[relation] = rel_counts.get(relation, 0) + 1
        return rel_counts
