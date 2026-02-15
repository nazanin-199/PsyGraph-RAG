"""
Graph-based retrieval with multi-hop expansion.
"""
from typing import List, Tuple, Set
import numpy as np
import numpy.typing as npt
import logging

from graph.typed_graph import TypedKnowledgeGraph
from embedding.vector_index import FaissIndex
from exceptions import RetrievalError

logger = logging.getLogger(__name__)


class GraphRetriever:
    """
    Retrieves relevant subgraphs using vector similarity and multi-hop expansion.
    
    Features:
    - Vector similarity search for seed nodes
    - Multi-hop graph traversal
    - Subgraph text generation
    """
    
    def __init__(self, graph: TypedKnowledgeGraph, vector_index: FaissIndex):
        """
        Initialize retriever.
        
        Args:
            graph: Knowledge graph
            vector_index: Vector similarity index
        """
        self.graph = graph
        self.index = vector_index
    
    def retrieve_seed_nodes(
        self,
        query_embedding: npt.NDArray[np.float32],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most similar nodes to query.
        
        Args:
            query_embedding: Query vector
            top_k: Number of seed nodes to return
            
        Returns:
            List of (node_id, similarity_score) tuples
            
        Raises:
            RetrievalError: If search fails
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        try:
            results = self.index.search(query_embedding, top_k)
            logger.info(f"Retrieved {len(results)} seed nodes")
            return results
        except Exception as e:
            logger.error(f"Seed node retrieval failed: {e}")
            raise RetrievalError(f"Failed to retrieve seed nodes: {e}") from e
    
    def multi_hop_expand(
        self,
        seed_nodes: List[Tuple[str, float]],
        hops: int = 2
    ) -> List[str]:
        """
        Expand seed nodes via multi-hop graph traversal.
        
        Args:
            seed_nodes: List of (node_id, score) tuples
            hops: Number of expansion steps (1-5)
            
        Returns:
            List of node IDs in expanded subgraph
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not seed_nodes:
            logger.warning("Empty seed nodes, returning empty result")
            return []
        
        if not 1 <= hops <= 5:
            raise ValueError(f"hops must be 1-5, got {hops}")
        
        visited: Set[str] = set()
        frontier = {node_id for node_id, _ in seed_nodes}
        
        for hop in range(hops):
            next_frontier: Set[str] = set()
            
            for node_id in frontier:
                if node_id in visited:
                    continue
                
                visited.add(node_id)
                
                try:
                    neighbors = self.graph.get_neighbors(node_id)
                    next_frontier.update(neighbors)
                except Exception as e:
                    logger.warning(f"Failed to get neighbors for {node_id}: {e}")
                    continue
            
            frontier = next_frontier
            logger.debug(f"Hop {hop + 1}: {len(visited)} nodes visited, {len(frontier)} in frontier")
        
        logger.info(f"Multi-hop expansion: {len(visited)} total nodes after {hops} hops")
        return list(visited)
    
    def build_subgraph_text(self, node_ids: List[str]) -> str:
        """
        Build text representation of subgraph.
        
        Args:
            node_ids: List of node IDs to include
            
        Returns:
            Text representation of subgraph
        """
        if not node_ids:
            return "No relevant knowledge found."
        
        lines = []
        
        for node_id in node_ids:
            try:
                node_data = self.graph.get_node_data(node_id)
                lines.append(
                    f"Node: {node_data['label']} (Type: {node_data['type']})"
                )
                
                # Get neighbors and their relations
                neighbors = self.graph.get_neighbors(node_id)
                if neighbors:
                    for neighbor_id in neighbors[:5]:  # Limit to avoid too much text
                        try:
                            neighbor_data = self.graph.get_node_data(neighbor_id)
                            lines.append(
                                f"  → {neighbor_data['label']} ({neighbor_data['type']})"
                            )
                        except Exception as e:
                            logger.debug(f"Failed to get neighbor data: {e}")
                            continue
            
            except Exception as e:
                logger.warning(f"Failed to get data for node {node_id}: {e}")
                continue
        
        text = "\n".join(lines)
        logger.info(f"Built subgraph text: {len(text)} characters, {len(lines)} lines")
        return text
