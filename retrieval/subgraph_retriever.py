"""
Subgraph extraction utilities.
"""
from typing import List, Set
import logging

logger = logging.getLogger(__name__)


class SubgraphRetriever:
    """
    Retrieves subgraphs from the knowledge graph.
    """
    
    def __init__(self, graph_store, max_hops: int):
        """
        Initialize subgraph retriever.
        
        Args:
            graph_store: Knowledge graph
            max_hops: Maximum traversal depth
        """
        self.graph = graph_store
        self.max_hops = max_hops
    
    def retrieve(self, seed_nodes: List[str]) -> List[str]:
        """
        Retrieve subgraph starting from seed nodes.
        
        Args:
            seed_nodes: Starting nodes
            
        Returns:
            List of all nodes in subgraph
        """
        visited: Set[str] = set(seed_nodes)
        frontier: Set[str] = set(seed_nodes)
        
        for _ in range(self.max_hops):
            next_frontier: Set[str] = set()
            
            for node_id in frontier:
                try:
                    neighbors = self.graph.get_neighbors(node_id)
                    for neighbor_id in neighbors:
                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            next_frontier.add(neighbor_id)
                except Exception as e:
                    logger.warning(f"Failed to get neighbors for {node_id}: {e}")
                    continue
            
            frontier = next_frontier
        
        return list(visited)
