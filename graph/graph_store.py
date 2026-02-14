from collections import defaultdict
from .node import Node
from .edge import Edge


class GraphStore:

    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.adjacency = defaultdict(list)

    def add_node(self, node: Node):
        if node.id not in self.nodes:
            self.nodes[node.id] = node

    def add_edge(self, edge: Edge):
        self.edges.append(edge)
        self.adjacency[edge.source].append(edge)
        self.nodes[edge.source].degree += 1
        self.nodes[edge.target].degree += 1

    def get_neighbors(self, node_id: str):
        return self.adjacency.get(node_id, [])

    def get_node(self, node_id: str):
        return self.nodes.get(node_id)
