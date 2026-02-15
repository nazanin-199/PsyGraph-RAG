import networkx as nx


class TypedKnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_node(self, node):
        self.graph.add_node(
            node.node_id,
            label=node.label,
            type=node.node_type,
            metadata=node.metadata,
            embedding=node.embedding,
        )

    def add_relation(self, relation):
        self.graph.add_edge(
            relation.source,
            relation.target,
            type=relation.relation_type,
        )

    def get_neighbors(self, node_id):
        return list(self.graph.successors(node_id)) + \
               list(self.graph.predecessors(node_id))

    def get_node_data(self, node_id):
        return self.graph.nodes[node_id]
