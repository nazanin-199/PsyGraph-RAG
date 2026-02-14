import networkx as nx


class GraphEvaluator:

    def __init__(self, graph_store):
        self.graph_store = graph_store

    def build_networkx_graph(self):
        G = nx.DiGraph()

        for node_id, node in self.graph_store.nodes.items():
            G.add_node(node_id, label=node.label, type=node.type)

        for edge in self.graph_store.edges:
            G.add_edge(edge.source, edge.target, relation=edge.relation)

        return G

    def compute_metrics(self):
        G = self.build_networkx_graph()

        avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
        density = nx.density(G)

        return {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "avg_degree": avg_degree,
            "density": density
        }
