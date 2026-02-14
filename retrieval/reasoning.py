class ReasoningEngine:

    def __init__(self, graph_store):
        self.graph = graph_store

    def build_reasoning_chain(self, node_ids: list):
        chains = []

        for node_id in node_ids:
            edges = self.graph.get_neighbors(node_id)
            for edge in edges:
                chain = f"{edge.source} {edge.relation} {edge.target}"
                chains.append(chain)

        return chains
