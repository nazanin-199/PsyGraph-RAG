class SubgraphRetriever:

    def __init__(self, graph_store, max_hops: int):
        self.graph = graph_store
        self.max_hops = max_hops

    def retrieve(self, seed_nodes: list):
        visited = set(seed_nodes)
        frontier = set(seed_nodes)

        for _ in range(self.max_hops):
            next_frontier = set()
            for node_id in frontier:
                neighbors = self.graph.get_neighbors(node_id)
                for edge in neighbors:
                    if edge.target not in visited:
                        visited.add(edge.target)
                        next_frontier.add(edge.target)
            frontier = next_frontier

        return list(visited)
