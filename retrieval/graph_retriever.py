class GraphRetriever:
    def __init__(self, graph, vector_index):
        self.graph = graph
        self.index = vector_index

    def retrieve_seed_nodes(self, query_embedding, top_k=5):
        return self.index.search(query_embedding, top_k)

    def multi_hop_expand(self, seed_nodes, hops=2):
        visited = set()
        frontier = {node_id for node_id, _ in seed_nodes}

        for _ in range(hops):
            next_frontier = set()
            for node_id in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)
                neighbors = self.graph.get_neighbors(node_id)
                next_frontier.update(neighbors)
            frontier = next_frontier

        return list(visited)

    def build_subgraph_text(self, node_ids):
        lines = []

        for node_id in node_ids:
            node = self.graph.get_node_data(node_id)
            lines.append(
                f"Node: {node['label']} ({node['type']})"
            )

            for neighbor in self.graph.get_neighbors(node_id):
                lines.append(
                    f"  Related to → {self.graph.get_node_data(neighbor)['label']}"
                )

        return "\n".join(lines)
