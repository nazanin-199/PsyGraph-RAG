class RetrievalEvaluator:

    def __init__(self):
        pass

    def hit_at_k(self, gold_nodes, retrieved_nodes):
        gold_set = set(gold_nodes)
        retrieved_set = set(retrieved_nodes)

        hits = len(gold_set & retrieved_set)
        return hits / len(gold_set) if len(gold_set) > 0 else 0

    def subgraph_coverage(self, gold_nodes, subgraph_nodes):
        gold_set = set(gold_nodes)
        subgraph_set = set(subgraph_nodes)

        covered = len(gold_set & subgraph_set)
        return covered / len(gold_set) if len(gold_set) > 0 else 0
