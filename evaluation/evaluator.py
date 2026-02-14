class PsyGraphEvaluator:

    def __init__(self, graph_store):
        self.graph_store = graph_store

    def evaluate_graph(self):
        from .graph_eval import GraphEvaluator
        return GraphEvaluator(self.graph_store).compute_metrics()

    def evaluate_extraction(self, gold_entities, pred_entities):
        from .extraction_eval import ExtractionEvaluator
        return ExtractionEvaluator().evaluate_entities(gold_entities, pred_entities)

    def evaluate_retrieval(self, gold_nodes, retrieved_nodes):
        from .retrieval_eval import RetrievalEvaluator
        return RetrievalEvaluator().hit_at_k(gold_nodes, retrieved_nodes)
