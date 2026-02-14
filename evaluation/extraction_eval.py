from sklearn.metrics import precision_score, recall_score, f1_score


class ExtractionEvaluator:

    def __init__(self):
        pass

    def evaluate_entities(self, gold_entities, predicted_entities):
        gold_set = set(gold_entities)
        pred_set = set(predicted_entities)

        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def evaluate_relations(self, gold_relations, predicted_relations):
        gold_set = set((r["source"], r["relation"], r["target"]) for r in gold_relations)
        pred_set = set((r.source, r.relation, r.target) for r in predicted_relations)

        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
