from sentence_transformers import SentenceTransformer
import numpy as np


class GenerationEvaluator:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def semantic_similarity(self, reference, prediction):
        emb_ref = self.model.encode(reference)
        emb_pred = self.model.encode(prediction)

        similarity = np.dot(emb_ref, emb_pred) / (
            np.linalg.norm(emb_ref) * np.linalg.norm(emb_pred)
        )

        return float(similarity)
