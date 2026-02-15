import numpy as np
from graph.typed_graph import TypedKnowledgeGraph
from embedding.vector_index import FaissIndex


class GraphBuilder:
    """
    Builds a knowledge graph and FAISS index from extracted text chunks.
    """

    def __init__(self, extractor, embedder):
        self.extractor = extractor
        self.embedder = embedder
        self.graph = TypedKnowledgeGraph()
        self.embedding_dim = None
        self.vector_index = None

    def _initialize_index(self, embedding: list):
        self.embedding_dim = len(embedding)
        self.vector_index = FaissIndex(self.embedding_dim)

    def process_text(self, text: str):
        """
        Extract entities and relations from text,
        add them to the graph, and index embeddings.
        """

        structured = self.extractor.extract(text)

        # Add entities
        for category, values in structured.entities.dict().items():
            for label in values:

                embedding = self.embedder.embed(label)

                if self.embedding_dim is None:
                    self._initialize_index(embedding)

                node_id = f"{label}_{category}"

                self.graph.graph.add_node(
                    node_id,
                    label=label,
                    type=category,
                    embedding=embedding,
                )

                self.vector_index.add(node_id, np.array(embedding))

        # Add relations
        for rel in structured.relations:
            self.graph.graph.add_edge(
                rel.source,
                rel.target,
                relation=rel.relation,
            )

    def build_from_chunks(self, chunks: list):
        """
        Build graph from a list of chunk dictionaries.
        Each chunk must contain a 'text' field.
        """

        for chunk in chunks:
            self.process_text(chunk["text"])

        return self.graph, self.vector_index
