from config.settings import Settings
from services.llm_client import LLMClient
from extraction.structured_extractor import StructuredExtractor
from graph.typed_graph import TypedKnowledgeGraph
from embedding.node_embedder import NodeEmbedder
from embedding.vector_index import FaissIndex
from retrieval.graph_retriever import GraphRetriever
from generation.graph_reasoner import GraphReasoner
from models.graph_models import GraphNode, GraphRelation


def main():
    settings = Settings()
    settings.validate()

    llm = LLMClient(settings.openrouter_api_key, settings.base_url)
    extractor = StructuredExtractor(llm, settings.extraction_model)

    graph = TypedKnowledgeGraph()
    embedder = NodeEmbedder(settings.embedding_model_name)

    sample_text = "Low self-esteem caused by family pressure and academic failure."

    structured = extractor.extract(sample_text)

    embedding_dim = None

    for entity in structured["entities"]:
        embedding = embedder.embed(entity["label"])

        if embedding_dim is None:
            embedding_dim = len(embedding)
            vector_index = FaissIndex(embedding_dim)

        node = GraphNode(
            node_id=entity["id"],
            label=entity["label"],
            node_type=entity["type"],
            embedding=embedding,
        )

        graph.add_node(node)
        vector_index.add(entity["id"], embedding)

    for rel in structured["relations"]:
        graph.add_relation(
            GraphRelation(
                source=rel["source"],
                target=rel["target"],
                relation_type=rel["type"],
            )
        )

    retriever = GraphRetriever(graph, vector_index)
    reasoner = GraphReasoner(llm, settings.reasoning_model)

    query = "How can I improve my self-esteem?"
    query_embedding = embedder.embed(query)

    seed_nodes = retriever.retrieve_seed_nodes(query_embedding, top_k=3)
    expanded_nodes = retriever.multi_hop_expand(seed_nodes, hops=2)
    subgraph_text = retriever.build_subgraph_text(expanded_nodes)

    answer = reasoner.generate_answer(query, subgraph_text)

    print(answer)


if __name__ == "__main__":
    main()
