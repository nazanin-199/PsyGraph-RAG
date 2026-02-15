"""
Main pipeline orchestration with error handling, logging, and caching.
"""
import logging
import logging.config
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import numpy as np

from config.settings import Settings
from dataset.dataset_loader import load_dataset
from dataset.graph_builder import GraphBuilder
from extraction.llm_extractor import LLMExtractor
from embedding.embedder import Embedder
from embedding.vector_index import FaissIndex
from graph.typed_graph import TypedKnowledgeGraph
from retrieval.graph_retriever import GraphRetriever
from generation.graph_reasoner import GraphReasoner
from exceptions import PipelineError


# Configure logging
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'detailed'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'pipeline.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'level': 'DEBUG',
            'formatter': 'detailed'
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console', 'file']
    }
}


def setup_logging(config: Settings) -> None:
    """Configure logging based on settings."""
    if config.log_file:
        LOGGING_CONFIG['handlers']['file']['filename'] = str(config.log_file)
    
    LOGGING_CONFIG['root']['level'] = config.log_level
    logging.config.dictConfig(LOGGING_CONFIG)


logger = logging.getLogger(__name__)


def simple_chunk_text(text: str, max_chunk_length: int = 2000, min_chunk_length: int = 500) -> list:
    """
    Simple chunking strategy: split by sentences into roughly equal chunks.
    
    Args:
        text: Input text
        max_chunk_length: Maximum characters per chunk
        min_chunk_length: Minimum characters per chunk
        
    Returns:
        List of chunk dicts with 'text' field
    """
    # Simple sentence splitting (can be improved with proper NLP)
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_length = len(sentence)
        
        if current_length + sentence_length > max_chunk_length and current_length >= min_chunk_length:
            # Start new chunk
            chunks.append({'text': ' '.join(current_chunk)})
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add last chunk
    if current_chunk and current_length >= min_chunk_length:
        chunks.append({'text': ' '.join(current_chunk)})
    
    return chunks


def run_full_pipeline(
    csv_path: str, 
    query: str,
    config: Optional[Settings] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Execute complete RAG pipeline with KG.
    
    Args:
        csv_path: Path to dataset CSV
        query: User query
        config: Optional configuration (defaults to environment)
        use_cache: Whether to use cached graph/index
        
    Returns:
        Dict with answer and metadata
        
    Raises:
        PipelineError: If pipeline fails
    """
    # Load configuration
    if config is None:
        try:
            config = Settings()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise PipelineError(f"Configuration error: {e}") from e
    
    config.validate()
    setup_logging(config)
    
    logger.info("="*80)
    logger.info("Starting KG-RAG Pipeline")
    logger.info(f"Dataset: {csv_path}")
    logger.info(f"Query: {query}")
    logger.info(f"Configuration: {config.dict(exclude={'openai_api_key'})}")
    logger.info("="*80)
    
    # Define cache paths
    graph_cache = config.cache_dir / "knowledge_graph.pkl"
    index_cache = config.cache_dir / "faiss_index"
    
    try:
        # Initialize API key
        api_key = config.openai_api_key.get_secret_value()
        
        # Initialize embedder (needed for both cache and no-cache paths)
        embedder = Embedder(
            model_name=config.embedding_model,
            api_key=api_key,
            max_retries=config.max_retries,
            timeout=config.timeout
        )
        
        # Check cache
        if use_cache and graph_cache.exists() and (Path(str(index_cache) + ".index").exists()):
            logger.info("="*80)
            logger.info("Loading from cache...")
            logger.info("="*80)
            
            try:
                graph = TypedKnowledgeGraph()
                graph.load(str(graph_cache))
                
                # Need to get embedding dimension from a test embedding
                test_emb = embedder.embed("test")
                vector_index = FaissIndex(embedding_dim=len(test_emb), use_gpu=config.use_gpu)
                vector_index.load(str(index_cache))
                
                logger.info("Successfully loaded cached graph and index")
                logger.info(f"Graph stats: {graph.get_stats()}")
                logger.info(f"Index stats: {vector_index.get_stats()}")
                
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Building from scratch.")
                use_cache = False
        
        if not use_cache or not graph_cache.exists():
            logger.info("="*80)
            logger.info("Building graph from scratch...")
            logger.info("="*80)
            
            # Load dataset
            logger.info("Loading dataset...")
            try:
                df = load_dataset(csv_path)
                logger.info(f"Loaded {len(df)} videos")
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                raise PipelineError(f"Dataset loading failed: {e}") from e
            
            # Chunk transcripts
            logger.info("Chunking transcripts...")
            chunks = []
            for idx, row in df.iterrows():
                try:
                    video_chunks = simple_chunk_text(
                        row['transcript'],
                        max_chunk_length=config.max_chunk_length,
                        min_chunk_length=config.min_chunk_length
                    )
                    
                    for chunk in video_chunks:
                        chunk['video_id'] = row['video_id']
                        chunks.append(chunk)
                        
                except Exception as e:
                    logger.warning(f"Failed to chunk video {row.get('video_id', idx)}: {e}")
                    continue
            
            logger.info(f"Created {len(chunks)} text chunks from {len(df)} videos")
            
            # Initialize components
            logger.info("Initializing extraction and embedding...")
            extractor = LLMExtractor(
                model=config.llm_model,
                api_key=api_key,
                timeout=config.timeout
            )
            
            # Build graph
            logger.info("Building knowledge graph...")
            builder = GraphBuilder(
                extractor=extractor,
                embedder=embedder,
                store_embeddings_in_graph=False  # Save memory
            )
            
            try:
                graph, vector_index = builder.build_from_chunks(chunks)
            except Exception as e:
                logger.error(f"Graph building failed: {e}")
                raise PipelineError(f"Graph construction failed: {e}") from e
            
            logger.info("Graph building complete!")
            logger.info(f"Builder stats: {builder.get_stats()}")
            logger.info(f"Graph stats: {graph.get_stats()}")
            logger.info(f"Index stats: {vector_index.get_stats()}")
            
            # Cache results
            logger.info("Saving graph and index to cache...")
            try:
                graph.save(str(graph_cache))
                vector_index.save(str(index_cache))
                logger.info("Successfully cached graph and index")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
        
        # Retrieval phase
        logger.info("="*80)
        logger.info("Retrieval Phase")
        logger.info("="*80)
        
        retriever = GraphRetriever(graph, vector_index)
        
        # Embed query
        logger.info(f"Embedding query: {query}")
        try:
            query_embedding = np.array(embedder.embed(query), dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise PipelineError(f"Query embedding failed: {e}") from e
        
        # Retrieve seed nodes
        logger.info(f"Retrieving top {config.top_k_nodes} seed nodes...")
        try:
            seed_nodes = retriever.retrieve_seed_nodes(
                query_embedding,
                top_k=config.top_k_nodes
            )
        except Exception as e:
            logger.error(f"Seed node retrieval failed: {e}")
            raise PipelineError(f"Retrieval failed: {e}") from e
        
        if not seed_nodes:
            logger.warning("No seed nodes found for query")
            return {
                'answer': "I couldn't find relevant information in my knowledge base for this query.",
                'metadata': {
                    'num_seed_nodes': 0,
                    'num_expanded_nodes': 0,
                    'subgraph_size': 0,
                    'query': query
                }
            }
        
        logger.info(f"Seed nodes: {[(nid, f'{score:.3f}') for nid, score in seed_nodes]}")
        
        # Expand via multi-hop
        logger.info(f"Expanding {config.max_hops} hops...")
        try:
            expanded_nodes = retriever.multi_hop_expand(seed_nodes, hops=config.max_hops)
            logger.info(f"Expanded to {len(expanded_nodes)} nodes")
        except Exception as e:
            logger.error(f"Multi-hop expansion failed: {e}")
            raise PipelineError(f"Graph expansion failed: {e}") from e
        
        # Build subgraph text
        try:
            subgraph_text = retriever.build_subgraph_text(expanded_nodes)
            logger.info(f"Subgraph text length: {len(subgraph_text)} chars")
        except Exception as e:
            logger.error(f"Subgraph text generation failed: {e}")
            raise PipelineError(f"Subgraph generation failed: {e}") from e
        
        # Generation phase
        logger.info("="*80)
        logger.info("Generation Phase")
        logger.info("="*80)
        
        reasoner = GraphReasoner(
            llm_model=config.llm_model,
            api_key=api_key,
            timeout=config.timeout
        )
        
        try:
            answer = reasoner.generate_answer(query, subgraph_text)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise PipelineError(f"Answer generation failed: {e}") from e
        
        logger.info("="*80)
        logger.info("Pipeline Complete")
        logger.info("="*80)
        
        # Collect statistics
        result = {
            'answer': answer,
            'metadata': {
                'num_seed_nodes': len(seed_nodes),
                'num_expanded_nodes': len(expanded_nodes),
                'subgraph_size': len(subgraph_text),
                'query': query,
                'embedder_stats': embedder.get_stats(),
                'extractor_stats': extractor.get_stats() if 'extractor' in locals() else {},
                'reasoner_stats': reasoner.get_stats()
            }
        }
        
        return result
        
    except PipelineError:
        raise
    except Exception as e:
        logger.error(f"Unexpected pipeline error: {e}", exc_info=True)
        raise PipelineError(f"Pipeline execution failed: {e}") from e


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <query> [--no-cache]")
        print("Example: python main.py 'How can I improve my self-esteem?'")
        print("\nOptions:")
        print("  --no-cache    Build graph from scratch, ignoring cache")
        sys.exit(1)
    
    # Parse arguments
    use_cache = "--no-cache" not in sys.argv
    args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    query = ' '.join(args)
    
    if not query:
        print("Error: Query cannot be empty")
        sys.exit(1)
    
    try:
        result = run_full_pipeline(
            csv_path="youtube_videos_export.csv",
            query=query,
            use_cache=use_cache
        )
        
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(result['answer'])
        print("\n" + "="*80)
        print("METADATA:")
        print("="*80)
        for key, value in result['metadata'].items():
            if key not in ['embedder_stats', 'extractor_stats', 'reasoner_stats']:
                print(f"{key}: {value}")
        
        # Print statistics
        print("\n" + "="*80)
        print("STATISTICS:")
        print("="*80)
        if 'embedder_stats' in result['metadata']:
            print(f"Embeddings: {result['metadata']['embedder_stats']}")
        if 'extractor_stats' in result['metadata']:
            print(f"Extractions: {result['metadata']['extractor_stats']}")
        if 'reasoner_stats' in result['metadata']:
            print(f"Generations: {result['metadata']['reasoner_stats']}")
        
    except PipelineError as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\nError: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error: {e}")
        sys.exit(1)
