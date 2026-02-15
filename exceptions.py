"""
Custom exceptions for the pipeline.
"""

class PipelineError(Exception):
    """Base exception for all pipeline errors."""
    pass

class DataLoadError(PipelineError):
    """Failed to load or validate dataset."""
    pass

class ExtractionError(PipelineError):
    """Entity extraction failed."""
    pass

class EmbeddingError(PipelineError):
    """Embedding generation failed."""
    pass

class GraphError(PipelineError):
    """Graph operation failed."""
    pass

class VectorIndexError(PipelineError):
    """Vector index operation failed."""
    pass

class RetrievalError(PipelineError):
    """Node retrieval failed."""
    pass

class GenerationError(PipelineError):
    """Answer generation failed."""
    pass
