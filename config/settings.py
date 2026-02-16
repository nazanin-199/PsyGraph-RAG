from pydantic import Field, SecretStr, validator
from pydantic_settings import BaseSettings
from typing import Optional, Literal
from pathlib import Path


class Settings(BaseSettings):
    """Application configuration with validation."""
    
    # API Keys (OpenRouter)
    openai_api_key: SecretStr = Field(..., env="OPENAI_API_KEY")
    
    # OpenRouter Models
    llm_model: str = Field(default="openai/gpt-4o-mini", env="LLM_MODEL")
    embedding_model: str = Field(default="openai/text-embedding-3-small", env="EMBEDDING_MODEL")
    
    # Retrieval Parameters
    top_k_nodes: int = Field(default=5, ge=1, le=20, env="TOP_K_NODES")
    max_hops: int = Field(default=2, ge=1, le=5, env="MAX_HOPS")
    
    # Processing Configuration
    batch_size: int = Field(default=30, ge=1, le=2048, env="BATCH_SIZE")
    max_workers: int = Field(default=1, ge=1, le=16, env="MAX_WORKERS")
    enable_parallel: bool = Field(default=False, env="ENABLE_PARALLEL")
    
    # Chunking Parameters
    min_chunk_length: int = Field(default=500, ge=100)
    max_chunk_length: int = Field(default=2000, ge=500)
    chunk_overlap: int = Field(default=100, ge=0)
    
    # Storage Paths
    cache_dir: Path = Field(default=Path("./cache"), env="CACHE_DIR")
    data_dir: Path = Field(default=Path("./data"), env="DATA_DIR")
    output_dir: Path = Field(default=Path("./output"), env="OUTPUT_DIR")
    
    # Performance
    embedding_cache_size: int = Field(default=10000, ge=0)
    use_gpu: bool = Field(default=False, env="USE_GPU")
    
    # Retry Configuration
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout: int = Field(default=60, ge=5, le=120)
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        env="LOG_LEVEL"
    )
    log_file: Optional[Path] = Field(default=Path("pipeline.log"), env="LOG_FILE")
    
    @validator("cache_dir", "data_dir", "output_dir")
    def create_directories(cls, v):
        if v and isinstance(v, Path):
            v.mkdir(parents=True, exist_ok=True)
        return v
    
    def validate(self) -> None:
        key = self.openai_api_key.get_secret_value()
        
        if not key.startswith("sk-or-"):
            raise ValueError(" This configuration requires OpenRouter API key (sk-or-)")
        
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Backward compatibility constants
MODEL_NAME = "openai/gpt-4o-mini"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
TOP_K_NODES = 5
MAX_HOPS = 2
