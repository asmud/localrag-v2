import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application Settings
    app_name: str = Field(default="LocalRAG", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8000, description="Port to bind to")


    # Database Configuration
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_db: str = Field(default="localrag", description="PostgreSQL database name")
    postgres_user: str = Field(default="localrag", description="PostgreSQL username")
    postgres_password: str = Field(default="localrag_password", description="PostgreSQL password")
    postgres_sslmode: str = Field(default="prefer", description="PostgreSQL SSL mode")
    database_url: Optional[str] = Field(default=None, description="Complete database URL")

    @property
    def get_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}?ssl={self.postgres_sslmode}"

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="neo4j_password", description="Neo4j password")

    # Redis Configuration
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_url: Optional[str] = Field(default=None, description="Complete Redis URL")

    @property
    def get_redis_url(self) -> str:
        if self.redis_url:
            return self.redis_url
        password_part = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{password_part}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # Model Configuration
    models_path: str = Field(default="./models", description="Path to store models")
    embedding_model: str = Field(default="intfloat/multilingual-e5-base", description="Embedding model name")
    extraction_model: str = Field(default="asmud/cahya-indonesian-ner-tuned", description="Extraction model name")
    auto_download_models: bool = Field(default=True, description="Automatically download models")
    hf_token: Optional[str] = Field(default=None, description="Hugging Face token")

    # Data Paths
    data_path: str = Field(default="./data", description="Base data directory")
    uploads_path: str = Field(default="./data/uploads", description="Uploads directory")
    processed_path: str = Field(default="./data/processed", description="Processed data directory")

    # RAG Configuration
    chunk_size: int = Field(default=1000, description="Text chunk size in words")
    chunk_overlap: int = Field(default=200, description="Text chunk overlap in words")
    max_chunk_characters: int = Field(default=8000, description="Maximum characters per chunk for Neo4j indexing")
    enable_chunk_splitting: bool = Field(default=True, description="Enable automatic splitting of oversized chunks")
    max_tokens: int = Field(default=2048, description="Maximum tokens in response")
    temperature: float = Field(default=0.7, description="Generation temperature")
    top_k: int = Field(default=10, description="Top-k retrieval")
    similarity_threshold: float = Field(default=0.4, description="Similarity threshold for vector search")
    indonesian_similarity_multiplier: float = Field(default=0.9, description="Multiplier for Indonesian query similarity threshold")
    graph_similarity_threshold: float = Field(default=0.6, description="Similarity threshold for graph relationships")

    # LLM Configuration for Chat Response Generation
    enable_llm_chat: bool = Field(default=False, description="Enable LLM for chat response generation")
    llm_provider: str = Field(default="openai", description="LLM provider (openai, anthropic, ollama)")
    llm_model: str = Field(default="gpt-3.5-turbo", description="LLM model name")
    llm_api_key: Optional[str] = Field(default=None, description="LLM API key")
    llm_endpoint: Optional[str] = Field(default=None, description="Custom LLM endpoint URL")
    llm_temperature: float = Field(default=0.7, description="LLM generation temperature")
    llm_max_tokens: int = Field(default=1024, description="Maximum tokens for LLM response")
    llm_streaming: bool = Field(default=True, description="Enable LLM streaming responses")
    llm_timeout: int = Field(default=30, description="LLM request timeout in seconds")
    llm_fallback_to_extractive: bool = Field(default=True, description="Fallback to extractive if LLM fails")
    extractive_default_mode: bool = Field(default=False, description="Use extractive summarization by default")
    extractive_min_confidence: float = Field(default=0.3, description="Minimum confidence for extractive responses")
    extractive_max_sentences: int = Field(default=5, description="Maximum sentences in extractive response")
    llm_system_prompt: str = Field(
        default="You are a helpful AI assistant with access to relevant document information. Use the provided context to answer the user's question accurately and naturally in a conversational manner.",
        description="System prompt for LLM chat responses"
    )
    llm_system_prompt_file: Optional[str] = Field(
        default=None,
        description="Path to file containing system prompt (overrides llm_system_prompt if specified)"
    )

    @property
    def get_system_prompt(self) -> str:
        """Get the system prompt, loading from file if specified"""
        if self.llm_system_prompt_file:
            try:
                if os.path.isfile(self.llm_system_prompt_file):
                    with open(self.llm_system_prompt_file, 'r', encoding='utf-8') as f:
                        return f.read().strip()
                else:
                    # Try relative to project root
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    file_path = os.path.join(project_root, self.llm_system_prompt_file)
                    if os.path.isfile(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            return f.read().strip()
                    else:
                        # Use default prompt if file not found
                        return self.llm_system_prompt
            except Exception as e:
                # Use default prompt if error loading file
                return self.llm_system_prompt
        return self.llm_system_prompt

    # Knowledge Graph
    kg_update_interval: int = Field(default=300, description="KG update interval in seconds")
    kg_batch_size: int = Field(default=100, description="KG batch processing size")

    # API Configuration
    api_v1_prefix: str = Field(default="/api/v1", description="API v1 prefix")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        description="CORS allowed origins (comma-separated)"
    )
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from string to list"""
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(',')]
        return self.cors_origins
    max_upload_size: str = Field(default="100MB", description="Maximum upload size")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute")


    # Monitoring and Logging
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    log_file_path: str = Field(default="./logs/localrag.log", description="Log file path")
    log_rotation: str = Field(default="1 day", description="Log rotation interval")
    log_retention: str = Field(default="30 days", description="Log retention period")

    # OCR Configuration
    ocr_engine: str = Field(default="markitdown", description="Primary OCR engine (markitdown|easyocr|paddleocr|tesseract)")
    ocr_fallback_engine: str = Field(default="tesseract", description="Fallback OCR engine")
    ocr_languages: List[str] = Field(default=["en", "id"], description="OCR languages to support")
    ocr_confidence_threshold: float = Field(default=0.6, description="Minimum OCR confidence score")
    ocr_enable_preprocessing: bool = Field(default=True, description="Enable image preprocessing")
    ocr_enable_postprocessing: bool = Field(default=True, description="Enable text post-processing")
    ocr_gpu_enabled: bool = Field(default=True, description="Enable GPU acceleration for OCR")
    ocr_dpi_threshold: int = Field(default=300, description="Minimum DPI for OCR processing")
    ocr_max_image_size: int = Field(default=4096, description="Maximum image size for OCR")
    
    # OCR Model Pre-downloading Configuration
    ocr_predownload_models: bool = Field(default=True, description="Pre-download OCR models during startup")
    ocr_models_cache_dir: str = Field(default="./models/ocr", description="Directory to cache OCR models")
    ocr_download_timeout: int = Field(default=600, description="Timeout for model downloads in seconds")
    ocr_lazy_loading: bool = Field(default=True, description="Enable lazy loading for OCR engines to prevent immediate model downloads")
    
    # OCR Engine Control Configuration
    ocr_disable_engines: List[str] = Field(default=[], description="List of OCR engines to disable (e.g., ['paddleocr'] if problematic)")
    ocr_force_cpu_mode: bool = Field(default=False, description="Force all OCR engines to use CPU mode for stability")
    ocr_safe_mode: bool = Field(default=True, description="Enable safe mode with timeouts and error handling")
    ocr_initialization_timeout: int = Field(default=60, description="Timeout for OCR engine initialization in seconds")
    
    # Task Queue Configuration
    task_queue_enabled: bool = Field(default=True, description="Enable background task processing")
    task_workers_auto_start: bool = Field(default=True, description="Automatically start background workers on application startup")
    task_queue_name: str = Field(default="localrag_tasks", description="Redis task queue name")
    task_max_workers: int = Field(default=4, description="Maximum number of background workers")
    task_max_retries: int = Field(default=3, description="Maximum retry attempts for failed jobs")
    task_retry_delay: int = Field(default=60, description="Delay between retries in seconds")
    task_job_retention: int = Field(default=86400, description="Job data retention period in seconds (24 hours)")
    task_progress_update_interval: int = Field(default=5, description="Progress update interval in seconds")
    task_high_priority_workers: int = Field(default=2, description="Workers dedicated to high priority tasks")
    task_batch_size: int = Field(default=10, description="Batch size for directory processing")
    
    # Duplicate Detection Configuration
    duplicate_detection_enabled: bool = Field(default=True, description="Enable content-based duplicate detection")
    duplicate_detection_method: str = Field(default="content_hash", description="Method for duplicate detection: 'content_hash', 'file_path', or 'both'")
    duplicate_detection_skip_duplicates: bool = Field(default=True, description="Skip processing duplicate content or force re-ingest")
    duplicate_detection_cleanup_failed: bool = Field(default=True, description="Automatically cleanup failed/incomplete documents with same content hash")
    duplicate_detection_log_level: str = Field(default="info", description="Log level for duplicate detection messages")
    
    @field_validator('duplicate_detection_method')
    @classmethod
    def validate_duplicate_detection_method(cls, v):
        valid_methods = ['content_hash', 'file_path', 'both']
        if v not in valid_methods:
            raise ValueError(f"duplicate_detection_method must be one of {valid_methods}")
        return v
    
    @field_validator('duplicate_detection_log_level')
    @classmethod
    def validate_duplicate_detection_log_level(cls, v):
        valid_levels = ['debug', 'info', 'warning', 'error']
        if v not in valid_levels:
            raise ValueError(f"duplicate_detection_log_level must be one of {valid_levels}")
        return v
    
    # Development Settings
    reload: bool = Field(default=False, description="Auto-reload on changes")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # Field validators
    @field_validator('ocr_languages', mode='before')
    @classmethod
    def parse_ocr_languages(cls, v):
        if isinstance(v, str):
            # Handle comma-separated string from environment variable
            return [lang.strip() for lang in v.split(',') if lang.strip()]
        return v
    
    @field_validator('ocr_disable_engines', mode='before')
    @classmethod
    def parse_ocr_disable_engines(cls, v):
        if isinstance(v, str):
            # Handle comma-separated string from environment variable
            if not v.strip():
                return []
            return [engine.strip() for engine in v.split(',') if engine.strip()]
        return v


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()