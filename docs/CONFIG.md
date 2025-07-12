# Configuration Guide

Complete guide to configuring LocalRAG through environment variables, settings, and deployment options.

## Overview

LocalRAG uses environment variables as the single source of truth for all configuration. The `.env` file contains all settings needed for:

- Database connections (PostgreSQL, Neo4j, Redis)
- AI model configuration
- RAG parameters and tuning
- API and networking settings
- Docker and deployment options

## Environment File Setup

### Creating Configuration

```bash
# Copy example configuration
cp .env.example .env

# Edit with your settings
nano .env
```

### Environment File Structure

The `.env` file is organized into logical sections:

```bash
# Application Settings
APP_NAME=LocalRAG
DEBUG=false
PORT=8080

# Database Configuration
POSTGRES_HOST=host.docker.internal
POSTGRES_SSLMODE=disable

# Model Configuration  
EMBEDDING_MODEL=intfloat/multilingual-e5-base
AUTO_DOWNLOAD_MODELS=true

# RAG Settings
CHUNK_SIZE=1000
TOP_K=10
```

## Configuration Sections

### Application Settings

Core application configuration for runtime behavior:

```bash
# Application Identity
APP_NAME=LocalRAG                    # Application name
APP_VERSION=0.1.0                   # Version identifier
DEBUG=false                         # Enable debug mode (true/false)
LOG_LEVEL=INFO                      # Logging level (DEBUG/INFO/WARNING/ERROR)

# Network Settings
HOST=0.0.0.0                        # Host to bind to (0.0.0.0 for all interfaces)
PORT=8080                           # Port for API server
RELOAD=false                        # Auto-reload on changes (development only)
WORKERS=1                           # Number of worker processes
```

### Database Configuration

#### PostgreSQL (Primary Database)

```bash
# Connection Settings
POSTGRES_HOST=host.docker.internal  # Database host
POSTGRES_PORT=5432                  # Database port
POSTGRES_DB=local_rag              # Database name
POSTGRES_USER=dtsen                # Database username
POSTGRES_PASSWORD=dtsen            # Database password
POSTGRES_SSLMODE=disable           # SSL mode (disable/require/prefer)

# Alternative: Complete URL
DATABASE_URL=postgresql+asyncpg://user:pass@host:port/database?ssl=disable
```

**SSL Mode Options:**
- `disable` - No SSL (recommended for local development)
- `require` - SSL required (production)
- `prefer` - SSL preferred but not required
- `allow` - SSL allowed but not required

#### Neo4j (Knowledge Graph)

```bash
# Connection
NEO4J_URI=bolt://host.docker.internal:7687  # Neo4j Bolt protocol
NEO4J_USER=                                 # Username (empty for default)
NEO4J_PASSWORD=                             # Password (empty for default)

# Ports
NEO4J_HTTP_PORT=7474                        # HTTP interface port
NEO4J_BOLT_PORT=7687                        # Bolt protocol port

# Docker Configuration
NEO4J_AUTH=None                             # Authentication (None/neo4j/password)
NEO4J_dbms_default__database=neo4j          # Default database name
NEO4J_dbms_memory_heap_initial__size=512m   # Initial heap size
NEO4J_dbms_memory_heap_max__size=2G         # Maximum heap size
NEO4J_dbms_memory_pagecache_size=1G         # Page cache size
```

#### Redis (Caching & Sessions)

```bash
# Connection
REDIS_HOST=host.docker.internal     # Redis host
REDIS_PORT=6379                     # Redis port
REDIS_DB=0                          # Redis database number
REDIS_PASSWORD=                     # Redis password (optional)
REDIS_MAXMEMORY=512mb              # Maximum memory usage

# Alternative: Complete URL
REDIS_URL=redis://host.docker.internal:6379/0
```

### Model Configuration

AI models for embeddings and text processing:

```bash
# Model Selection
EMBEDDING_MODEL=intfloat/multilingual-e5-base    # Embedding model (multilingual support)
EXTRACTION_MODEL=asmud/cahya-indonesian-ner-tuned   # Indonesian BERT model for text extraction

# Model Management
MODELS_PATH=./models                              # Local model storage path
AUTO_DOWNLOAD_MODELS=true                        # Auto-download missing models
HF_TOKEN=hf_your_token_here                      # Hugging Face token (required for some models)
```

**Supported Embedding Models:**
- `intfloat/multilingual-e5-base` (default) - 768 dimensions, multilingual
- `sentence-transformers/all-MiniLM-L6-v2` - 384 dimensions, English
- `sentence-transformers/all-mpnet-base-v2` - 768 dimensions, English

**Indonesian Language Models:**
- `asmud/cahya-indonesian-ner-tuned` (default) - Indonesian BERT base model
- `asmud/cahya-indonesian-ner-tuned` - Indonesian BERT base model
- `cahya/bert-base-indonesian-522M` - Large Indonesian BERT model

**Model Storage:**
- Models are cached locally in `./models/` directory
- First run downloads models automatically
- Models are mounted as Docker volumes for persistence

### LLM Configuration

Large Language Model integration for enhanced response generation:

```bash
# LLM Service Control
ENABLE_LLM_CHAT=true                             # Enable/disable LLM chat features
LLM_FALLBACK_TO_EXTRACTIVE=true                 # Use extractive summarization if LLM fails

# Provider Configuration
LLM_PROVIDER=openai                              # LLM provider (openai/anthropic/ollama)
LLM_MODEL=google/gemini-2.0-flash-exp:free      # Model identifier
LLM_API_KEY=sk-or-v1-your-api-key-here         # Provider API key
LLM_ENDPOINT=https://openrouter.ai/api/v1       # Custom API endpoint (optional)

# Generation Parameters
LLM_TEMPERATURE=0.7                             # Generation creativity (0.0-1.0)
LLM_MAX_TOKENS=4196                             # Maximum response tokens
LLM_STREAMING=true                              # Enable streaming responses
LLM_TIMEOUT=30                                  # Request timeout (seconds)

# System Prompts
LLM_SYSTEM_PROMPT_FILE=prompts/asisten_sederhana_id.txt  # Indonesian assistant prompt
# LLM_SYSTEM_PROMPT_FILE=prompts/general_rag_assistant.txt  # English assistant prompt
```

**Supported LLM Providers:**

1. **OpenAI Compatible (via OpenRouter)**
   ```bash
   LLM_PROVIDER=openai
   LLM_ENDPOINT=https://openrouter.ai/api/v1
   LLM_MODEL=google/gemini-2.0-flash-exp:free    # Gemini via OpenRouter
   LLM_MODEL=openai/gpt-4                        # GPT-4 via OpenRouter
   LLM_MODEL=anthropic/claude-3-sonnet           # Anthropic via OpenRouter
   ```

2. **Direct OpenAI**
   ```bash
   LLM_PROVIDER=openai
   LLM_ENDPOINT=https://api.openai.com/v1        # Optional, defaults to OpenAI
   LLM_MODEL=gpt-4
   LLM_API_KEY=sk-your-openai-key
   ```

3. **Anthropic**
   ```bash
   LLM_PROVIDER=anthropic
   LLM_MODEL=claude-3-sonnet-20240229
   LLM_API_KEY=sk-ant-your-anthropic-key
   ```

4. **Local Ollama**
   ```bash
   LLM_PROVIDER=ollama
   LLM_ENDPOINT=http://localhost:11434
   LLM_MODEL=llama2:13b
   # No API key required
   ```

**System Prompt Configuration:**

Available prompt templates:
- `prompts/asisten_sederhana_id.txt` - Simple Indonesian assistant
- `prompts/asisten_rag_umum_id.txt` - General Indonesian RAG assistant  
- `prompts/asisten_teknis_id.txt` - Technical Indonesian assistant
- `prompts/general_rag_assistant.txt` - General English RAG assistant
- `prompts/technical_assistant.txt` - Technical English assistant

### RAG Configuration

Parameters for Retrieval-Augmented Generation:

```bash
# Text Processing (Optimized for Indonesian Documents)
CHUNK_SIZE=256                      # Text chunk size (characters) - optimized for regulations
CHUNK_OVERLAP=64                    # Overlap between chunks
MAX_CHUNK_CHARACTERS=7000           # Maximum characters per chunk
ENABLE_CHUNK_SPLITTING=true         # Enable intelligent chunk splitting

# Generation Settings  
MAX_TOKENS=4196                     # Maximum response tokens
TEMPERATURE=0.7                     # Generation temperature (0.0-1.0)

# Retrieval Settings (Tuned for Indonesian Content)
TOP_K=5                            # Number of chunks to retrieve (reduced for quality)
SIMILARITY_THRESHOLD=0.4           # Minimum similarity score (lowered for Indonesian)

# Knowledge Graph
KG_UPDATE_INTERVAL=300             # Update interval (seconds)
KG_BATCH_SIZE=100                  # Batch processing size

# Indonesian Processing Features
ENABLE_CORRUPTION_DETECTION=true   # Detect and filter garbled text
ENABLE_REGULATION_PATTERN_DETECTION=true  # Detect Indonesian regulation patterns
ENABLE_SEMANTIC_CHUNKING=true      # Use document structure-aware chunking
HEADER_PRIORITY_BOOST=1.3          # Similarity boost for document headers
```

**Parameter Tuning for Indonesian Documents:**

- **CHUNK_SIZE**: 256 chars optimized for Indonesian regulations (balance of context vs precision)
- **TOP_K**: 5 chunks recommended for Indonesian content (quality over quantity)
- **SIMILARITY_THRESHOLD**: 0.4 for Indonesian (lower due to language complexity)
- **TEMPERATURE**: 0.7 for balanced creativity and accuracy
- **HEADER_PRIORITY_BOOST**: 1.3x boost ensures document titles/headers rank higher
- **Corruption Detection**: Essential for OCR-processed Indonesian government PDFs

### API Configuration

REST API and networking settings:

```bash
# API Settings
API_V1_PREFIX=/api/v1              # API version prefix
CORS_ORIGINS=*                     # CORS allowed origins (* or comma-separated URLs)
MAX_UPLOAD_SIZE=100MB              # Maximum file upload size
RATE_LIMIT_PER_MINUTE=60          # API rate limiting

# Security (Production)
SECRET_KEY=your-secret-key-here    # JWT secret key
ACCESS_TOKEN_EXPIRE_MINUTES=30     # Token expiration
```

### Data Paths

File system paths for data storage:

```bash
# Data Directories
DATA_PATH=./data                   # Base data directory
UPLOADS_PATH=./data/uploads        # Document uploads
PROCESSED_PATH=./data/processed    # Processed documents

# Docker Volumes
# These paths are mounted as volumes in Docker containers
```

### Monitoring & Logging

Observability and debugging configuration:

```bash
# Metrics
ENABLE_METRICS=false               # Enable Prometheus metrics
METRICS_PORT=9090                  # Metrics server port

# Logging
LOG_FILE_PATH=./logs/localrag.log  # Log file location
LOG_ROTATION="1 day"               # Log rotation interval
LOG_RETENTION="30 days"            # Log retention period
```

## Environment-Specific Configuration

### Development Environment

```bash
# .env.development
DEBUG=true
LOG_LEVEL=DEBUG
RELOAD=true
POSTGRES_HOST=localhost
ENABLE_METRICS=true
```

### Production Environment

```bash
# .env.production
DEBUG=false
LOG_LEVEL=INFO
POSTGRES_SSLMODE=require
SECRET_KEY=complex-production-secret
CORS_ORIGINS=https://your-domain.com
RATE_LIMIT_PER_MINUTE=30
```

### Docker Environment

```bash
# For Docker Compose
POSTGRES_HOST=host.docker.internal  # Docker Desktop
# OR
POSTGRES_HOST=postgres              # If PostgreSQL in same compose

# Network Configuration
NEO4J_URI=bolt://host.docker.internal:7687
REDIS_HOST=host.docker.internal
```

## Advanced Configuration

### Custom Database URLs

For complex database setups:

```bash
# PostgreSQL with connection pooling
DATABASE_URL=postgresql+asyncpg://user:pass@host:port/db?ssl=disable&pool_size=20&max_overflow=30

# Redis with authentication
REDIS_URL=redis://:password@host:port/db

# Neo4j with authentication
NEO4J_URI=bolt://user:password@host:port
```

### Model Configuration

```bash
# Custom model paths
MODELS_PATH=/mnt/shared/models

# Specific model versions
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EXTRACTION_MODEL=bert-base-uncased

# GPU Configuration (if available)
CUDA_VISIBLE_DEVICES=0
```

### Performance Tuning

```bash
# High-throughput settings
WORKERS=4                          # Multiple worker processes
CHUNK_SIZE=1500                    # Larger chunks for better context
TOP_K=15                          # More retrieval candidates
KG_BATCH_SIZE=200                 # Larger knowledge graph batches

# Memory optimization
POSTGRES_POOL_SIZE=10             # Connection pool size
REDIS_MAXMEMORY=1GB               # Increase Redis memory
```

## Validation & Testing

### Configuration Validation

```bash
# Check configuration
python -c "from core.config import settings; print('Config loaded successfully')"

# Validate database connection
python utils/migrate.py health

# Test API configuration
curl http://localhost:8080/api/v1/health
```

### Environment Testing

```bash
# Test all services
docker-compose up -d
python utils/migrate.py status

# Validate embeddings
python -c "
from core.models import model_manager
import asyncio
async def test():
    await model_manager.initialize()
    print('Models loaded successfully')
asyncio.run(test())
"
```

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check host resolution: `ping $POSTGRES_HOST`
   - Verify credentials and database existence
   - Test SSL settings with different `POSTGRES_SSLMODE` values

2. **Model Download Issues**
   - Check internet connectivity
   - Verify `HF_TOKEN` if using private models
   - Ensure sufficient disk space in `MODELS_PATH`

3. **Port Conflicts**
   - Check if ports are available: `netstat -ln | grep :8080`
   - Use different ports in configuration
   - Verify Docker port mapping

4. **Memory Issues**
   - Increase Docker memory limits
   - Reduce `CHUNK_SIZE` and `TOP_K`
   - Optimize `REDIS_MAXMEMORY` and Neo4j heap size

### Configuration Debug

```bash
# Print current configuration
python -c "
from core.config import settings
import json
config_dict = {k: v for k, v in settings.__dict__.items() if not k.startswith('_')}
print(json.dumps(config_dict, indent=2, default=str))
"
```

## Security Considerations

### Production Security

```bash
# Strong secrets
SECRET_KEY=$(openssl rand -base64 32)

# Restricted CORS
CORS_ORIGINS=https://your-app.com,https://admin.your-app.com

# SSL enforcement
POSTGRES_SSLMODE=require

# Rate limiting
RATE_LIMIT_PER_MINUTE=30
```

### Network Security

- Use private networks for database connections
- Configure firewall rules for service ports
- Enable SSL/TLS for all external connections
- Use environment-specific credentials

For production deployment, refer to the Docker Compose configuration and environment-specific settings above.