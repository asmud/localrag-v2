# Setup Guide

Comprehensive setup instructions for LocalRAG development and deployment environments.

## Prerequisites

### System Requirements

- **Python 3.11+**
- **Docker 20.10+** with Docker Compose
- **Git** for version control
- **Minimum 8GB RAM** (16GB recommended)
- **10GB free disk space** for models and data

### External Services

LocalRAG requires these database services:

- **PostgreSQL 14+** with pgvector extension
- **Neo4j 5.x** for knowledge graphs  
- **Redis 6+** for caching and sessions

These can be:
- Installed locally
- Run via Docker containers
- Used as managed cloud services

## Installation Methods

### Option 1: Quick Start (Recommended)

For rapid setup with all services:

```bash
# Clone repository
git clone https://github.com/your-org/localrag.git
cd localrag

# Run quick setup script
./quick-start.sh
```

This will:
- Create `.env` from template
- Start all services with Docker Compose  
- Run database migrations
- Download AI models
- Show service status and URLs

### Option 2: Docker Compose Setup

For containerized deployment:

```bash
# Clone and configure
git clone https://github.com/your-org/localrag.git
cd localrag

# Copy and edit configuration
cp .env.example .env
nano .env  # Edit database credentials

# Start all services
docker-compose up -d

# Run database migrations
python utils/migrate.py migrate-manual \
  --host=localhost \
  --database=local_rag \
  --user=your_user

# Check system health
curl http://localhost:8080/api/v1/health
```

### Option 3: Local Development Setup

For development with local Python environment:

```bash
# Clone repository
git clone https://github.com/your-org/localrag.git
cd localrag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -e ".[dev]"

# Setup configuration
cp .env.example .env
# Edit .env with your database settings

# Run migrations
python utils/migrate.py migrate

# Start development server
python api/main.py
```

## Detailed Configuration

### Database Setup

#### PostgreSQL with pgvector

**Docker (Recommended):**
```bash
docker run -d \
  --name postgres \
  -e POSTGRES_DB=local_rag \
  -e POSTGRES_USER=localrag \
  -e POSTGRES_PASSWORD=localrag_password \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

**Local Installation:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql-16 postgresql-16-pgvector

# Create database
sudo -u postgres createdb local_rag
sudo -u postgres createuser localrag
sudo -u postgres psql -c "ALTER USER localrag PASSWORD 'localrag_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE local_rag TO localrag;"

# Enable extensions
sudo -u postgres psql local_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres psql local_rag -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"
```

#### Neo4j

**Docker:**
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=none \
  neo4j:5-community
```

**Local Installation:**
```bash
# Download and install Neo4j Community
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 5' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install neo4j

# Start service
sudo systemctl enable neo4j
sudo systemctl start neo4j
```

#### Redis

**Docker:**
```bash
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:7-alpine
```

**Local Installation:**
```bash
# Ubuntu/Debian
sudo apt install redis-server

# Start service
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

### Environment Configuration

Create and configure `.env` file:

```bash
# Copy template
cp .env.example .env

# Edit configuration
nano .env
```

Key settings to update:

```bash
# Database connections
POSTGRES_HOST=localhost        # or container IP
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_DB=local_rag

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=                    # empty for no auth
NEO4J_PASSWORD=

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# API
PORT=8080
DEBUG=false
```

For complete configuration options, see [CONFIG.md](CONFIG.md).

## Database Migration

### Initial Migration

After database setup, run migrations:

```bash
# Check database health
python utils/migrate.py health

# View migration status
python utils/migrate.py status

# Run migrations
python utils/migrate.py migrate

# For manual setup (initial installation)
python utils/migrate.py migrate-manual \
  --host=localhost \
  --database=local_rag \
  --user=localrag
```

### Verify Migration

```bash
# Check database tables
docker exec postgres psql -U localrag -d local_rag -c "\dt"

# Verify vector extension
docker exec postgres psql -U localrag -d local_rag -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"

# Check migration status
python utils/migrate.py status
```

For detailed migration guide, see [MIGRATION.md](MIGRATION.md).

## Model Setup

### Automatic Model Download

Models are downloaded automatically on first run:

```bash
# Models will be downloaded to ./models/ directory
# This happens automatically when starting LocalRAG

# To pre-download models:
python -c "
from core.models import model_manager
import asyncio
asyncio.run(model_manager.initialize())
"
```

### Manual Model Download

```bash
# Download specific models
python -c "
import sentence_transformers
model = sentence_transformers.SentenceTransformer('intfloat/multilingual-e5-base')
model.save('./models/embedding/')
"
```

### Model Configuration

Configure models in `.env`:

```bash
# Embedding model (768 dimensions, multilingual support)
EMBEDDING_MODEL=intfloat/multilingual-e5-base

# Indonesian text extraction model (optimized for Indonesian NER)
EXTRACTION_MODEL=aadhistii/IndoBERT-NER

# Model storage path
MODELS_PATH=./models

# Auto-download missing models
AUTO_DOWNLOAD_MODELS=true

# Hugging Face token (required for some Indonesian models)
HF_TOKEN=your_hf_token_here

# LLM Configuration
ENABLE_LLM_CHAT=true
LLM_PROVIDER=openai
LLM_MODEL=google/gemini-2.0-flash-exp:free
LLM_API_KEY=sk-or-v1-your-openrouter-key
LLM_ENDPOINT=https://openrouter.ai/api/v1
```

**Supported Models:**

*Embedding Models:*
- `intfloat/multilingual-e5-base` (default) - Multilingual, 768 dimensions
- `sentence-transformers/all-MiniLM-L6-v2` - English only, 384 dimensions
- `sentence-transformers/all-mpnet-base-v2` - English only, 768 dimensions

*Indonesian Language Models:*
- `aadhistii/IndoBERT-NER` (default) - Indonesian NER and text extraction
- `asmud/cahya-indonesian-ner-tuned` - Base Indonesian BERT model
- `cahya/bert-base-indonesian-522M` - Large Indonesian BERT (522M parameters)

## Development Environment

### IDE Setup

**VS Code Configuration** (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.sortImports.args": ["--profile", "black"]
}
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Testing Setup

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py
```

### Code Quality Tools

```bash
# Code formatting
black .
isort .

# Linting
flake8 .

# Type checking
mypy .

# Security scanning
bandit -r src/
```

## Running Services

### Development Mode

```bash
# API server with auto-reload
python api/main.py

# Or with uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# Background workers (if using Celery)
celery -A core.tasks worker --loglevel=info

# Task scheduler
celery -A core.tasks beat --loglevel=info
```

### Production Mode

```bash
# Using Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Or with production server
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8080
```

## Verification

### Health Checks

```bash
# Check system health
curl http://localhost:8080/api/v1/health

# Detailed health check
curl http://localhost:8080/api/v1/health | jq

# Database-specific health
curl http://localhost:8080/api/v1/health/database

# Models health
curl http://localhost:8080/api/v1/health/models
```

### Functional Testing

```bash
# Upload Indonesian regulation document
curl -X POST \
  -F "file=@Permendagri_Nomor_90_Tahun_2019.pdf" \
  http://localhost:8080/api/v1/documents/upload

# Test RAG query (English)
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?"}' \
  http://localhost:8080/api/v1/rag/query

# Test Indonesian regulation query
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "Apa intisari dari Permendagri nomor 90 tahun 2019?"}' \
  http://localhost:8080/api/v1/rag/query

# Test LLM service
curl -X GET http://localhost:8080/api/v1/health/llm

# Create chat session
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user"}' \
  http://localhost:8080/api/v1/chat/session
```

### LLM Service Verification

```bash
# Check LLM service status
curl -X GET http://localhost:8080/api/v1/health/llm

# Test LLM generation with chat
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your_session_id",
    "message": "Test LLM response generation",
    "use_rag": false
  }' \
  http://localhost:8080/api/v1/chat/message

# Verify OpenRouter connection (if using)
python -c "
import requests
headers = {'Authorization': 'Bearer sk-or-v1-your-key-here'}
response = requests.get('https://openrouter.ai/api/v1/models', headers=headers)
print('OpenRouter connection:', response.status_code == 200)
"
```

### API Documentation

Once running, access interactive documentation:

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **OpenAPI JSON**: http://localhost:8080/openapi.json

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port
   lsof -i :8080
   
   # Kill process or change port in .env
   PORT=8081
   ```

2. **Database Connection Failed**
   ```bash
   # Check database is running
   docker ps | grep postgres
   
   # Test connection
   psql -h localhost -U localrag -d local_rag
   ```

3. **Models Not Loading**
   ```bash
   # Check disk space
   df -h
   
   # Check Hugging Face token
   echo $HF_TOKEN
   
   # Clear model cache
   rm -rf ./models/*
   
   # Re-download models
   python -c "from core.models import model_manager; import asyncio; asyncio.run(model_manager.initialize())"
   ```

4. **LLM Service Issues**
   ```bash
   # Check LLM service status
   curl http://localhost:8080/api/v1/health/llm
   
   # Verify API key
   echo $LLM_API_KEY
   
   # Test OpenRouter connection
   curl -H "Authorization: Bearer $LLM_API_KEY" https://openrouter.ai/api/v1/models
   
   # Check fallback mode
   grep "LLM_FALLBACK_TO_EXTRACTIVE" .env
   ```

5. **Indonesian Processing Issues**
   ```bash
   # Check Indonesian model loading
   python -c "
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained('aadhistii/IndoBERT-NER')
   print('Indonesian model loaded successfully')
   "
   
   # Test corruption detection
   python -c "
   from core.models import model_manager
   text = 'U M A R G O R P N A T A I G E K'
   result = model_manager._sanitize_text_for_embedding(text)
   print('Corruption detection working:', len(result) == 0)
   "
   ```

6. **Permission Errors**
   ```bash
   # Fix Docker permissions
   sudo usermod -aG docker $USER
   
   # Fix file permissions
   chmod -R 755 ./data ./models ./logs
   ```

### Log Analysis

```bash
# Check application logs
tail -f logs/localrag.log

# Docker container logs
docker-compose logs -f localrag

# Database logs
docker-compose logs postgres
```

### Performance Monitoring

```bash
# Check resource usage
docker stats

# Monitor API performance
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8080/api/v1/health

# Database performance
docker exec postgres psql -U localrag -d local_rag -c "
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC LIMIT 5;
"
```

## Next Steps

After successful setup:

1. **Upload Documents**: Use API or CLI to ingest documents
2. **Test RAG**: Query your documents with RAG endpoints
3. **Explore Chat**: Create chat sessions for conversational AI
4. **Monitor Performance**: Set up logging and metrics
5. **Scale Up**: Configure for production deployment

For production deployment, refer to the production environment configurations in [CONFIG.md](CONFIG.md).
For API usage, see [API.md](API.md).