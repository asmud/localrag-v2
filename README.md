# LocalRAG

Enterprise-grade local RAG (Retrieval-Augmented Generation) system with specialized Indonesian document processing, knowledge graph integration, and intelligent hallucination reduction.

## Features

🏢 **Enterprise-Ready** • 🧠 **Knowledge Graph Integration** • 🇮🇩 **Indonesian Language Specialist** • 🤖 **Multi-LLM Provider Support** • 📊 **RESTful API** • 🐳 **Docker Ready** • 🔄 **Semantic-Aware Chunking** • ⚡ **Async Background Processing**

### Key Features

- **Indonesian Document Processing**: Specialized for government regulations with corruption detection
- **Enhanced OCR Pipeline**: Multi-engine OCR with Indonesian text optimization
- **Async Background Processing**: Non-blocking uploads with real-time job tracking
- **Knowledge Graph Integration**: Neo4j-powered semantic relationships
- **Multi-LLM Support**: OpenAI, Anthropic, Ollama, and OpenRouter compatibility


## Quick Start

### Prerequisites
- Python 3.11+ • Docker & Docker Compose • 8GB+ RAM

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd localrag

# Option 1: One-command setup (recommended)
./quick-start.sh

# Option 2: Manual setup
cp .env.example .env
# Edit .env with your database settings
docker-compose up -d
python utils/migrate.py migrate-manual --host=localhost --database=local_rag --user=your_user
```

### Basic Usage

```bash
# Check system health (includes background workers status)
curl http://localhost:8080/api/v1/health

# Upload document (returns immediately with job ID)
curl -X POST -F "file=@Permendagri_Nomor_90_Tahun_2019.pdf" \
  http://localhost:8080/api/v1/documents/upload
# Returns: {"job_id": "uuid-here", "status": "queued", ...}

# Track job progress
curl http://localhost:8080/api/v1/jobs/{job_id}
curl http://localhost:8080/api/v1/jobs/{job_id}/progress

# Process directory (background batch processing)
curl -X POST -H "Content-Type: application/json" \
  -d '{"directory_path": "/path/to/documents"}' \
  http://localhost:8080/api/v1/documents/ingest-directory

# Monitor workers and queue
curl http://localhost:8080/api/v1/health/workers
curl http://localhost:8080/api/v1/workers

# RAG query (English)
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?"}' \
  http://localhost:8080/api/v1/rag/query

# RAG query (Indonesian) - Optimized for regulation documents
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "Apa intisari dari Permendagri nomor 90 tahun 2019?"}' \
  http://localhost:8080/api/v1/rag/query

# LLM-free extractive query (fast, no API costs)
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "klasifikasi urusan pemerintahan", "extractive_only": true}' \
  http://localhost:8080/api/v1/rag/query
```


## Architecture

- **`core/`** - Engine components (database, models, RAG, chat, background workers)
- **`api/`** - REST API endpoints and routing  
- **`utils/`** - CLI tools and migration utilities
- **`docs/`** - Comprehensive documentation
- **`migrations/`** - Database schema migrations including job tracking

**Tech Stack**: FastAPI • PostgreSQL+pgvector • Neo4j • Redis • Sentence Transformers • OpenRouter/Gemini • Indonesian BERT Models • Async Processing


## Documentation

📋 **[Setup Guide](docs/SETUP.md)** - Complete installation and development setup  
🔧 **[Configuration](docs/CONFIG.md)** - Environment variables and settings  
⚡ **[Background Processing](docs/BACKGROUND_PROCESSING.md)** - Async job management and worker configuration  
🗄️ **[Migration Guide](docs/MIGRATION.md)** - Database setup and troubleshooting  
🌐 **[API Reference](docs/API.md)** - REST API endpoints and examples  
🇮🇩 **[Indonesian Processing](docs/INDONESIAN_PROCESSING.md)** - Specialized Indonesian document features  
🔍 **[OCR Configuration](docs/OCR_CONFIGURATION.md)** - Enhanced OCR setup and optimization  
🤖 **[LLM Integration](docs/LLM_INTEGRATION.md)** - Multi-provider LLM configuration  
🚫 **[LLM-Free Usage](docs/LLM_FREE_USAGE.md)** - Operating without LLM dependencies  
💬 **[Chat Interface](docs/CHAT.md)** - Conversational AI setup and usage  

## API Documentation

- **Interactive Docs**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/api/v1/health

## Support

- **Issues**: [GitHub Issues](https://github.com/localrag/localrag/issues)
- **CLI Help**: `python utils/migrate.py --help`

## License

MIT License - see [LICENSE](LICENSE) file for details.