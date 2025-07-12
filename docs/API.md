# API Reference

Complete REST API documentation for LocalRAG endpoints, including request/response formats and examples.

## Base URL

```
http://localhost:8080/api/v1
```

## Authentication

Currently, LocalRAG operates without authentication. For production use, implement authentication middleware.

## Health & Status

### Health Check

Check system health including database connections, model status, and LLM service availability.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "databases": {
    "postgres": true,
    "neo4j": true,
    "redis": true
  },
  "models": {
    "embedding_model": "intfloat/multilingual-e5-base",
    "extraction_model": "aadhistii/IndoBERT-NER",
    "device": "cpu",
    "models_path": "models",
    "embedding_model_loaded": true,
    "extraction_model_loaded": true
  },
  "llm_service": {
    "enabled": true,
    "provider": "openai",
    "model": "google/gemini-2.0-flash-exp:free",
    "endpoint": "https://openrouter.ai/api/v1",
    "available": true,
    "streaming_enabled": true
  },
  "components": {
    "database_manager": "healthy",
    "model_manager": "healthy",
    "llm_service": "healthy",
    "background_workers": "healthy (4 workers)"
  }
}
```

### Database Health

Check specific database connection status.

```http
GET /health/database
```

### Models Health

Check AI models loading status.

```http
GET /health/models
```

### LLM Service Health

Check LLM service status and provider availability.

```http
GET /health/llm
```

**Response:**
```json
{
  "llm_service": {
    "enabled": true,
    "provider": "openai",
    "model": "google/gemini-2.0-flash-exp:free",
    "endpoint": "https://openrouter.ai/api/v1",
    "available": true,
    "streaming_enabled": true,
    "fallback_enabled": true
  }
}
```

### Background Workers Health

Check background worker status and processing queue.

```http
GET /health/workers
```

**Response:**
```json
{
  "status": "healthy",
  "worker_count": 4,
  "running_workers": 4,
  "total_processed": 1248,
  "total_failed": 23,
  "queue_stats": {
    "queue_lengths": {
      "high": 2,
      "medium": 8,
      "low": 0
    },
    "total_pending": 10
  },
  "workers": [
    {
      "worker_id": "high-priority-1",
      "priority": "high",
      "is_running": true,
      "processed_count": 45,
      "failed_count": 2,
      "current_job": "550e8400-e29b-41d4-a716-446655440000",
      "start_time": "2024-01-15T08:30:00Z"
    }
  ]
}
```

## Document Management

### Upload Document (Async)

Upload a document for background processing. Returns immediately with a job ID for tracking.

```http
POST /documents/upload
Content-Type: multipart/form-data
```

**Parameters:**
- `file` (required): Document file (PDF, TXT, DOCX, etc.)

**Example:**
```bash
curl -X POST \
  -F "file=@document.pdf" \
  http://localhost:8080/api/v1/documents/upload
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "file_path": "/app/data/uploads/document.pdf",
  "status": "queued",
  "message": "Document 'document.pdf' queued for processing. Use job ID 550e8400-e29b-41d4-a716-446655440000 to track progress."
}
```

### List Documents

Get list of uploaded documents.

```http
GET /documents
```

**Query Parameters:**
- `limit` (optional): Number of documents to return (default: 50)
- `offset` (optional): Pagination offset (default: 0)
- `search` (optional): Search in document names

**Response:**
```json
{
  "documents": [
    {
      "id": 123,
      "file_name": "document.pdf",
      "file_size": 1024000,
      "chunk_count": 45,
      "created_at": "2025-07-02T08:00:00Z",
      "metadata": {}
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

### Get Document

Retrieve specific document information.

```http
GET /documents/{document_id}
```

### Delete Document

Remove document and all associated chunks.

```http
DELETE /documents/{document_id}
```

## Job Management

The async background processing system provides comprehensive job tracking and management capabilities.

### Get Job Status

Get detailed status of a background processing job.

```http
GET /jobs/{job_id}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "task_type": "single_document",
  "priority": "high",
  "status": "processing",
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:15Z",
  "completed_at": null,
  "progress": 65.5,
  "error_message": null,
  "retry_count": 0,
  "worker_id": "high-priority-1",
  "actual_duration": null
}
```

### Get Job Progress

Get real-time progress updates for a job.

```http
GET /jobs/{job_id}/progress
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "progress": 65.5,
  "message": "Processing page 13 of 20",
  "updated_at": "2024-01-15T10:35:22Z"
}
```

### Get Job Result

Get the final result of a completed job.

```http
GET /jobs/{job_id}/result
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result": {
    "document_id": "doc_123",
    "pg_document_id": 456,
    "file_path": "/app/data/uploads/document.pdf",
    "chunks_created": 45,
    "status": "success",
    "language": "indonesian",
    "processing_stats": {
      "pages_processed": 20,
      "ocr_engine_used": "page-by-page (easyocr,markitdown)",
      "avg_confidence": 0.89
    }
  },
  "completed_at": "2024-01-15T10:45:30Z"
}
```

### List Jobs

Get list of recent jobs with optional filtering.

```http
GET /jobs?status={status}&limit={limit}&offset={offset}
```

**Query Parameters:**
- `status` (optional): Filter by job status (pending, processing, completed, failed, cancelled)
- `limit` (optional): Number of jobs to return (default: 50, max: 200)
- `offset` (optional): Number of jobs to skip (default: 0)

**Example:**
```bash
curl "http://localhost:8080/api/v1/jobs?status=processing&limit=10"
```

### Cancel Job

Cancel a pending or processing job.

```http
POST /jobs/{job_id}/cancel
```

**Response:**
```json
{
  "message": "Job 550e8400-e29b-41d4-a716-446655440000 cancelled successfully"
}
```

### Retry Job

Retry a failed job.

```http
POST /jobs/{job_id}/retry
```

**Response:**
```json
{
  "message": "Job 550e8400-e29b-41d4-a716-446655440000 queued for retry"
}
```

### Job Statistics

Get processing statistics and metrics.

```http
GET /jobs/statistics
```

**Response:**
```json
{
  "total_24h": 156,
  "status_counts": {
    "completed": 142,
    "failed": 8,
    "processing": 4,
    "pending": 2
  },
  "processing_times": {
    "completed": {
      "avg_seconds": 23.5,
      "max_seconds": 145.2,
      "min_seconds": 2.1
    }
  },
  "queue_stats": {
    "queue_lengths": {
      "high": 2,
      "medium": 8,
      "low": 0
    },
    "total_pending": 10
  }
}
```

### Clean Up Old Jobs

Remove old completed/failed jobs from the database.

```http
DELETE /jobs/cleanup?days_old={days}
```

**Query Parameters:**
- `days_old` (optional): Age in days for job cleanup (default: 7, min: 1, max: 30)

**Response:**
```json
{
  "message": "Cleaned up 23 jobs older than 7 days"
}
```

## Worker Management

### Get Worker Status

Get status of all background workers.

```http
GET /workers
```

**Response:**
```json
[
  {
    "worker_id": "high-priority-1",
    "priority": "high",
    "is_running": true,
    "processed_count": 45,
    "failed_count": 2,
    "current_job": "550e8400-e29b-41d4-a716-446655440000",
    "start_time": "2024-01-15T08:30:00Z"
  },
  {
    "worker_id": "general-1",
    "priority": "any",
    "is_running": true,
    "processed_count": 123,
    "failed_count": 5,
    "current_job": null,
    "start_time": "2024-01-15T08:30:00Z"
  }
]
```

### Start Workers

Start background workers (if not auto-started).

```http
POST /workers/start
```

### Stop Workers

Stop all background workers.

```http
POST /workers/stop
```

## RAG Operations

### RAG Query

Perform retrieval-augmented generation query.

```http
POST /rag/query
Content-Type: application/json
```

**Request:**
```json
{
  "query": "What is the main topic of the uploaded documents?",
  "limit": 10,
  "similarity_threshold": 0.7,
  "include_metadata": true,
  "use_llm": true,
  "extractive_only": false
}
```

**Parameters:**
- `query` (required): The question or search query
- `limit` (optional): Maximum chunks to retrieve (default: 10)
- `similarity_threshold` (optional): Minimum similarity score (default: 0.7)
- `include_metadata` (optional): Include chunk metadata (default: false)
- `use_llm` (optional): Enable LLM generation for responses (default: true)
- `extractive_only` (optional): Return only retrieved chunks without LLM generation (default: false)

**Response:**
```json
{
  "query": "What is the main topic of the uploaded documents?",
  "response": "Based on the available information:\n\nThe main topic appears to be artificial intelligence and machine learning applications...",
  "context_chunks": 3,
  "metadata": {
    "retrieval_count": 3,
    "average_similarity": 0.85,
    "max_similarity": 0.92,
    "min_similarity": 0.78,
    "generation_method": "llm_generation",
    "llm_provider": "openai",
    "enhanced_similarity_used": true,
    "indonesian_processing": false,
    "regulation_pattern_detected": false
  },
  "chunks": [
    {
      "content": "Artificial intelligence and machine learning...",
      "similarity": 0.92,
      "enhanced_similarity": 0.95,
      "chunk_index": 1,
      "document_id": 123,
      "section_type": "document_header",
      "priority_boost_applied": true
    }
  ]
}
```

### Indonesian Regulation Query Example

**Request:**
```json
{
  "query": "Apa intisari dari Permendagri nomor 90 tahun 2019?",
  "limit": 5,
  "similarity_threshold": 0.4,
  "include_metadata": true,
  "use_llm": true,
  "extractive_only": false
}
```

**Response:**
```json
{
  "query": "Apa intisari dari Permendagri nomor 90 tahun 2019?",
  "enhanced_query": "apa intisari ringkasan isi pokok dari permendagri nomor 90 tahun 2019 PERATURAN MENTERI DALAM NEGERI REPUBLIK INDONESIA NOMOR 90 TAHUN 2019 MENDAGRI 90/2019",
  "response": "Berdasarkan dokumen yang tersedia:\n\nPermendagri Nomor 90 Tahun 2019 mengatur tentang Klasifikasi, Kodefikasi, dan Nomenklatur Perencanaan Pembangunan dan Keuangan Daerah...",
  "context_chunks": 5,
  "metadata": {
    "retrieval_count": 5,
    "average_similarity": 0.78,
    "max_similarity": 0.95,
    "min_similarity": 0.65,
    "generation_method": "llm_generation",
    "llm_provider": "openai",
    "enhanced_similarity_used": true,
    "indonesian_processing": true,
    "regulation_pattern_detected": true,
    "regulation_reference": "90/2019",
    "corruption_filtered_chunks": 15
  },
  "chunks": [
    {
      "content": "PERATURAN MENTERI DALAM NEGERI REPUBLIK INDONESIA NOMOR 90 TAHUN 2019...",
      "similarity": 0.89,
      "enhanced_similarity": 0.95,
      "chunk_index": 0,
      "document_id": 456,
      "section_type": "document_header",
      "priority_boost_applied": true,
      "regulation_reference_match": true,
      "language": "indonesian"
    }
  ]
}
```

### LLM-Free Query Examples

**Extractive-Only Query (No LLM Generation):**
```json
{
  "query": "What is the main topic of the uploaded documents?",
  "limit": 10,
  "similarity_threshold": 0.7,
  "include_metadata": true,
  "use_llm": false,
  "extractive_only": true
}
```

**Response (Extractive-Only):**
```json
{
  "query": "What is the main topic of the uploaded documents?",
  "response": null,
  "context_chunks": 3,
  "metadata": {
    "retrieval_count": 3,
    "average_similarity": 0.85,
    "max_similarity": 0.92,
    "min_similarity": 0.78,
    "generation_method": "extractive_only",
    "llm_provider": null,
    "enhanced_similarity_used": true,
    "indonesian_processing": false,
    "regulation_pattern_detected": false
  },
  "chunks": [
    {
      "content": "Artificial intelligence and machine learning...",
      "similarity": 0.92,
      "enhanced_similarity": 0.95,
      "chunk_index": 1,
      "document_id": 123,
      "section_type": "document_header",
      "priority_boost_applied": true
    }
  ]
}
```

### Search Documents

Search for similar content across documents.

```http
POST /rag/search
Content-Type: application/json
```

**Request:**
```json
{
  "query": "machine learning algorithms",
  "limit": 20,
  "similarity_threshold": 0.6
}
```

## Chat Interface

### Create Chat Session

Start a new chat session.

```http
POST /chat/session
Content-Type: application/json
```

**Request:**
```json
{
  "user_id": "user123"
}
```

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user123",
  "created_at": "2025-07-02T08:00:00Z"
}
```

### Send Message

Send a message in a chat session.

```http
POST /chat/message
Content-Type: application/json
```

**Request:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "What are the key benefits of RAG systems?",
  "use_rag": true,
  "rag_limit": 10
}
```

**Parameters:**
- `session_id` (required): Chat session ID
- `message` (required): User message
- `use_rag` (optional): Enable RAG for response (default: true)
- `rag_limit` (optional): Max chunks for RAG (default: 10)

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_message": {
    "id": "msg-001",
    "role": "user",
    "content": "What are the key benefits of RAG systems?",
    "timestamp": "2025-07-02T08:00:00Z"
  },
  "assistant_message": {
    "id": "msg-002",
    "role": "assistant",
    "content": "RAG systems offer several key benefits...",
    "timestamp": "2025-07-02T08:00:01Z",
    "metadata": {
      "rag_used": true,
      "context_chunks": 5,
      "rag_metadata": {
        "retrieval_count": 5,
        "average_similarity": 0.82
      }
    }
  },
  "conversation_length": 2
}
```

### Get Session History

Retrieve chat session conversation history.

```http
GET /chat/session/{session_id}
```

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user123",
  "created_at": "2025-07-02T08:00:00Z",
  "message_count": 4,
  "messages": [
    {
      "id": "msg-001",
      "role": "user",
      "content": "Hello",
      "timestamp": "2025-07-02T08:00:00Z"
    },
    {
      "id": "msg-002",
      "role": "assistant",
      "content": "Hello! How can I help you?",
      "timestamp": "2025-07-02T08:00:01Z"
    }
  ]
}
```

### List Chat Sessions

Get list of chat sessions for a user.

```http
GET /chat/sessions?user_id=user123
```

## Error Responses

### Standard Error Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "query",
      "issue": "Required field missing"
    }
  },
  "timestamp": "2025-07-02T08:00:00Z"
}
```

### HTTP Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request (invalid parameters)
- `404` - Not Found
- `422` - Unprocessable Entity (validation error)
- `500` - Internal Server Error
- `503` - Service Unavailable (system unhealthy)

### Common Error Codes

- `VALIDATION_ERROR` - Request validation failed
- `DOCUMENT_NOT_FOUND` - Document ID not found
- `SESSION_NOT_FOUND` - Chat session not found
- `PROCESSING_ERROR` - Document processing failed
- `RAG_ERROR` - RAG operation failed
- `MODEL_ERROR` - AI model error
- `DATABASE_ERROR` - Database operation failed
- `LLM_UNAVAILABLE` - LLM service unavailable, fallback used
- `LLM_PROVIDER_ERROR` - LLM provider API error
- `CORRUPTED_CONTENT` - Document content corruption detected
- `INDONESIAN_PROCESSING_ERROR` - Indonesian language processing failed
- `REGULATION_PARSING_ERROR` - Indonesian regulation parsing failed

## Rate Limiting

API endpoints are rate limited based on configuration:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1625097600
```

## SDK Examples

### Python

```python
import requests

class LocalRAGClient:
    def __init__(self, base_url="http://localhost:8080/api/v1"):
        self.base_url = base_url
    
    def health_check(self):
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def upload_document(self, file_path, metadata=None):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'metadata': metadata} if metadata else {}
            response = requests.post(f"{self.base_url}/documents/upload", 
                                   files=files, data=data)
        return response.json()
    
    def rag_query(self, query, limit=10, use_llm=True, extractive_only=False):
        data = {
            "query": query, 
            "limit": limit,
            "use_llm": use_llm,
            "extractive_only": extractive_only
        }
        response = requests.post(f"{self.base_url}/rag/query", json=data)
        return response.json()

# Usage
client = LocalRAGClient()
result = client.rag_query("What is machine learning?")
print(result['response'])
```

### JavaScript

```javascript
class LocalRAGClient {
  constructor(baseUrl = 'http://localhost:8080/api/v1') {
    this.baseUrl = baseUrl;
  }

  async healthCheck() {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }

  async uploadDocument(file, metadata = {}) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('metadata', JSON.stringify(metadata));
    
    const response = await fetch(`${this.baseUrl}/documents/upload`, {
      method: 'POST',
      body: formData
    });
    return response.json();
  }

  async ragQuery(query, limit = 10, useLlm = true, extractiveOnly = false) {
    const response = await fetch(`${this.baseUrl}/rag/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        query, 
        limit, 
        use_llm: useLlm, 
        extractive_only: extractiveOnly 
      })
    });
    return response.json();
  }
}

// Usage
const client = new LocalRAGClient();
const result = await client.ragQuery('What is machine learning?');
console.log(result.response);
```

### cURL Examples

```bash
# Upload Indonesian regulation document
curl -X POST \
  -F "file=@Permendagri_Nomor_90_Tahun_2019.pdf" \
  -F "metadata={\"category\": \"regulation\", \"language\": \"indonesian\"}" \
  http://localhost:8080/api/v1/documents/upload

# RAG query (English)
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "limit": 5, "use_llm": true, "extractive_only": false}' \
  http://localhost:8080/api/v1/rag/query

# RAG query (Indonesian regulation)
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "Apa intisari dari Permendagri nomor 90 tahun 2019?", "limit": 5, "similarity_threshold": 0.4, "use_llm": true, "extractive_only": false}' \
  http://localhost:8080/api/v1/rag/query

# Extractive-only query (no LLM generation)
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "limit": 10, "use_llm": false, "extractive_only": true}' \
  http://localhost:8080/api/v1/rag/query

# Create chat session
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123"}' \
  http://localhost:8080/api/v1/chat/session

# Send chat message (Indonesian)
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Bagaimana cara menggunakan sistem RAG untuk dokumen Indonesia?",
    "use_rag": true
  }' \
  http://localhost:8080/api/v1/chat/message

# Check LLM service health
curl -X GET http://localhost:8080/api/v1/health/llm
```

## OpenAPI Specification

The complete OpenAPI specification is available at:
```
http://localhost:8080/docs
```

Interactive API documentation (Swagger UI) is available at:
```
http://localhost:8080/docs
```

For setup and deployment information, see [SETUP.md](SETUP.md).