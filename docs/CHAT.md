# Chat System Documentation

## Overview

The LocalRAG chat system combines local models for document retrieval with LLM integration for conversational responses. This hybrid approach ensures privacy and speed for RAG operations while leveraging AI for natural conversation quality.

## Architecture Components

- **Local Models**: Handle embeddings (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`), Indonesian text processing (`asmud/cahya-indonesian-ner-tuned`), and context extraction
- **Enhanced RAG Engine**: Multi-factor similarity scoring, Indonesian regulation pattern detection, query enhancement
- **Indonesian Knowledge Graph**: Specialized processing for Indonesian government documents with entity validation
- **LLM Integration**: Multi-provider support (OpenRouter/Gemini, OpenAI, Anthropic, Ollama) with intelligent fallback
- **Hallucination Reduction**: Response validation against retrieved context with confidence scoring
- **Temporal Engine**: Time-aware context processing for enhanced relevance
- **Session Management**: Handle conversation persistence and threading with PostgreSQL storage
- **Streaming Support**: Real-time response delivery via Server-Sent Events with detailed progress tracking

## Conversation Flow

### 1. Initial Setup & Session Management

```
User Request → Create Chat Session → Session stored in Redis/Memory
```

- **Session Creation**: Each conversation gets a unique session ID
- **User Context**: Optional user_id for multi-user scenarios
- **Threading Support**: Messages can branch into new conversation threads
- **Persistence**: Sessions stored in Redis (temporary) with option to save to PostgreSQL (permanent)

### 2. Message Processing Flow

#### Step 1: Message Reception & Context Setup
```
User Message → Chat Engine → Add to Session History → Extract Thread Context
```

- User sends message via `/chat/message` or `/chat/message/stream`
- Message added to session with threading support (parent_message_id, thread_id)
- Recent conversation history extracted (configurable context window)

#### Step 2: Enhanced RAG Document Retrieval (Local Models Only)
```
User Query → Query Enhancement → Embedding Generation → Multi-Source Search → Enhanced Similarity Scoring → Context Retrieval
```

**Enhanced Local Processing:**
- **Query Enhancement**: 
  - Indonesian regulation pattern detection (Permendagri, Perda, etc.)
  - Synonym expansion for Indonesian terms
  - Formal regulation reference expansion
- **Embedding Model** (`intfloat/multilingual-e5-base`): Converts enhanced query to vector
- **Multi-Source Retrieval**:
  - **PostgreSQL Vector Search**: Fast similarity search using pgvector with enhanced embeddings
  - **Indonesian Knowledge Graph**: Entity-aware context from Indonesian government document relationships
  - **Neo4j Knowledge Graph**: Additional structured context from graph relationships
- **Enhanced Similarity Scoring**: Multi-factor algorithm considering:
  - Document structure priority (headers get 1.3x boost)
  - Regulation reference matching (1.25x boost)
  - Document position priority (earlier chunks prioritized)
  - Exact phrase matching (1.15x boost)
  - Content type weighting (headings, tables, etc.)
- **Context Ranking**: Top-K most relevant document chunks with enhanced scores
- **Corruption Detection**: Filter out garbled OCR text from Indonesian PDFs

#### Step 3: Response Generation (LLM or Extractive)

**If LLM Enabled and Available:**
```
Context Chunks + User Query → LLM Service → Conversational Response
```

**LLM Processing:**
- **Context-Enhanced Prompt**: Retrieved documents embedded in LLM prompt with Indonesian-aware formatting
- **Multi-Provider Support**: 
  - **OpenRouter** (Gemini 2.0 Flash, GPT-4, Anthropic) - Recommended
  - **Direct OpenAI** (GPT-3.5/4)
  - **Anthropic** (Sonnet, Haiku)
  - **Local Ollama** (Llama2, etc.)
- **Intelligent Provider Selection**: Automatic failover between providers
- **Streaming Response**: Real-time token streaming with progress indicators
- **Response Enhancement**: Indonesian entity validation and confidence scoring
- **Error Handling**: Automatic fallback to extractive if LLM fails

**If LLM Disabled or Unavailable:**
```
Context Chunks → Extractive Summarization → Document-Based Response
```

**Enhanced Extractive Processing:**
- **Intelligent Sentence Extraction**: Break context into sentences with Indonesian text awareness
- **Advanced Similarity Scoring**: Rank sentences by relevance using vector similarity
- **Corruption Filtering**: Remove garbled text patterns from OCR processing
- **Response Assembly**: Combine top sentences into coherent response with validation
- **Fallback Mechanisms**: Multiple levels of graceful degradation

### 3. Streaming Response Flow

#### Server-Sent Events (SSE) Sequence:

1. **`message_start`**: Session info and user message confirmation
2. **`rag_start`**: Beginning document retrieval
3. **`context_found`**: Number of relevant documents found
4. **`generation_start`**: Beginning response generation (LLM or extractive)
5. **`text_chunk`**: Real-time response tokens (multiple events)
6. **`message_complete`**: Final response with metadata
7. **`[DONE]`**: Stream completion signal

#### Example Streaming Sequence:
```json
data: {"type": "message_start", "session_id": "abc123", "user_message": {...}}

data: {"type": "rag_start", "message": "Retrieving relevant context..."}

data: {"type": "context_found", "context_chunks": 5, "metadata": {...}}

data: {"type": "generation_start", "message": "Generating response using openai..."}

data: {"type": "text_chunk", "content": "Based", "accumulated_text": "Based"}

data: {"type": "text_chunk", "content": " on", "accumulated_text": "Based on"}

data: {"type": "text_chunk", "content": " the", "accumulated_text": "Based on the"}

data: {"type": "message_complete", "assistant_message": {...}, "metadata": {...}}

data: [DONE]
```

### 4. Response Enhancement & Validation

#### Advanced Indonesian Language Support:
- **Automatic Detection**: Indonesian query pattern recognition (20%+ Indonesian indicators)
- **Enhanced Retrieval**: 
  - Indonesian Knowledge Graph entity enhancement
  - Regulation pattern detection (Permendagri, Perda, Perbup, etc.)
  - OCR corruption detection and filtering
  - Semantic-aware chunking for government documents
- **Entity Validation**: Indonesian government entity verification and confidence scoring
- **Localized Responses**: Indonesian system prompts and culturally appropriate responses
- **Performance Optimization**: Lowered similarity thresholds for Indonesian language complexity

#### Advanced Hallucination Reduction:
- **Multi-Layer Validation**: 
  - Response-context overlap analysis
  - Indonesian entity cross-validation
  - Confidence threshold enforcement (>70%)
- **Source Attribution**: Clear document source indication with similarity scores
- **Reliability Scoring**: Response confidence assessment with disclaimers when needed
- **Fallback Chain**: LLM → Extractive → Simple text processing

### 5. Session Persistence & Threading

#### Memory/Redis (Temporary):
- Active conversations stored in memory
- Redis backup for session recovery
- 24-hour TTL for automatic cleanup

#### PostgreSQL (Permanent):
- User-triggered persistence via `/chat/sessions/{id}/save`
- Full conversation history with threading
- Message relationships and metadata
- Search and retrieval capabilities

#### Threading Support:
- **Linear Conversations**: Standard back-and-forth chat
- **Branching Conversations**: Create new threads from any message
- **Thread Management**: List, navigate, and switch between threads
- **Context Inheritance**: Threads maintain parent context

## API Endpoints

### Core Chat Endpoints

#### Create Chat Session
```http
POST /api/v1/chat/sessions
Content-Type: application/json

{
  "user_id": "user123"
}
```

#### Send Message (Non-streaming)
```http
POST /api/v1/chat/message
Content-Type: application/json

{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "What are the key benefits of RAG systems?",
  "use_rag": true,
  "rag_limit": 10
}
```

#### Send Message (Streaming)
```http
POST /api/v1/chat/message/stream
Content-Type: application/json

{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Explain how LocalRAG works",
  "use_rag": true,
  "parent_message_id": "msg-123",
  "thread_id": "thread-456"
}
```

### Session Management

#### Save Session to PostgreSQL
```http
POST /api/v1/chat/sessions/{session_id}/save
Content-Type: application/json

{
  "metadata": {"source": "web_app", "category": "support"}
}
```

#### Get Session History
```http
GET /api/v1/chat/sessions/{session_id}/history
```

#### List Active Sessions
```http
GET /api/v1/chat/sessions
```

### Threading Endpoints

#### Create New Thread
```http
POST /api/v1/chat/sessions/{session_id}/threads
Content-Type: application/json

{
  "parent_message_id": "msg-123",
  "new_thread_id": "thread-456"
}
```

#### List Session Threads
```http
GET /api/v1/chat/sessions/{session_id}/threads
```

#### Get Thread Messages
```http
GET /api/v1/chat/sessions/{session_id}/threads/{thread_id}
```

## Configuration

### LLM Configuration

The system supports multiple LLM providers. Configure via environment variables:

#### OpenRouter Configuration (Recommended)
```bash
ENABLE_LLM_CHAT=true
LLM_PROVIDER=openai                              # Use OpenAI-compatible interface
LLM_ENDPOINT=https://openrouter.ai/api/v1       # OpenRouter endpoint
LLM_API_KEY=sk-or-v1-your-openrouter-key-here  # OpenRouter API key
LLM_MODEL=google/gemini-2.0-flash-exp:free      # Free Gemini 2.0 Flash
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4196
LLM_STREAMING=true
LLM_FALLBACK_TO_EXTRACTIVE=true
```

#### Direct OpenAI Configuration
```bash
ENABLE_LLM_CHAT=true
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
LLM_API_KEY=sk-your-openai-api-key-here
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024
LLM_STREAMING=true
LLM_FALLBACK_TO_EXTRACTIVE=true
```

#### Anthropic Configuration
```bash
ENABLE_LLM_CHAT=true
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-sonnet-20240229
LLM_API_KEY=sk-ant-your-anthropic-api-key-here
```

#### Local Ollama Configuration
```bash
ENABLE_LLM_CHAT=true
LLM_PROVIDER=ollama
LLM_MODEL=llama2
LLM_ENDPOINT=http://localhost:11434
```

#### Disable LLM (Extractive Only)
```bash
ENABLE_LLM_CHAT=false
```

### Enhanced RAG Configuration
```bash
# Local model settings (always used for retrieval)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2  # Current embedding model
EXTRACTION_MODEL=asmud/cahya-indonesian-ner-tuned  # Indonesian NER model
AUTO_DOWNLOAD_MODELS=true
HF_TOKEN=your_huggingface_token                   # Required for some Indonesian models

# Enhanced retrieval settings
TOP_K=5                                           # Quality over quantity
SIMILARITY_THRESHOLD=0.2                         # Current setting (lowered for Indonesian complexity)
CHUNK_SIZE=256                                    # Optimized for Indonesian regulations
CHUNK_OVERLAP=64                                  # Balanced overlap

# Indonesian processing
ENABLE_CORRUPTION_DETECTION=true                 # Filter garbled OCR text
ENABLE_REGULATION_PATTERN_DETECTION=true         # Detect regulation patterns
ENABLE_SEMANTIC_CHUNKING=true                    # Structure-aware chunking
HEADER_PRIORITY_BOOST=1.3                        # Boost document headers

# System prompts
LLM_SYSTEM_PROMPT_FILE=prompts/asisten_sederhana_id.txt  # Indonesian assistant
```

## Enhanced System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   User UI   │───▶│  FastAPI     │───▶│  Chat Engine    │
│             │    │  (REST/SSE)  │    │                 │
└─────────────┘    └──────────────┘    └─────────────────┘
                                                │
                                                ▼
                   ┌─────────────────────────────────────────┐
                   │         Enhanced RAG Engine             │
                   │  ┌─────────────┐  ┌─────────────────┐   │
                   │  │Local Models │  │Query Enhancement│   │
                   │  │Multilingual │  │Indonesian Regex │   │
                   │  │E5 + IndoBERT│  │Pattern Detection│   │
                   │  └─────────────┘  └─────────────────┘   │
                   │  ┌─────────────┐  ┌─────────────────┐   │
                   │  │Multi-Source │  │Enhanced Scoring │   │
                   │  │PostgreSQL + │  │Structure Priority│   │
                   │  │Indonesian KG│  │Regulation Match │   │
                   │  └─────────────┘  └─────────────────┘   │
                   └─────────────────────────────────────────┘
                                                │
                                                ▼
                   ┌─────────────────────────────────────────┐
                   │     Multi-Provider LLM Service         │
                   │  ┌─────────────┐  ┌─────────────────┐   │
                   │  │OpenRouter   │  │Direct Providers │   │
                   │  │Gemini/GPT-4 │  │OpenAI/Anthropic │   │
                   │  │/Anthropic   │  │/Ollama          │   │
                   │  └─────────────┘  └─────────────────┘   │
                   │  ┌─────────────┐  ┌─────────────────┐   │
                   │  │Hallucination│  │Enhanced Extract.│   │
                   │  │Reducer      │  │Indonesian-Aware │   │
                   │  │+ Validation │  │Corruption Filter│   │
                   │  └─────────────┘  └─────────────────┘   │
                   └─────────────────────────────────────────┘
                                                │
                                                ▼
                   ┌─────────────────────────────────────────┐
                   │      Advanced Session Storage           │
                   │  ┌─────────────┐  ┌─────────────────┐   │
                   │  │   Redis     │  │   PostgreSQL    │   │
                   │  │ (Temporary) │  │  (Permanent)    │   │
                   │  │+ Threading  │  │+ Chat History   │   │
                   │  └─────────────┘  └─────────────────┘   │
                   │  ┌─────────────┐  ┌─────────────────┐   │
                   │  │Indonesian KG│  │   Temporal      │   │
                   │  │Entity Store │  │   Engine        │   │
                   │  │+ Validation │  │Time-aware Context│   │
                   │  └─────────────┘  └─────────────────┘   │
                   └─────────────────────────────────────────┘
```

## Implementation Examples

### JavaScript Client (Streaming)
```javascript
async function sendStreamingMessage(sessionId, message) {
  const response = await fetch('/api/v1/chat/message/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      message: message,
      use_rag: true
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (data === '[DONE]') return;

        try {
          const event = JSON.parse(data);
          handleStreamEvent(event);
        } catch (e) {
          console.error('Failed to parse event:', e);
        }
      }
    }
  }
}

function handleStreamEvent(event) {
  switch (event.type) {
    case 'message_start':
      console.log('Message started:', event.session_id);
      break;
    case 'rag_start':
      console.log('Retrieving context...');
      break;
    case 'context_found':
      console.log(`Found ${event.context_chunks} relevant documents`);
      break;
    case 'generation_start':
      console.log(event.message);
      break;
    case 'text_chunk':
      appendToResponse(event.content);
      break;
    case 'message_complete':
      console.log('Response complete:', event.metadata);
      break;
    case 'error':
      console.error('Stream error:', event.error);
      break;
  }
}
```

### Python Client
```python
import httpx
import json

class LocalRAGClient:
    def __init__(self, base_url="http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.session_id = None

    async def create_session(self, user_id=None):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/sessions",
                json={"user_id": user_id}
            )
            data = response.json()
            self.session_id = data["session_id"]
            return self.session_id

    async def send_message(self, message, use_rag=True):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/message",
                json={
                    "session_id": self.session_id,
                    "message": message,
                    "use_rag": use_rag
                }
            )
            return response.json()

    async def stream_message(self, message, use_rag=True):
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/message/stream",
                json={
                    "session_id": self.session_id,
                    "message": message,
                    "use_rag": use_rag
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                            yield event
                        except json.JSONDecodeError:
                            continue

# Usage
async def main():
    client = LocalRAGClient()
    await client.create_session("user123")
    
    async for event in client.stream_message("What is LocalRAG?"):
        if event["type"] == "text_chunk":
            print(event["content"], end="", flush=True)
        elif event["type"] == "message_complete":
            print(f"\n\nGeneration method: {event['metadata']['generation_method']}")
```

## Response Types & Metadata

### Enhanced LLM Response with Indonesian Processing
```json
{
  "session_id": "abc123",
  "user_message": {"content": "Apa intisari dari Permendagri nomor 90 tahun 2019?"},
  "assistant_message": {
    "id": "msg-456",
    "role": "assistant",
    "content": "Intisari dari Permendagri Nomor 90 Tahun 2019 adalah tentang Klasifikasi, Kodefikasi, dan Nomenklatur Perencanaan Pembangunan dan Keuangan Daerah...",
    "metadata": {
      "rag_used": true,
      "context_chunks": 5,
      "generation_method": "llm_generation",
      "enhanced_similarity_used": true,
      "regulation_pattern_detected": true,
      "language_detected": "indonesian",
      "llm_info": {
        "provider": "openai",
        "model": "google/gemini-2.0-flash-exp:free",
        "available": true,
        "streaming_enabled": true
      }
    }
  },
  "language": "indonesian",
  "indonesian_context": {
    "entities_in_response": [
      {"text": "Nomor", "matched_entity": "Nomor 90 Tahun 2019", "confidence": 0.9}
    ],
    "kg_quality_score": 0.81,
    "total_indonesian_entities": 114
  },
  "conversation_length": 2,
  "metadata": {
    "retrieval_count": 5,
    "average_similarity": 0.90,
    "max_similarity": 1.0,
    "corruption_filtered_chunks": 15
  }
}
```

### Extractive Response
```json
{
  "assistant_message": {
    "metadata": {
      "rag_used": true,
      "context_chunks": 3,
      "generation_method": "extractive_summarization",
      "llm_info": null
    }
  }
}
```

### Fallback Response
```json
{
  "assistant_message": {
    "metadata": {
      "generation_method": "extractive_fallback",
      "fallback_reason": "LLM unavailable"
    }
  }
}
```

## Key Advantages

✅ **Local Privacy**: Document processing and embeddings stay local  
✅ **Enhanced LLM Integration**: Multi-provider support with intelligent fallback  
✅ **Indonesian Specialization**: Advanced processing for Indonesian government documents  
✅ **Corruption-Resistant**: OCR error detection and filtering for Indonesian PDFs  
✅ **Regulation-Aware**: Automatic detection and enhancement of Indonesian regulation queries  
✅ **Multi-Factor Scoring**: Enhanced similarity algorithms considering document structure  
✅ **Real-time Feedback**: Streaming with detailed progress indicators  
✅ **Hallucination Reduction**: Response validation and confidence scoring  
✅ **Graceful Degradation**: Multiple fallback layers (LLM → Extractive → Simple)  
✅ **Enterprise-Ready**: Scalable architecture with session management and threading  
✅ **Quality Optimization**: Optimized parameters for Indonesian language complexity  

## Troubleshooting

### Common Issues

#### LLM Not Working
- Check `ENABLE_LLM_CHAT=true` in environment
- Verify API key is correct and has permissions
- Check network connectivity to LLM provider
- Review logs for specific error messages

#### Slow Response Times
- Reduce `TOP_K` value for fewer document chunks
- Optimize PostgreSQL indexes
- Use faster LLM models (gpt-3.5-turbo vs gpt-4)
- Enable Redis for session caching

#### Empty/Poor Responses
- Check if documents are properly indexed
- Verify embedding model is working
- Adjust `SIMILARITY_THRESHOLD` (lower = more results)
- Review document chunk size and overlap settings

#### Streaming Issues
- Ensure client supports Server-Sent Events
- Check firewall/proxy settings for streaming
- Verify proper HTTP headers for SSE
- Test with curl or browser developer tools

### Debug Mode
Enable debug logging to trace the conversation flow:
```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

This will provide detailed logs for each step of the conversation process, including RAG retrieval, LLM calls, and response generation.