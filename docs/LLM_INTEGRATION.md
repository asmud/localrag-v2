# LLM Integration Guide

Comprehensive guide to configuring and using Large Language Model (LLM) providers with LocalRAG for enhanced response generation and conversational AI capabilities.

## Overview

LocalRAG supports multiple LLM providers with intelligent fallback mechanisms, streaming responses, and optimized RAG-enhanced generation. The system automatically integrates retrieved context with LLM responses for accurate, source-grounded answers.

## Supported Providers

### 1. OpenRouter (Recommended)

Access to multiple models through a single API, including Gemini, GPT-4, Anthropic models, and others.

```bash
# OpenRouter Configuration
LLM_PROVIDER=openai                              # Use OpenAI-compatible interface
LLM_ENDPOINT=https://openrouter.ai/api/v1       # OpenRouter endpoint
LLM_API_KEY=sk-or-v1-your-openrouter-key-here  # OpenRouter API key
LLM_MODEL=google/gemini-2.0-flash-exp:free      # Model selection
```

**Available Models via OpenRouter:**
- `google/gemini-2.0-flash-exp:free` - Gemini 2.0 Flash (free tier)
- `google/gemini-pro` - Gemini Pro (paid)
- `openai/gpt-4` - GPT-4 (paid)
- `openai/gpt-3.5-turbo` - GPT-3.5 Turbo (paid)
- `anthropic/claude-3-sonnet` - Anthropic Sonnet (paid)
- `meta-llama/llama-2-70b-chat` - Llama 2 70B (paid)

### 2. Direct OpenAI

```bash
LLM_PROVIDER=openai
LLM_API_KEY=sk-your-openai-api-key
LLM_MODEL=gpt-4                                 # or gpt-3.5-turbo
# LLM_ENDPOINT is optional, defaults to OpenAI
```

### 3. Anthropic

```bash
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-your-anthropic-key
LLM_MODEL=claude-3-sonnet-20240229             # or claude-3-haiku-20240307
LLM_ENDPOINT=https://api.anthropic.com/v1/messages
```

### 4. Local Ollama

```bash
LLM_PROVIDER=ollama
LLM_ENDPOINT=http://localhost:11434            # Local Ollama instance
LLM_MODEL=llama2:13b                           # Installed model
# No API key required
```

## Configuration

### Environment Variables

```bash
# LLM Service Control
ENABLE_LLM_CHAT=true                           # Enable/disable LLM features
LLM_FALLBACK_TO_EXTRACTIVE=true               # Fallback to extractive on LLM failure

# Provider Settings
LLM_PROVIDER=openai                            # Provider type
LLM_MODEL=google/gemini-2.0-flash-exp:free    # Model identifier
LLM_API_KEY=sk-or-v1-your-key                 # API key
LLM_ENDPOINT=https://openrouter.ai/api/v1     # API endpoint (optional)

# Generation Parameters
LLM_TEMPERATURE=0.7                           # Creativity level (0.0-1.0)
LLM_MAX_TOKENS=4196                           # Maximum response length
LLM_STREAMING=true                            # Enable streaming responses
LLM_TIMEOUT=30                                # Request timeout (seconds)

# System Prompts
LLM_SYSTEM_PROMPT_FILE=prompts/asisten_sederhana_id.txt  # Prompt template file
```

### System Prompts

LocalRAG includes several system prompt templates:

**Indonesian Prompts:**
- `prompts/asisten_sederhana_id.txt` - Simple Indonesian assistant
- `prompts/asisten_rag_umum_id.txt` - General RAG assistant (Indonesian)
- `prompts/asisten_teknis_id.txt` - Technical assistant (Indonesian)
- `prompts/asisten_perusahaan_id.txt` - Corporate assistant (Indonesian)
- `prompts/asisten_penelitian_id.txt` - Research assistant (Indonesian)

**English Prompts:**
- `prompts/general_rag_assistant.txt` - General RAG assistant
- `prompts/technical_assistant.txt` - Technical assistant
- `prompts/simple_assistant.txt` - Simple assistant

**Custom System Prompt:**
```bash
# Use custom prompt directly
LLM_SYSTEM_PROMPT="You are a helpful AI assistant specializing in Indonesian government regulations..."

# Or use file-based prompt
LLM_SYSTEM_PROMPT_FILE=prompts/custom_assistant.txt
```

## OpenRouter Setup (Detailed)

### 1. Create OpenRouter Account

```bash
# Visit https://openrouter.ai/
# Sign up and get API key
# Copy API key starting with 'sk-or-v1-...'
```

### 2. Configure Environment

```bash
# Add to .env file
LLM_PROVIDER=openai
LLM_ENDPOINT=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-v1-your-actual-key-here
LLM_MODEL=google/gemini-2.0-flash-exp:free
LLM_STREAMING=true
LLM_TIMEOUT=30
```

### 3. Test Connection

```bash
# Test OpenRouter connection
curl -H "Authorization: Bearer $LLM_API_KEY" \
     -H "Content-Type: application/json" \
     https://openrouter.ai/api/v1/models

# Test LocalRAG LLM service
curl http://localhost:8080/api/v1/health/llm
```

### 4. Model Selection

**Free Models:**
```bash
LLM_MODEL=google/gemini-2.0-flash-exp:free     # Best free option
LLM_MODEL=meta-llama/llama-3.2-3b-instruct:free
LLM_MODEL=microsoft/phi-3-mini-128k-instruct:free
```

**Paid Models (Better Quality):**
```bash
LLM_MODEL=google/gemini-pro                    # Google Gemini Pro
LLM_MODEL=openai/gpt-4                         # OpenAI GPT-4
LLM_MODEL=anthropic/claude-3-sonnet            # Anthropic
```

## Usage Examples

### Basic RAG Query with LLM

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key requirements in Permendagri 90/2019?",
    "limit": 5,
    "use_llm": true
  }' \
  http://localhost:8080/api/v1/rag/query
```

**Response with LLM Generation:**
```json
{
  "query": "What are the key requirements in Permendagri 90/2019?",
  "response": "Based on the retrieved documents, Permendagri 90/2019 establishes several key requirements:\n\n1. **Classification Standards**: All regional programs must follow the standardized classification system...",
  "context_chunks": 3,
  "metadata": {
    "generation_method": "llm_generation",
    "llm_provider": "openai",
    "llm_model": "google/gemini-2.0-flash-exp:free",
    "llm_tokens_used": 245,
    "enhanced_similarity_used": true,
    "fallback_used": false
  }
}
```

### Chat Interface with LLM

```bash
# Create chat session
SESSION_ID=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123"}' \
  http://localhost:8080/api/v1/chat/session | jq -r '.session_id')

# Send message with RAG
curl -X POST \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION_ID\",
    \"message\": \"Explain the classification system in Indonesian regulations\",
    \"use_rag\": true,
    \"rag_limit\": 5
  }" \
  http://localhost:8080/api/v1/chat/message
```

### Streaming Responses

```bash
# Enable streaming in configuration
LLM_STREAMING=true

# Use Server-Sent Events endpoint
curl -N -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "message": "Tell me about Indonesian government document classification",
    "use_rag": true
  }' \
  http://localhost:8080/api/v1/chat/stream
```

## RAG-Enhanced Generation

### Context Integration

LocalRAG automatically integrates retrieved context with LLM prompts:

```python
# Automatic context building
def _build_chat_prompt(user_message, context_chunks):
    context_parts = ["Relevant information from documents:", ""]
    
    for i, chunk in enumerate(context_chunks[:5], 1):
        context_parts.append(f"[Source {i}] (Relevance: {chunk['similarity']:.1%})")
        context_parts.append(chunk["content"])
        context_parts.append("")
    
    prompt = f"""
{context_text}

User: {user_message}

Instructions:
- Provide a helpful response based primarily on the context above
- If context doesn't contain enough information, acknowledge this clearly
- Be specific and reference relevant information when appropriate
- Keep response natural and engaging

    Assistant: """
    
    return prompt
```

### Quality Enhancement

**Response Validation:**
```python
# Automatic response validation
validation_result = await hallucination_reducer.validate_response(
    response=llm_response,
    context_chunks=context_chunks
)

if validation_result["confidence"] < 0.7:
    # Add disclaimer or use extractive fallback
    enhanced_response = f"{llm_response}\n\n[Note: Response may have limited accuracy]"
```

**Fallback Strategy:**
```python
# Intelligent fallback mechanism
try:
    llm_response = await llm_service.generate_response(query, context_chunks)
    generation_method = "llm_generation"
except Exception as llm_error:
    logger.warning(f"LLM generation failed: {llm_error}")
    if settings.llm_fallback_to_extractive:
        extractive_response = await self._generate_extractive_response(query, context_chunks)
        generation_method = "extractive_fallback"
```

## Performance Optimization

### Token Management

```bash
# Optimize token usage
LLM_MAX_TOKENS=4196                    # Balance between quality and cost
LLM_TEMPERATURE=0.7                    # Balance creativity and consistency

# Context optimization
TOP_K=5                                # Limit context chunks for token efficiency
MAX_CHUNK_CHARACTERS=7000              # Control individual chunk size
```

### Request Optimization

```bash
# Timeout and retry settings
LLM_TIMEOUT=30                         # Prevent hanging requests
LLM_RETRY_ATTEMPTS=3                   # Automatic retry on failures
LLM_RETRY_DELAY=1                      # Delay between retries (seconds)

# Concurrent request limits
LLM_MAX_CONCURRENT_REQUESTS=5          # Prevent API rate limiting
```

### Model Selection Guidelines

**For Cost-Sensitive Applications:**
```bash
LLM_MODEL=google/gemini-2.0-flash-exp:free     # Free tier
LLM_MAX_TOKENS=2048                            # Reduce token usage
TOP_K=3                                        # Fewer context chunks
```

**For Quality-Critical Applications:**
```bash
LLM_MODEL=openai/gpt-4                         # Premium model
LLM_MAX_TOKENS=8192                            # Longer responses
TOP_K=8                                        # More context
LLM_TEMPERATURE=0.3                            # More consistent responses
```

## Monitoring and Debugging

### Health Monitoring

```bash
# Check LLM service status
curl http://localhost:8080/api/v1/health/llm

# Monitor LLM usage
curl http://localhost:8080/api/v1/metrics/llm
```

**Health Response:**
```json
{
  "llm_service": {
    "enabled": true,
    "provider": "openai",
    "model": "google/gemini-2.0-flash-exp:free",
    "endpoint": "https://openrouter.ai/api/v1",
    "available": true,
    "streaming_enabled": true,
    "fallback_enabled": true,
    "last_successful_request": "2025-07-04T15:30:00Z",
    "error_rate_24h": 0.02,
    "avg_response_time": "1.2s"
  }
}
```

### Logging

**Key Log Messages:**
```bash
# LLM service initialization
"LLM service initialized with openai provider"

# Request processing
"Generated response using LLM (openai)"
"LLM generation failed: API error, falling back to extractive"

# Performance metrics
"LLM response generated in 1.2s with 245 tokens"
```

### Debug Mode

```bash
# Enable detailed LLM debugging
DEBUG=true
LOG_LEVEL=DEBUG

# View LLM request/response details
tail -f logs/localrag.log | grep "LLM"
```

## Troubleshooting

### Common Issues

**1. LLM Service Not Available**
```bash
# Check configuration
grep LLM_ .env

# Verify API key
echo "API Key: ${LLM_API_KEY:0:10}..."

# Test provider connection
curl -H "Authorization: Bearer $LLM_API_KEY" $LLM_ENDPOINT/models
```

**2. High Error Rates**
```bash
# Check rate limiting
curl -H "Authorization: Bearer $LLM_API_KEY" \
     https://openrouter.ai/api/v1/auth/key

# Monitor API quotas
curl -H "Authorization: Bearer $LLM_API_KEY" \
     https://openrouter.ai/api/v1/generation
```

**3. Slow Response Times**
```bash
# Reduce token limits
LLM_MAX_TOKENS=2048

# Decrease context chunks
TOP_K=3

# Enable response caching
ENABLE_LLM_CACHING=true
CACHE_TTL=3600
```

**4. Poor Response Quality**
```bash
# Increase context relevance
SIMILARITY_THRESHOLD=0.6

# Improve prompt engineering
LLM_SYSTEM_PROMPT_FILE=prompts/technical_assistant.txt

# Use higher quality model
LLM_MODEL=openai/gpt-4
```

### Error Codes

**LLM-Specific Error Codes:**
- `LLM_UNAVAILABLE` - Service unavailable, fallback used
- `LLM_PROVIDER_ERROR` - Provider API error
- `LLM_TIMEOUT` - Request timeout exceeded
- `LLM_RATE_LIMITED` - API rate limit exceeded
- `LLM_INVALID_API_KEY` - Authentication failed
- `LLM_MODEL_NOT_FOUND` - Specified model unavailable

## Advanced Configuration

### Custom Providers

**Adding New Provider:**
```python
# In core/llm_service.py
class CustomProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, endpoint: str):
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        # Custom implementation
        pass
```

### Prompt Engineering

**Advanced System Prompts:**
```bash
# Role-specific prompts
LLM_SYSTEM_PROMPT="You are an Indonesian government regulation expert..."

# Context-aware prompts
LLM_SYSTEM_PROMPT="Given the following Indonesian regulation documents..."

# Task-specific prompts
LLM_SYSTEM_PROMPT="Extract key information from Indonesian legal documents..."
```

### Multi-Model Setup

**Load Balancing:**
```bash
# Primary provider
LLM_PROVIDER=openai
LLM_MODEL=google/gemini-2.0-flash-exp:free

# Fallback provider
LLM_FALLBACK_PROVIDER=ollama
LLM_FALLBACK_MODEL=llama2:13b
LLM_FALLBACK_ENDPOINT=http://localhost:11434
```

## Best Practices

### Cost Optimization

1. **Use Free Models for Development:**
   ```bash
   LLM_MODEL=google/gemini-2.0-flash-exp:free
   ```

2. **Optimize Context Length:**
   ```bash
   TOP_K=3                            # Fewer chunks
   CHUNK_SIZE=256                     # Smaller chunks
   ```

3. **Enable Caching:**
   ```bash
   ENABLE_LLM_CACHING=true
   CACHE_TTL=3600
   ```

### Quality Optimization

1. **Use Premium Models for Production:**
   ```bash
   LLM_MODEL=openai/gpt-4
   LLM_TEMPERATURE=0.3
   ```

2. **Enhance Context Relevance:**
   ```bash
   SIMILARITY_THRESHOLD=0.6
   ENABLE_ENHANCED_SIMILARITY=true
   ```

3. **Custom System Prompts:**
   ```bash
   LLM_SYSTEM_PROMPT_FILE=prompts/domain_expert.txt
   ```

### Security

1. **Protect API Keys:**
   ```bash
   # Use environment variables
   export LLM_API_KEY="sk-or-v1-..."
   
   # Rotate keys regularly
   # Monitor usage for anomalies
   ```

2. **Input Validation:**
   ```bash
   # Enable input sanitization
   ENABLE_INPUT_VALIDATION=true
   MAX_QUERY_LENGTH=1000
   ```

3. **Rate Limiting:**
   ```bash
   # API-level rate limiting
   LLM_RATE_LIMIT_PER_MINUTE=60
   LLM_MAX_CONCURRENT_REQUESTS=5
   ```

For more information on Indonesian-specific processing, see [INDONESIAN_PROCESSING.md](INDONESIAN_PROCESSING.md).
For general configuration options, see [CONFIG.md](CONFIG.md).