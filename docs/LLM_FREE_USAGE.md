# LLM-Free Usage Guide

This guide explains how to use LocalRAG's `/rag/query` and `/rag/retrieve` endpoints without requiring an LLM (Large Language Model). The system provides robust extractive summarization as an alternative to LLM-generated responses.

## Overview

LocalRAG supports multiple modes of operation:

1. **Pure Retrieval** (`/rag/retrieve`) - Returns relevant context chunks without response generation
2. **Extractive Summarization** (`/rag/query`) - Generates responses using intelligent text extraction
3. **LLM-Powered Responses** (`/rag/query`) - Uses external LLM for response generation (optional)

## Benefits of LLM-Free Operation

✅ **No API Costs** - No external LLM API calls required  
✅ **Fast Response** - Lower latency without LLM processing  
✅ **Privacy** - All processing remains local  
✅ **Reliability** - No dependency on external LLM services  
✅ **Indonesian Optimized** - Specialized extractive logic for Indonesian documents  

## Configuration Options

### Global Configuration (`.env`)

```env
# Disable LLM entirely
ENABLE_LLM_CHAT=false

# OR use extractive as default
EXTRACTIVE_DEFAULT_MODE=true
ENABLE_LLM_CHAT=true  # Available as fallback if needed

# Extractive quality settings
EXTRACTIVE_MIN_CONFIDENCE=0.3
EXTRACTIVE_MAX_SENTENCES=5
```

### Per-Request Control

Control LLM usage on individual requests using query parameters.

## API Usage Examples

### 1. Pure Document Retrieval

Returns relevant text chunks without any response generation:

```bash
curl -X POST "http://localhost:8080/api/v1/rag/retrieve" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "query=klasifikasi urusan pemerintahan&limit=5"
```

**Response:**
```json
{
  "query": "klasifikasi urusan pemerintahan",
  "context_chunks": [
    {
      "content": "Klasifikasi urusan pemerintahan...",
      "similarity": 0.92,
      "source": "postgresql",
      "file_name": "Permendagri_90_2019.pdf"
    }
  ],
  "count": 5
}
```

### 2. Extractive Summarization (Forced)

Forces extractive mode regardless of global LLM settings:

```bash
curl -X POST "http://localhost:8080/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apa itu klasifikasi urusan pemerintahan?",
    "extractive_only": true,
    "limit": 5
  }'
```

**Response:**
```json
{
  "query": "Apa itu klasifikasi urusan pemerintahan?",
  "response": "Klasifikasi urusan pemerintahan adalah pengelompokan urusan pemerintahan berdasarkan fungsi dan kewenangan. Urusan pemerintahan dibagi menjadi urusan absolut, urusan konkuren, dan urusan umum.",
  "context_chunks": 5,
  "metadata": {
    "generation_method": "extractive_forced",
    "extractive_confidence": 0.87,
    "response_words": 24,
    "content_overlap_ratio": 0.92,
    "query_coverage": 0.75,
    "indonesian_quality": 0.85,
    "llm_available": false,
    "extractive_forced": true
  }
}
```

### 3. Disable LLM Per Request

Disables LLM for this specific request:

```bash
curl -X POST "http://localhost:8080/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "DTSEN implementation guidelines",
    "use_llm": false,
    "limit": 3
  }'
```

### 4. Indonesian Government Document Queries

Optimized for Indonesian regulatory documents:

```bash
curl -X POST "http://localhost:8080/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Bagaimana prosedur kodefikasi belanja daerah?",
    "extractive_only": true
  }'
```

**Response with Indonesian optimization:**
```json
{
  "metadata": {
    "generation_method": "extractive_forced",
    "extractive_confidence": 0.91,
    "indonesian_quality": 0.88,
    "content_overlap_ratio": 0.94,
    "response_sentences": 3,
    "language_detected": "indonesian"
  }
}
```

## Response Metadata Explanation

### Standard Metadata
- `generation_method`: How the response was generated
  - `"extractive_forced"` - Explicitly requested extractive mode
  - `"extractive_requested"` - LLM disabled via `use_llm=false`
  - `"extractive_summarization"` - Default extractive (LLM disabled globally)
  - `"extractive_fallback"` - LLM failed, fell back to extractive

### Extractive-Specific Metrics
- `extractive_confidence`: Overall confidence score (0.0-1.0)
- `response_length`: Character count of response
- `response_words`: Word count of response
- `response_sentences`: Number of sentences in response
- `content_overlap_ratio`: How much response overlaps with source content
- `query_coverage`: How well response addresses the query
- `indonesian_quality`: Quality score for Indonesian text (0.0-1.0)

### Control Flags
- `extractive_forced`: Whether extractive mode was explicitly requested
- `llm_requested`: Value of `use_llm` parameter (true/false/null)
- `llm_available`: Whether LLM service is available

## Quality Assessment

### High-Quality Extractive Responses
Look for these indicators:
- `extractive_confidence` > 0.7
- `content_overlap_ratio` > 0.6
- `query_coverage` > 0.5
- `indonesian_quality` > 0.6 (for Indonesian queries)

### Example High-Quality Response:
```json
{
  "response": "Kodefikasi belanja daerah menggunakan struktur hierarkis dengan 6 digit: urusan (2 digit), program (3 digit), dan kegiatan (3 digit). Setiap belanja harus sesuai dengan Permendagri Nomor 90 Tahun 2019.",
  "metadata": {
    "extractive_confidence": 0.89,
    "content_overlap_ratio": 0.87,
    "query_coverage": 0.82,
    "indonesian_quality": 0.91
  }
}
```

## Indonesian Document Optimization

### Specialized Features for Indonesian Text

1. **Corruption Detection**: Filters out garbled OCR text patterns
2. **Indonesian Word Recognition**: Identifies Indonesian government terminology
3. **Regulation Pattern Matching**: Recognizes legal document structures
4. **Quality Validation**: Assesses response quality using Indonesian language patterns

### Indonesian Query Examples

```bash
# Government regulation query
curl -X POST "http://localhost:8080/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apa fungsi klasifikasi kodefikasi nomenklatur?",
    "extractive_only": true
  }'

# Budget classification query
curl -X POST "http://localhost:8080/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Bagaimana struktur kode belanja modal?",
    "use_llm": false
  }'

# DTSEN portal query
curl -X POST "http://localhost:8080/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Fitur utama Datahub Portal DTSEN",
    "extractive_only": true
  }'
```

## Performance Considerations

### Extractive Mode Advantages
- **Latency**: ~200-500ms vs 2-5s for LLM
- **Throughput**: Higher concurrent requests
- **Consistency**: Deterministic responses
- **Cost**: No API usage costs

### When to Use Extractive vs LLM

**Use Extractive When:**
- ✅ Need fast responses
- ✅ Working with structured documents
- ✅ Want to avoid API costs
- ✅ Need consistent, deterministic outputs
- ✅ Processing Indonesian government documents

**Consider LLM When:**
- ⚡ Need conversational responses
- ⚡ Require synthesis across multiple concepts
- ⚡ Want natural language generation
- ⚡ Need creative or analytical insights

## Troubleshooting

### Low Extractive Confidence

If `extractive_confidence` is consistently low (<0.5):

1. **Check similarity threshold**:
   ```env
   SIMILARITY_THRESHOLD=0.3  # Lower for more results
   ```

2. **Verify document quality**:
   ```bash
   # Check for corrupted content
   curl -X GET "http://localhost:8080/api/v1/rag/config"
   ```

3. **Adjust extractive settings**:
   ```env
   EXTRACTIVE_MIN_CONFIDENCE=0.2
   EXTRACTIVE_MAX_SENTENCES=7
   ```

### No Results Found

If getting "No relevant information found":

1. **Lower similarity threshold**
2. **Check if documents are uploaded**
3. **Verify query language detection**
4. **Review corruption filtering logs**

### Indonesian Text Issues

For Indonesian document processing:

1. **Enable Indonesian language support**:
   ```env
   OCR_LANGUAGES=["en", "id"]
   OCR_ENABLE_POSTPROCESSING=true
   ```

2. **Check Indonesian quality metrics**:
   - Look for `indonesian_quality` in response metadata
   - Values < 0.3 indicate possible corruption

## Best Practices

1. **Use `/rag/retrieve` for exploration** - Understand what content is available
2. **Set appropriate confidence thresholds** - Balance quality vs recall
3. **Monitor extractive metrics** - Track response quality over time
4. **Optimize for your content type** - Adjust settings based on document types
5. **Use extractive for structured queries** - Better for factual, specific questions
6. **Test with representative queries** - Validate performance with real use cases

## Migration from LLM to Extractive

To migrate from LLM-dependent to extractive-only:

1. **Test Current Queries**:
   ```bash
   # Test each query with extractive_only=true
   # Compare quality and coverage
   ```

2. **Adjust Thresholds**:
   ```env
   SIMILARITY_THRESHOLD=0.3
   EXTRACTIVE_MIN_CONFIDENCE=0.4
   ```

3. **Update Configuration**:
   ```env
   ENABLE_LLM_CHAT=false
   # OR
   EXTRACTIVE_DEFAULT_MODE=true
   ```

4. **Monitor Metrics**:
   - Track `extractive_confidence` scores
   - Monitor user satisfaction
   - Adjust settings based on feedback

This approach provides a robust, cost-effective alternative to LLM-based responses while maintaining high quality for Indonesian document processing.