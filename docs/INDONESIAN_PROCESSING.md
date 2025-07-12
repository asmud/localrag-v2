# Indonesian Document Processing

Comprehensive guide to LocalRAG's specialized Indonesian document processing capabilities, optimized for government regulations and OCR-processed PDFs.

## Overview

LocalRAG includes advanced Indonesian language processing features specifically designed for:

- **Government Regulations** (Permendagri, Permen, Peraturan Daerah)
- **OCR-processed PDFs** with corruption detection and filtering
- **Complex Document Structures** with semantic-aware chunking
- **Regulation Pattern Recognition** for enhanced retrieval accuracy

## Key Features

### 🔍 **Intelligent Corruption Detection**

Automatically detects and filters garbled text patterns common in OCR-processed Indonesian documents:

```
FILTERED: "U M A R G O R P N A T A I G E K B U S N A T A I G E K"
CLEANED:  "PROGRAM KEGIATAN SUBKEGIATAN"
```

### 📊 **Semantic-Aware Chunking**

Document structure recognition with priority-based processing:

- **Header Priority**: Document titles and introductions get 1.3x similarity boost
- **Section Awareness**: Recognizes document structure (header, content, appendix)
- **Regulation Patterns**: Detects formal regulation references and terminology

### 🎯 **Enhanced Similarity Scoring**

Multi-factor scoring system optimized for Indonesian content:

- **Document Position**: Earlier chunks prioritized over appendices
- **Regulation References**: Exact regulation number/year matching
- **Phrase Matching**: Indonesian synonym recognition and expansion
- **Content Type**: Structural elements (headings, lists) get appropriate weighting

## Configuration

### Environment Variables

```bash
# Indonesian Language Processing
EXTRACTION_MODEL=asmud/cahya-indonesian-ner-tuned   # Indonesian BERT model
ENABLE_CORRUPTION_DETECTION=true                # Filter garbled text
ENABLE_REGULATION_PATTERN_DETECTION=true        # Detect regulation patterns
ENABLE_SEMANTIC_CHUNKING=true                   # Structure-aware chunking

# Optimized RAG Parameters
CHUNK_SIZE=256                                   # Optimized for Indonesian regulations
CHUNK_OVERLAP=64                                 # Balanced overlap
SIMILARITY_THRESHOLD=0.4                        # Lowered for Indonesian complexity
TOP_K=5                                          # Quality over quantity
HEADER_PRIORITY_BOOST=1.3                       # Boost document headers

# System Prompts
LLM_SYSTEM_PROMPT_FILE=prompts/asisten_sederhana_id.txt  # Indonesian assistant
```

### Model Requirements

**Required Models:**
- `intfloat/multilingual-e5-base` - Multilingual embeddings
- `asmud/cahya-indonesian-ner-tuned` - Indonesian BERT base model

**Alternative Indonesian Models:**
- `asmud/cahya-indonesian-ner-tuned` - Base Indonesian BERT
- `cahya/bert-base-indonesian-522M` - Large Indonesian BERT

## Document Processing Pipeline

### 1. Upload and Detection

```bash
curl -X POST \
  -F "file=@Permendagri_Nomor_90_Tahun_2019.pdf" \
  -F "metadata={\"category\": \"regulation\", \"language\": \"indonesian\"}" \
  http://localhost:8080/api/v1/documents/upload
```

The system automatically:
- Detects Indonesian content using language indicators
- Applies specialized Indonesian processing pipeline
- Creates high-priority header chunks from document beginning

### 2. Text Extraction and Cleaning

**MarkItDown Extraction:**
```python
# PDF → Text conversion using MarkItDown
raw_text = markitdown.convert(pdf_file)

# Indonesian text validation
if self._is_indonesian_content(raw_text):
    # Apply Indonesian-specific processing
    enhanced_processing = True
```

**Built-in OCR Processing:**
```python
# OCR processing is automatically handled by the enhanced_ocr module
# Current configuration from .env:
# OCR_ENGINE=easyocr
# OCR_LANGUAGES=["en","id"]
# OCR_CONFIDENCE_THRESHOLD=0.4
# OCR_ENABLE_POSTPROCESSING=true
```

### 3. Semantic Chunking

**Automatic Document Processing:**
```python
# Document chunking is handled automatically by the ingestion system
# Current configuration from .env:
# CHUNK_SIZE=256
# CHUNK_OVERLAP=64
# MAX_CHUNK_CHARACTERS=7000
# Document structure is automatically detected during processing
```

**Indonesian Government Document Processing:**
```python
# Indonesian regulation patterns are automatically handled
# by the semantic analyzer and Indonesian KG manager
# Text extraction uses: asmud/cahya-indonesian-ner-tuned
# Processing includes automatic entity recognition
```

### 4. Enhanced Similarity Scoring

**Multi-Factor Scoring:**
```python
def _calculate_enhanced_similarity(chunk, query, base_similarity):
    score = base_similarity
    
    # Factor 1: Document structure priority
    if chunk["section_type"] == "document_header":
        score *= 1.3  # Strong boost for headers
    elif chunk["section_type"] == "appendix":
        score *= 0.8  # Reduce appendix priority
    
    # Factor 2: Regulation reference matching
    if self._contains_regulation_reference(content, query):
        score *= 1.25
    
    # Factor 3: Document position (earlier = better)
    if chunk["chunk_index"] <= 5:
        score *= 1.1
    
    # Factor 4: Exact phrase matching
    if self._has_exact_phrase_match(content, query):
        score *= 1.15
    
    return min(score, 1.0)
```

## Query Processing

### Query Enhancement

**Indonesian Regulation Queries:**
```python
# Input query
query = "Apa intisari dari Permendagri nomor 90 tahun 2019?"

# Enhanced query expansion
enhanced = [
    "apa", "bagaimana", "what", "how",           # Question words
    "intisari", "ringkasan", "isi", "pokok",     # Synonym expansion
    "PERATURAN MENTERI DALAM NEGERI",            # Formal patterns
    "NOMOR 90 TAHUN 2019",                       # Regulation reference
    "MENDAGRI 90/2019"                           # Informal reference
]
```

**Pattern Detection:**
```python
# Regulation pattern detection
permendagri_pattern = r'permendagri.*nomor.*(\d+).*tahun.*(\d{4})'
if re.search(permendagri_pattern, query.lower()):
    # Add formal regulation patterns
    enhanced_query += f" PERATURAN MENTERI DALAM NEGERI NOMOR {number} TAHUN {year}"
```

### Retrieval Process

**Indonesian Context Retrieval:**
```python
async def _retrieve_indonesian_context(query, query_vec, limit):
    # 1. Get Indonesian entity context
    entities = await indonesian_kg_manager.stream_indonesian_entities()
    
    # 2. Enhanced embedding with entity terms
    entity_terms = [entity['name'] for entity in entities[:5]]
    enhanced_embedding = await model_manager.get_embeddings([
        f"{query} {' '.join(entity_terms)}"
    ])
    
    # 3. PostgreSQL vector search with enhanced similarity
    chunks = await postgres_storage.find_similar_chunks(
        query_embedding=enhanced_embedding[0],
        similarity_threshold=0.4  # Lowered for Indonesian
    )
    
    # 4. Apply enhanced similarity scoring
    for chunk in chunks:
        enhanced_score = self._calculate_enhanced_similarity(
            chunk, query, chunk["similarity_score"]
        )
        chunk["enhanced_similarity"] = enhanced_score
```

## API Usage Examples

### Basic Indonesian RAG Query

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apa intisari dari Permendagri nomor 90 tahun 2019?",
    "limit": 5,
    "similarity_threshold": 0.4
  }' \
  http://localhost:8080/api/v1/rag/query
```

**Response Features:**
```json
{
  "enhanced_query": "apa intisari ringkasan dari permendagri PERATURAN MENTERI...",
  "metadata": {
    "indonesian_processing": true,
    "regulation_pattern_detected": true,
    "regulation_reference": "90/2019",
    "corruption_filtered_chunks": 15,
    "enhanced_similarity_used": true
  },
  "chunks": [
    {
      "content": "PERATURAN MENTERI DALAM NEGERI...",
      "similarity": 0.89,
      "enhanced_similarity": 0.95,
      "section_type": "document_header",
      "priority_boost_applied": true,
      "regulation_reference_match": true
    }
  ]
}
```

### Complex Regulation Query

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Bagaimana nomenklatur yang benar dalam penyusunan Laporan Program Bansos di Kota Bandung?",
    "limit": 10,
    "similarity_threshold": 0.3,
    "include_metadata": true
  }' \
  http://localhost:8080/api/v1/rag/query
```

### Chat with Indonesian Documents

```bash
# Create session
SESSION_ID=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"user_id": "admin"}' \
  http://localhost:8080/api/v1/chat/session | jq -r '.session_id')

# Send Indonesian message
curl -X POST \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION_ID\",
    \"message\": \"Jelaskan tentang klasifikasi dan kodefikasi dalam Permendagri 90/2019\",
    \"use_rag\": true
  }" \
  http://localhost:8080/api/v1/chat/message
```

## Performance Optimization

### Indonesian-Specific Tuning

**Chunk Size Optimization:**
```bash
# Smaller chunks for Indonesian regulations
CHUNK_SIZE=256          # Better granularity for complex Indonesian text
CHUNK_OVERLAP=64        # Adequate context preservation
MAX_CHUNK_CHARACTERS=7000  # Maximum chunk size limit
```

**Similarity Threshold Tuning:**
```bash
# Lower threshold for Indonesian language complexity
SIMILARITY_THRESHOLD=0.4    # Accommodates Indonesian language nuances
TOP_K=5                     # Focus on quality over quantity
```

**Memory Optimization:**
```bash
# Corruption filtering reduces memory usage
ENABLE_CORRUPTION_DETECTION=true   # Filter out garbage content
CORRUPTION_FILTER_RATIO=0.15       # Filter chunks with >15% corrupted content
```

## Monitoring and Debugging

### Logging Indonesian Processing

**Key Log Messages:**
```bash
# Document detection
"Detected Indonesian content, using enhanced processing"

# Header chunk creation
"Created high-priority header chunk with 669 characters"

# Corruption detection
"Detected garbled text pattern, skipping: U M A R G O R P..."

# Query enhancement
"Enhanced query: 'Apa intisari...' → 'apa intisari ringkasan...'"

# Regulation pattern detection
"Detected regulation pattern: 90/2019"

# Enhanced similarity application
"Enhanced similarity: 0.75 → 0.95 (section: document_header)"
```

### Performance Metrics

**Indonesian Processing Statistics:**
```bash
curl http://localhost:8080/api/v1/health | jq '.indonesian_processing'
```

```json
{
  "indonesian_processing": {
    "corruption_filtered_chunks": 45,
    "regulation_patterns_detected": 12,
    "header_chunks_created": 3,
    "enhanced_similarity_applied": 156,
    "average_processing_time": "2.3s",
    "model_performance": {
      "indonesian_model_loaded": true,
      "embedding_generation_avg": "0.12s",
      "corruption_detection_avg": "0.03s"
    }
  }
}
```

## Troubleshooting

### Common Issues

**1. Indonesian Model Loading Issues**
```bash
# Check model availability
python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('asmud/cahya-indonesian-ner-tuned')
    print('✓ Indonesian model loaded successfully')
except Exception as e:
    print(f'✗ Model loading failed: {e}')
"

# Check Hugging Face token
echo "HF_TOKEN: ${HF_TOKEN:0:10}..."
```

**2. Corruption Detection Not Working**
```bash
# Test corruption detection
python -c "
from core.models import model_manager
test_text = 'U M A R G O R P N A T A I G E K B U S N A T A I G E K'
result = model_manager._sanitize_text_for_embedding(test_text)
print(f'Input: {test_text[:50]}...')
print(f'Output: {result}')
print(f'Corruption detected: {len(result) == 0}')
"
```

**3. Low Similarity Scores**
```bash
# Check similarity threshold
grep SIMILARITY_THRESHOLD .env

# Verify Indonesian query detection
python -c "
from core.rag import rag_engine
query = 'Apa intisari dari Permendagri nomor 90 tahun 2019?'
is_indonesian = rag_engine._is_indonesian_query(query)
print(f'Query: {query}')
print(f'Detected as Indonesian: {is_indonesian}')
"
```

**4. Regulation Pattern Not Detected**
```bash
# Test regulation pattern detection
python -c "
import re
query = 'Apa intisari dari Permendagri nomor 90 tahun 2019?'
pattern = r'permendagri.*nomor.*(\d+).*tahun.*(\d{4})'
match = re.search(pattern, query.lower())
if match:
    print(f'✓ Regulation detected: {match.groups()}')
else:
    print('✗ No regulation pattern found')
"
```

### Performance Tuning

**For High-Volume Indonesian Processing:**
```bash
# Increase chunk processing parallelization
KG_BATCH_SIZE=200

# Optimize memory usage
REDIS_MAXMEMORY=1GB
POSTGRES_POOL_SIZE=15

# Fine-tune corruption detection sensitivity
CORRUPTION_DETECTION_THRESHOLD=0.15
GARBLED_PATTERN_MIN_LENGTH=8
```

**For Accuracy Optimization:**
```bash
# Lower similarity threshold for better recall
SIMILARITY_THRESHOLD=0.3

# Increase context retrieval
TOP_K=8

# Boost header priority more aggressively
HEADER_PRIORITY_BOOST=1.5
```

## Advanced Features

### Custom Indonesian Patterns

**Adding New Regulation Patterns:**
```python
# In core/rag.py - _expand_regulation_patterns method
custom_patterns = {
    'perda': r'peraturan\s*daerah.*nomor.*(\d+).*tahun.*(\d{4})',
    'perbup': r'peraturan\s*bupati.*nomor.*(\d+).*tahun.*(\d{4})',
    'perwali': r'peraturan\s*walikota.*nomor.*(\d+).*tahun.*(\d{4})'
}
```

**Custom Corruption Patterns:**
```python
# In core/models.py - _sanitize_text_for_embedding method
indonesian_specific_patterns = [
    r'(MARGORP|NATAIGEK|BUSNATAI)',           # Reversed common words
    r'([A-Z]\s){7,}[A-Z]?\s*(NOMENKLATUR)',   # Spaced letters + keywords
    r'(URUSAN|PROGRAM)(\s[A-Z]){5,}'          # Keywords + spaced letters
]
```

### Integration with Knowledge Graph

**Indonesian Entity Enhancement:**
```python
# Enhance queries with Indonesian entities
async def enhance_with_indonesian_entities(query):
    entities = await indonesian_kg_manager.get_relevant_entities(query)
    entity_terms = [entity['name'] for entity in entities]
    return f"{query} {' '.join(entity_terms)}"
```

For more advanced configuration and customization, see [LLM_INTEGRATION.md](LLM_INTEGRATION.md) and [CONFIG.md](CONFIG.md).