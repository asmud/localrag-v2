# Enhanced OCR Configuration

This document describes the enhanced OCR capabilities in LocalRAG, designed to improve text extraction quality from Indonesian government documents.

## Overview

LocalRAG now supports multiple OCR engines with Indonesian language optimization:

- **EasyOCR** (recommended): Best accuracy for Indonesian documents
- **PaddleOCR**: Alternative engine with good performance
- **Tesseract**: Traditional OCR with Indonesian language support
- **MarkItDown**: Built-in engine (fallback)

## Configuration

### Environment Variables

Add these settings to your `.env` file:

```env
# OCR Engine Selection
OCR_ENGINE=easyocr                    # Primary engine: markitdown|easyocr|paddleocr|tesseract
OCR_FALLBACK_ENGINE=tesseract         # Fallback engine if primary fails

# Language Support
OCR_LANGUAGES=["en", "id"]            # English and Indonesian

# Quality Settings
OCR_CONFIDENCE_THRESHOLD=0.6          # Minimum confidence score (0.0-1.0)
OCR_ENABLE_PREPROCESSING=true         # Enable image enhancement
OCR_ENABLE_POSTPROCESSING=true       # Enable Indonesian text correction

# Performance Settings
OCR_GPU_ENABLED=true                  # Use GPU acceleration if available
OCR_DPI_THRESHOLD=300                 # Minimum DPI for processing
OCR_MAX_IMAGE_SIZE=4096              # Maximum image dimension
```

### Recommended Configurations

#### For Indonesian Government Documents
```env
OCR_ENGINE=easyocr
OCR_FALLBACK_ENGINE=tesseract
OCR_LANGUAGES=["en", "id"]
OCR_CONFIDENCE_THRESHOLD=0.6
OCR_ENABLE_PREPROCESSING=true
OCR_ENABLE_POSTPROCESSING=true
```

#### For High-Performance Systems (with GPU)
```env
OCR_ENGINE=easyocr
OCR_GPU_ENABLED=true
OCR_CONFIDENCE_THRESHOLD=0.7
OCR_DPI_THRESHOLD=400
```

#### For CPU-Only Systems
```env
OCR_ENGINE=tesseract
OCR_FALLBACK_ENGINE=markitdown
OCR_GPU_ENABLED=false
OCR_CONFIDENCE_THRESHOLD=0.5
```

## Installation

### Automatic Setup
```bash
python utils/setup_ocr.py
```

### Manual Installation
```bash
# Install Python dependencies
pip install easyocr>=1.7.0 paddlepaddle>=2.6.0 paddleocr>=2.7.0 pytesseract>=0.3.10

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr tesseract-ocr-ind

# Install system dependencies (macOS)
brew install tesseract tesseract-lang
```

### Docker Installation
Add to your Dockerfile:
```dockerfile
# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ind \
    libgl1-mesa-glx \
    libglib2.0-0

# Python dependencies are handled by pyproject.toml
```

## Features

### Image Preprocessing
- **DPI Enhancement**: Automatically upscale low-resolution images
- **Contrast Enhancement**: Improve text clarity
- **Denoising**: Remove image artifacts
- **Format Normalization**: Convert to optimal format for OCR

### Indonesian Text Processing
- **Common OCR Error Correction**: Fix typical character misrecognitions
- **Spacing Normalization**: Correct spacing issues
- **Quality Validation**: Assess text quality using Indonesian language patterns
- **Corruption Detection**: Identify and filter garbled text patterns

### Multi-Engine Fallback
- **Primary Engine**: Use configured OCR engine first
- **Automatic Fallback**: Switch to backup engine if primary fails
- **Confidence-Based Selection**: Choose best result based on confidence scores
- **Error Recovery**: Graceful handling of OCR failures

## Testing

### Basic OCR Test
```bash
python utils/test_ocr.py
```

### Engine Comparison
```bash
# Test all available engines on sample documents
python utils/test_ocr.py --compare
```

### Document Processing Test
```bash
# Test with your documents
python -c "
from core.ingestion import DocumentProcessor
import asyncio

async def test():
    processor = DocumentProcessor()
    result = await processor.process_document('path/to/your/document.pdf')
    print(f'Processed with: {result[\"metadata\"][\"processed_with\"]}')
    print(f'Confidence: {result[\"metadata\"][\"ocr_confidence\"]}')

asyncio.run(test())
"
```

## Performance Optimization

### GPU Acceleration
EasyOCR and PaddleOCR support GPU acceleration:
```env
OCR_GPU_ENABLED=true
```

Requires:
- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- PyTorch with CUDA support

### Memory Optimization
For large documents:
```env
OCR_MAX_IMAGE_SIZE=2048          # Reduce for memory constraints
OCR_ENABLE_PREPROCESSING=false   # Disable if processing speed is critical
```

### Batch Processing
For multiple documents:
```python
from core.enhanced_ocr import enhanced_ocr_engine

# Process multiple files efficiently
for file_path in document_paths:
    result = enhanced_ocr_engine.extract_text_with_fallback(file_path)
    # Process result...
```

## Troubleshooting

### Common Issues

#### "EasyOCR not available"
```bash
pip install easyocr>=1.7.0
```

#### "Tesseract not found"
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-ind

# macOS
brew install tesseract tesseract-lang
```

#### "CUDA out of memory"
```env
OCR_GPU_ENABLED=false
# or reduce batch size/image size
```

#### Low OCR Quality
1. Check image resolution (should be ≥300 DPI)
2. Enable preprocessing: `OCR_ENABLE_PREPROCESSING=true`
3. Try different engines: `OCR_ENGINE=easyocr`
4. Lower confidence threshold: `OCR_CONFIDENCE_THRESHOLD=0.4`

### Logging and Debugging

Enable detailed OCR logging:
```python
import logging
logging.getLogger('core.enhanced_ocr').setLevel(logging.DEBUG)
```

Check OCR results:
```python
from core.enhanced_ocr import enhanced_ocr_engine

result = enhanced_ocr_engine.extract_text_with_fallback('document.pdf')
print(f"Engine: {result.engine}")
print(f"Confidence: {result.confidence}")
print(f"Metadata: {result.metadata}")
```

## Integration with Existing Documents

### Re-processing Corrupted Documents
If you have documents with garbled text (like "K B U S N A T A I G E K" patterns):

1. Update OCR configuration
2. Re-ingest the problematic documents
3. The enhanced OCR will produce cleaner text

### Monitoring OCR Quality
The system now tracks OCR confidence and engine used:
```sql
-- Check OCR quality in database
SELECT 
    file_name,
    metadata->>'processed_with' as ocr_engine,
    metadata->>'ocr_confidence' as confidence
FROM documents 
WHERE metadata->>'ocr_confidence' IS NOT NULL
ORDER BY (metadata->>'ocr_confidence')::float ASC;
```

## Best Practices

1. **Use EasyOCR for Indonesian documents** - highest accuracy
2. **Enable preprocessing** - improves quality significantly  
3. **Set appropriate confidence thresholds** - balance quality vs. recall
4. **Monitor OCR results** - track confidence scores and quality
5. **Test with your specific document types** - different formats may need different settings
6. **Use GPU acceleration** - if available, significantly improves speed
7. **Keep fallback engines configured** - ensures robust processing

## Future Enhancements

Planned improvements:
- Custom Indonesian OCR model training
- Document-type specific optimization
- Advanced corruption detection
- Quality-based re-processing workflows
- OCR result caching
- Performance metrics and monitoring