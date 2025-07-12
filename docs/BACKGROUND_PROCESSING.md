# Background Processing

LocalRAG features a comprehensive async background processing system that enables non-blocking document uploads, batch processing, and real-time job tracking. This system is designed for enterprise-scale document processing with robust error handling and monitoring capabilities.

## Overview

The background processing system consists of:

- **Redis Task Queue**: Priority-based job queuing with retry logic
- **PostgreSQL Job Tracking**: Persistent job status and progress tracking
- **Multi-Worker Processing**: Configurable high-priority and general workers
- **Auto-Start Workers**: Background workers start automatically with Docker
- **Real-time Monitoring**: Job progress tracking and worker status monitoring

## Features

### ⚡ Non-Blocking Document Processing
- Upload returns immediately with job ID
- No timeouts on large PDF processing
- Real-time progress updates via API

### 📈 Priority-Based Processing
- **High Priority**: Single document uploads (immediate processing)
- **Medium Priority**: Directory batch processing
- **Low Priority**: Background maintenance tasks

### 🔄 Intelligent Retry Logic
- Automatic retry for failed jobs (configurable attempts)
- Exponential backoff delay between retries
- Permanent failure after max attempts exceeded

### 🏗️ Enterprise Architecture
- Horizontal scaling with multiple workers
- Graceful shutdown handling
- Resource management and monitoring
- Job persistence across application restarts

## Configuration

### Environment Variables

```bash
# Enable/disable background processing
TASK_QUEUE_ENABLED=true

# Auto-start workers with application (recommended)
TASK_WORKERS_AUTO_START=true

# Worker configuration
TASK_MAX_WORKERS=4                    # Total number of workers
TASK_HIGH_PRIORITY_WORKERS=2          # Workers dedicated to high-priority jobs
TASK_BATCH_SIZE=10                    # Files per batch in directory processing

# Job management
TASK_MAX_RETRIES=3                    # Maximum retry attempts
TASK_RETRY_DELAY=60                   # Delay between retries (seconds)
TASK_JOB_RETENTION=86400              # Job data retention (24 hours)
TASK_PROGRESS_UPDATE_INTERVAL=5       # Progress update frequency (seconds)

# Queue configuration
TASK_QUEUE_NAME=localrag_tasks        # Redis queue name prefix
```

### Redis Configuration

Background processing requires Redis for task queuing:

```bash
# Redis connection (used for both caching and task queue)
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_URL=redis://host.docker.internal:6379/0
```

## API Endpoints

### Document Upload (Async)

```bash
# Upload document - returns immediately with job ID
curl -X POST -F "file=@document.pdf" \
  http://localhost:8080/api/v1/documents/upload

# Response:
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "file_path": "/app/data/uploads/document.pdf",
  "status": "queued",
  "message": "Document 'document.pdf' queued for processing..."
}
```

### Directory Processing (Batch)

```bash
# Process entire directory in background
curl -X POST -H "Content-Type: application/json" \
  -d '{"directory_path": "/app/data/uploads", "file_patterns": ["*.pdf", "*.docx"]}' \
  http://localhost:8080/api/v1/documents/ingest-directory

# Response:
{
  "job_id": "550e8400-e29b-41d4-a716-446655440001",
  "directory_path": "/app/data/uploads",
  "total_files_found": 25,
  "status": "queued",
  "message": "Directory processing queued. Found 25 files..."
}
```

### Job Monitoring

```bash
# Get job status
curl http://localhost:8080/api/v1/jobs/{job_id}

# Get real-time progress
curl http://localhost:8080/api/v1/jobs/{job_id}/progress

# Get job result (when completed)
curl http://localhost:8080/api/v1/jobs/{job_id}/result

# List recent jobs
curl http://localhost:8080/api/v1/jobs?limit=20

# Filter by status
curl http://localhost:8080/api/v1/jobs?status=processing&limit=10
```

### Worker Management

```bash
# Get worker status
curl http://localhost:8080/api/v1/workers

# Worker health check
curl http://localhost:8080/api/v1/health/workers

# Start workers (if not auto-started)
curl -X POST http://localhost:8080/api/v1/workers/start

# Stop workers
curl -X POST http://localhost:8080/api/v1/workers/stop
```

### Job Control

```bash
# Cancel a job
curl -X POST http://localhost:8080/api/v1/jobs/{job_id}/cancel

# Retry a failed job
curl -X POST http://localhost:8080/api/v1/jobs/{job_id}/retry

# Clean up old jobs
curl -X DELETE "http://localhost:8080/api/v1/jobs/cleanup?days_old=7"

# Get processing statistics
curl http://localhost:8080/api/v1/jobs/statistics
```

## CLI Commands

### Worker Management

```bash
# Start background workers
python cli.py workers start

# Check worker status
python cli.py workers status

# Example output:
# 👷 Background Worker Status:
# ==============================
# ✅ high-priority-1 (high)
#    Processed: 45
#    Failed: 2
#    Current job: 550e8400-e29b-41d4-a716-446655440000
# 
# ✅ high-priority-2 (high)
#    Processed: 38
#    Failed: 1
# 
# ✅ general-1 (any)
#    Processed: 123
#    Failed: 5
# 
# ✅ general-2 (any)
#    Processed: 98
#    Failed: 3
```

### Job Management

```bash
# List recent jobs
python cli.py jobs list

# List jobs by status
python cli.py jobs list --status processing --limit 10

# Check specific job
python cli.py jobs status {job_id}

# Cancel a job
python cli.py jobs cancel {job_id}

# Clean up old jobs
python cli.py jobs cleanup --days 7
```

## Job States and Lifecycle

### Job States

- **`pending`**: Job queued, waiting for worker
- **`processing`**: Worker actively processing the job
- **`completed`**: Job finished successfully
- **`failed`**: Job failed after all retry attempts
- **`cancelled`**: Job cancelled by user or system
- **`retrying`**: Job failed, queued for retry

### Job Lifecycle

```
Upload/Queue → pending → processing → completed
                ↓           ↓
              cancelled   failed → retrying → processing
                            ↓
                         failed (permanent)
```

### Progress Tracking

Jobs report progress from 0.0 to 100.0 with descriptive messages:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "progress": 65.5,
  "message": "Processing page 13 of 20",
  "updated_at": "2024-01-15T10:30:45Z"
}
```

## Job Types

### Single Document (`single_document`)

- **Priority**: High
- **Processing**: Page-by-page PDF analysis with smart engine selection
- **Progress**: Real-time updates during OCR and embedding generation
- **Result**: Document ID, chunk count, processing metadata

### Directory Batch (`directory_batch`)

- **Priority**: Medium
- **Processing**: Batch processing with configurable batch size
- **Progress**: File-by-file progress updates
- **Result**: Summary of processed files, success/failure counts

### Document Reprocess (`document_reprocess`)

- **Priority**: Medium
- **Processing**: Re-process existing document with updated settings
- **Progress**: Similar to single document
- **Result**: Updated document data

## Monitoring and Observability

### Health Checks

```bash
# Overall system health (includes workers)
curl http://localhost:8080/api/v1/health

# Dedicated worker health check
curl http://localhost:8080/api/v1/health/workers

# Example response:
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
  }
}
```

### Job Statistics

```bash
curl http://localhost:8080/api/v1/jobs/statistics

# Response includes:
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
    "queue_lengths": {...},
    "total_pending": 6
  }
}
```

## Database Schema

The background processing system creates these tables:

### `processing_jobs`

Stores job metadata and tracking information:

```sql
CREATE TABLE processing_jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(36) UNIQUE NOT NULL,
    task_type VARCHAR(50) NOT NULL,
    priority VARCHAR(10) NOT NULL,
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    progress FLOAT DEFAULT 0.0,
    error_message TEXT NULL,
    retry_count INTEGER DEFAULT 0,
    payload JSONB NOT NULL,
    result JSONB NULL,
    worker_id VARCHAR(50) NULL,
    actual_duration INTEGER NULL
);
```

### `job_statistics`

Aggregated daily statistics:

```sql
CREATE TABLE job_statistics (
    id SERIAL PRIMARY KEY,
    date DATE DEFAULT CURRENT_DATE,
    total_jobs INTEGER DEFAULT 0,
    completed_jobs INTEGER DEFAULT 0,
    failed_jobs INTEGER DEFAULT 0,
    avg_processing_time FLOAT DEFAULT 0.0,
    -- ... additional metrics
);
```

## Troubleshooting

### Workers Not Starting

1. **Check Configuration**:
   ```bash
   # Verify settings
   curl http://localhost:8080/api/v1/health/workers
   ```

2. **Check Redis Connection**:
   ```bash
   # Test Redis connectivity
   docker exec localrag redis-cli ping
   ```

3. **Check Logs**:
   ```bash
   # Docker logs
   docker logs localrag
   
   # Look for worker startup messages
   grep -i "worker" logs/localrag.log
   ```

### Jobs Stuck in Processing

1. **Check Worker Status**:
   ```bash
   python cli.py workers status
   ```

2. **Restart Workers**:
   ```bash
   curl -X POST http://localhost:8080/api/v1/workers/stop
   curl -X POST http://localhost:8080/api/v1/workers/start
   ```

3. **Cancel and Retry**:
   ```bash
   python cli.py jobs cancel {job_id}
   # Re-upload the document
   ```

### High Memory Usage

1. **Reduce Worker Count**:
   ```bash
   # In .env
   TASK_MAX_WORKERS=2
   TASK_HIGH_PRIORITY_WORKERS=1
   ```

2. **Reduce Batch Size**:
   ```bash
   # In .env
   TASK_BATCH_SIZE=5
   ```

3. **Monitor Resource Usage**:
   ```bash
   docker stats localrag
   ```

### Failed Jobs

1. **Check Error Messages**:
   ```bash
   python cli.py jobs status {job_id}
   curl http://localhost:8080/api/v1/jobs/{job_id}
   ```

2. **Common Issues**:
   - **File not found**: File was moved/deleted after upload
   - **OCR timeout**: Large image files, increase timeout settings
   - **Memory issues**: Reduce concurrent workers or batch size
   - **Database connection**: Check PostgreSQL connectivity

3. **Retry Jobs**:
   ```bash
   # Retry specific job
   curl -X POST http://localhost:8080/api/v1/jobs/{job_id}/retry
   
   # Or via CLI
   python cli.py jobs list --status failed
   # Then retry individual jobs
   ```

## Performance Optimization

### Worker Configuration

```bash
# For CPU-intensive OCR processing
TASK_MAX_WORKERS=2  # Match CPU cores
TASK_HIGH_PRIORITY_WORKERS=1

# For I/O-intensive processing
TASK_MAX_WORKERS=6  # Higher than CPU cores
TASK_HIGH_PRIORITY_WORKERS=2
```

### Batch Processing

```bash
# Large files - smaller batches
TASK_BATCH_SIZE=5

# Small files - larger batches
TASK_BATCH_SIZE=20
```

### Resource Limits

```bash
# Docker Compose resource limits
services:
  localrag:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

## Best Practices

### Production Deployment

1. **Enable Auto-Start**: `TASK_WORKERS_AUTO_START=true`
2. **Set Appropriate Worker Count**: Match your server resources
3. **Configure Job Retention**: Balance storage vs. audit requirements
4. **Monitor Queue Depth**: Alert on excessive pending jobs
5. **Regular Cleanup**: Schedule old job cleanup

### Development

1. **Disable Auto-Start**: `TASK_WORKERS_AUTO_START=false` for debugging
2. **Start Workers Manually**: `python cli.py workers start`
3. **Monitor Logs**: Real-time worker activity
4. **Test with Small Batches**: Verify processing before large uploads

### Monitoring

1. **Health Checks**: Regular `/health/workers` monitoring
2. **Job Statistics**: Track processing performance trends
3. **Queue Depth**: Monitor pending job counts
4. **Worker Performance**: Track processed vs. failed job ratios

## Integration Examples

### Upload with Progress Tracking

```python
import requests
import time

# Upload document
response = requests.post(
    "http://localhost:8080/api/v1/documents/upload",
    files={"file": open("document.pdf", "rb")}
)
job_id = response.json()["job_id"]

# Track progress
while True:
    progress = requests.get(f"http://localhost:8080/api/v1/jobs/{job_id}/progress")
    data = progress.json()
    
    print(f"Progress: {data['progress']:.1f}% - {data['message']}")
    
    if data["progress"] >= 100:
        break
    
    time.sleep(2)

# Get final result
result = requests.get(f"http://localhost:8080/api/v1/jobs/{job_id}/result")
print("Processing complete:", result.json())
```

### Batch Processing with Monitoring

```python
import requests

# Start directory processing
response = requests.post(
    "http://localhost:8080/api/v1/documents/ingest-directory",
    json={"directory_path": "/app/data/uploads"}
)
job_id = response.json()["job_id"]

# Monitor batch progress
def monitor_batch_job(job_id):
    while True:
        status = requests.get(f"http://localhost:8080/api/v1/jobs/{job_id}")
        job_data = status.json()
        
        print(f"Status: {job_data['status']} - Progress: {job_data['progress']:.1f}%")
        
        if job_data["status"] in ["completed", "failed", "cancelled"]:
            break
        
        time.sleep(5)
    
    # Get final results
    if job_data["status"] == "completed":
        result = requests.get(f"http://localhost:8080/api/v1/jobs/{job_id}/result")
        batch_result = result.json()["result"]
        print(f"Processed {batch_result['successful']}/{batch_result['total_files']} files")

monitor_batch_job(job_id)
```