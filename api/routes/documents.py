from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from core.config import settings
from core.task_queue import TaskType, JobPriority, task_queue

router = APIRouter()


class DocumentUploadResponse(BaseModel):
    job_id: str
    file_path: str
    status: str
    message: str
    duplicate_detection_enabled: Optional[bool] = None


class DirectoryIngestRequest(BaseModel):
    directory_path: str
    file_patterns: Optional[List[str]] = None


class DirectoryIngestResponse(BaseModel):
    job_id: str
    directory_path: str
    total_files_found: int
    status: str
    message: str
    duplicate_detection_info: Optional[Dict] = None


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and queue a single document for background processing"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Ensure uploads directory exists
        uploads_dir = Path(settings.uploads_path)
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_path = uploads_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create background job for document processing
        job_payload = {
            "file_path": str(file_path),
            "filename": file.filename,
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "user_id": None  # Could be extracted from auth context
        }
        
        # Queue with high priority for single document uploads
        job_id = await task_queue.enqueue(
            task_type=TaskType.SINGLE_DOCUMENT,
            payload=job_payload,
            priority=JobPriority.HIGH
        )
        
        return DocumentUploadResponse(
            job_id=job_id,
            file_path=str(file_path),
            status="queued",
            message=f"Document '{file.filename}' queued for processing. Use job ID {job_id} to track progress.",
            duplicate_detection_enabled=settings.duplicate_detection_enabled
        )
        
    except Exception as e:
        # Clean up file if it was created
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")


@router.post("/documents/ingest-directory", response_model=DirectoryIngestResponse)
async def ingest_directory(request: DirectoryIngestRequest):
    """Queue directory ingestion for background processing"""
    try:
        directory_path = Path(request.directory_path)
        
        # Validate directory
        if not directory_path.exists() or not directory_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Invalid directory: {request.directory_path}")
        
        # Count files to be processed
        file_patterns = request.file_patterns or ["*.txt", "*.md", "*.pdf", "*.docx", "*.html"]
        files_to_process = []
        for pattern in file_patterns:
            files_to_process.extend(directory_path.glob(f"**/{pattern}"))
        
        if not files_to_process:
            return DirectoryIngestResponse(
                job_id="",
                directory_path=request.directory_path,
                total_files_found=0,
                status="completed",
                message="No files found to process"
            )
        
        # Create background job for directory processing
        job_payload = {
            "directory_path": str(directory_path),
            "file_patterns": file_patterns,
            "batch_size": settings.task_batch_size,
            "user_id": None  # Could be extracted from auth context
        }
        
        # Queue with medium priority for directory batch processing
        job_id = await task_queue.enqueue(
            task_type=TaskType.DIRECTORY_BATCH,
            payload=job_payload,
            priority=JobPriority.MEDIUM
        )
        
        duplicate_info = None
        if settings.duplicate_detection_enabled:
            duplicate_info = {
                "enabled": True,
                "method": settings.duplicate_detection_method,
                "skip_duplicates": settings.duplicate_detection_skip_duplicates,
                "message": "Duplicate content will be automatically detected and skipped during processing."
            }
        
        return DirectoryIngestResponse(
            job_id=job_id,
            directory_path=request.directory_path,
            total_files_found=len(files_to_process),
            status="queued",
            message=f"Directory processing queued. Found {len(files_to_process)} files. Use job ID {job_id} to track progress.",
            duplicate_detection_info=duplicate_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue directory ingestion: {str(e)}")


@router.get("/documents/search")
async def search_documents(query: str, limit: int = 10):
    """Search for similar content in processed documents"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Import here to avoid circular imports
        from core.ingestion import ingestion_pipeline
        results = await ingestion_pipeline.search_similar_content(query, limit)
        
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


