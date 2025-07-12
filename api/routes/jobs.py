"""
Job management API endpoints for background task monitoring.
"""

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.job_manager import job_manager
from core.task_queue import task_queue

router = APIRouter()


class JobResponse(BaseModel):
    """Unified job information response model."""
    job_id: str
    task_type: str
    priority: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float
    progress_message: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int
    worker_id: Optional[str] = None
    estimated_duration: Optional[int] = None
    actual_duration: Optional[int] = None
    result: Optional[Any] = None


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_info(job_id: str):
    """Get unified job information including status, progress, and results."""
    try:
        job_data = await job_manager.get_job_by_id(job_id)
        
        if not job_data:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Get real-time progress if available
        progress_data = await task_queue.get_job_progress(job_id)
        progress_message = None
        if progress_data:
            progress_message = progress_data.get("message", "")
        
        return JobResponse(
            job_id=job_data["job_id"],
            task_type=job_data["task_type"],
            priority=job_data["priority"],
            status=job_data["status"],
            created_at=job_data["created_at"],
            started_at=job_data.get("started_at"),
            completed_at=job_data.get("completed_at"),
            progress=job_data.get("progress", 0.0),
            progress_message=progress_message,
            error_message=job_data.get("error_message"),
            retry_count=job_data.get("retry_count", 0),
            worker_id=job_data.get("worker_id"),
            estimated_duration=job_data.get("estimated_duration"),
            actual_duration=job_data.get("actual_duration"),
            result=job_data.get("result") if job_data["status"] in ["completed", "failed"] else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job info: {str(e)}")


@router.delete("/jobs/cleanup")
async def cleanup_old_jobs(days_old: int = Query(7, ge=1, le=30, description="Age in days for job cleanup")):
    """Clean up old completed/failed jobs."""
    try:
        deleted_count = await job_manager.cleanup_old_jobs(days_old)
        return {
            "message": f"Cleaned up {deleted_count} jobs older than {days_old} days"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup jobs: {str(e)}")