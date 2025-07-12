from typing import Dict

from fastapi import APIRouter, HTTPException

from core.database import db_manager
from core.models import model_manager
from core.config import settings

router = APIRouter()


@router.get("/health", response_model=Dict)
async def health_check():
    """Check the health of all system components"""
    try:
        health_status = await db_manager.health_check()
        model_info = model_manager.get_model_info()
        
        all_healthy = all(health_status.values())
        
        # Check worker status if enabled
        worker_status = "disabled"
        if settings.task_queue_enabled and settings.task_workers_auto_start:
            try:
                from core.background_workers import worker_manager
                if worker_manager.is_running:
                    workers = await worker_manager.get_worker_status()
                    running_workers = sum(1 for w in workers if w["is_running"])
                    worker_status = f"healthy ({running_workers} workers)"
                else:
                    worker_status = "stopped"
            except Exception:
                worker_status = "error"
        
        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "databases": health_status,
            "models": model_info,
            "components": {
                "database_manager": "healthy" if all_healthy else "unhealthy",
                "model_manager": "healthy" if model_info.get("embedding_model_loaded") else "loading",
                "background_workers": worker_status
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@router.get("/health/database", response_model=Dict)
async def database_health():
    """Check database connections health"""
    try:
        health_status = await db_manager.health_check()
        return {
            "status": "healthy" if all(health_status.values()) else "unhealthy",
            "details": health_status
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database health check failed: {str(e)}")


@router.get("/health/models", response_model=Dict)
async def models_health():
    """Check models status"""
    try:
        model_info = model_manager.get_model_info()
        return {
            "status": "healthy" if model_info.get("embedding_model_loaded") else "loading",
            "details": model_info
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Models health check failed: {str(e)}")


@router.get("/health/workers", response_model=Dict)
async def workers_health():
    """Check background workers health"""
    try:
        if not settings.task_queue_enabled:
            return {
                "status": "disabled",
                "message": "Background workers are disabled in configuration"
            }
        
        if not settings.task_workers_auto_start:
            return {
                "status": "manual",
                "message": "Background workers not set to auto-start"
            }
        
        from core.background_workers import worker_manager
        from core.task_queue import task_queue
        
        if not worker_manager.is_running:
            return {
                "status": "stopped",
                "message": "Worker manager is not running"
            }
        
        # Get worker details
        workers = await worker_manager.get_worker_status()
        
        # Get queue statistics
        queue_stats = await task_queue.get_queue_stats()
        
        running_workers = sum(1 for w in workers if w["is_running"])
        total_processed = sum(w["processed_count"] for w in workers)
        total_failed = sum(w["failed_count"] for w in workers)
        
        return {
            "status": "healthy" if running_workers > 0 else "unhealthy",
            "worker_count": len(workers),
            "running_workers": running_workers,
            "total_processed": total_processed,
            "total_failed": total_failed,
            "queue_stats": queue_stats,
            "workers": workers
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Workers health check failed: {str(e)}")


