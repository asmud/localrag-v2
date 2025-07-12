"""
Job management system for tracking document processing jobs.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .database import db_manager
from .task_queue import JobData, JobStatus, JobPriority, TaskType, task_queue


class JobManager:
    """Manages job persistence and tracking in PostgreSQL."""
    
    def __init__(self):
        self.db_manager = db_manager
    
    def _safe_json_parse(self, data: Any, field_name: str, job_id: str = None) -> Any:
        """Safely parse JSON data with error handling."""
        if not data:
            return None if field_name == "result" else {}
            
        if isinstance(data, str):
            try:
                return json.loads(data)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse {field_name} JSON for job {job_id}: {e}")
                return None if field_name == "result" else {}
        
        # If it's already a dict or other object, return as-is
        return data
    
    async def create_job_tables(self):
        """Create job tracking tables if they don't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS processing_jobs (
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
            estimated_duration INTEGER NULL,
            actual_duration INTEGER NULL
        )
        """
        
        index_sqls = [
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status)",
            "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON processing_jobs(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_jobs_priority ON processing_jobs(priority)",
            "CREATE INDEX IF NOT EXISTS idx_jobs_task_type ON processing_jobs(task_type)"
        ]
        
        async with self.db_manager.get_postgres_session() as session:
            # Create table first
            await session.execute(text(create_table_sql))
            
            # Create indexes one by one
            for index_sql in index_sqls:
                await session.execute(text(index_sql))
            
            await session.commit()
        
        logger.info("Job tracking tables created/verified")
    
    async def save_job(self, job_data: JobData, worker_id: Optional[str] = None) -> int:
        """Save job to database."""
        query = """
        INSERT INTO processing_jobs (
            job_id, task_type, priority, status, created_at, started_at,
            completed_at, progress, error_message, retry_count, payload,
            result, worker_id
        ) VALUES (
            :job_id, :task_type, :priority, :status, :created_at, :started_at,
            :completed_at, :progress, :error_message, :retry_count, :payload,
            :result, :worker_id
        )
        ON CONFLICT (job_id) DO UPDATE SET
            status = EXCLUDED.status,
            started_at = EXCLUDED.started_at,
            completed_at = EXCLUDED.completed_at,
            progress = EXCLUDED.progress,
            error_message = EXCLUDED.error_message,
            retry_count = EXCLUDED.retry_count,
            result = EXCLUDED.result,
            worker_id = EXCLUDED.worker_id
        RETURNING id
        """
        
        params = {
            "job_id": job_data.job_id,
            "task_type": job_data.task_type.value,
            "priority": job_data.priority.value,
            "status": job_data.status.value,
            "created_at": job_data.created_at,
            "started_at": job_data.started_at,
            "completed_at": job_data.completed_at,
            "progress": job_data.progress,
            "error_message": job_data.error_message,
            "retry_count": job_data.retry_count,
            "payload": json.dumps(job_data.payload),
            "result": json.dumps(job_data.result) if job_data.result else None,
            "worker_id": worker_id
        }
        
        async with self.db_manager.get_postgres_session() as session:
            result = await session.execute(text(query), params)
            await session.commit()
            row = result.fetchone()
            return row[0] if row else None
    
    async def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID from database."""
        query = """
        SELECT * FROM processing_jobs WHERE job_id = :job_id
        """
        
        async with self.db_manager.get_postgres_session() as session:
            result = await session.execute(text(query), {"job_id": job_id})
            row = result.fetchone()
            
            if not row:
                return None
            
            # Convert row to dict
            job_dict = dict(row._mapping)
            
            # Parse JSON fields safely
            job_dict["payload"] = self._safe_json_parse(job_dict.get("payload"), "payload", job_id)
            job_dict["result"] = self._safe_json_parse(job_dict.get("result"), "result", job_id)
            
            return job_dict
    
    async def update_job_status(self, job_id: str, status: JobStatus, 
                               error_message: Optional[str] = None,
                               result: Optional[Dict[str, Any]] = None,
                               worker_id: Optional[str] = None):
        """Update job status in database."""
        query = """
        UPDATE processing_jobs SET
            status = :status,
            error_message = :error_message,
            result = :result,
            worker_id = :worker_id,
            completed_at = CASE 
                WHEN :status_for_completion IN (:completed_status, :failed_status, :cancelled_status) 
                THEN CURRENT_TIMESTAMP 
                ELSE completed_at 
            END,
            started_at = CASE 
                WHEN :status_for_started = :processing_status AND started_at IS NULL 
                THEN CURRENT_TIMESTAMP 
                ELSE started_at 
            END,
            actual_duration = CASE 
                WHEN :status_for_duration IN (:completed_status, :failed_status, :cancelled_status) AND started_at IS NOT NULL
                THEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - started_at))::INTEGER
                ELSE actual_duration
            END
        WHERE job_id = :job_id
        """
        
        params = {
            "job_id": job_id,
            "status": status.value,
            "error_message": error_message,
            "result": json.dumps(result) if result else None,
            "worker_id": worker_id,
            "status_for_completion": status.value,
            "status_for_started": status.value,
            "status_for_duration": status.value,
            "completed_status": JobStatus.COMPLETED.value,
            "failed_status": JobStatus.FAILED.value,
            "cancelled_status": JobStatus.CANCELLED.value,
            "processing_status": JobStatus.PROCESSING.value
        }
        
        async with self.db_manager.get_postgres_session() as session:
            await session.execute(text(query), params)
            await session.commit()
        
        # Also update in Redis
        await task_queue.update_job_status(job_id, status, error_message, result)
        
        # Auto-delete failed jobs from both database and queue
        if status == JobStatus.FAILED:
            logger.info(f"Auto-deleting failed job {job_id}")
            await self.delete_job(job_id)
            await task_queue.remove_job(job_id)
    
    async def update_job_progress(self, job_id: str, progress: float, 
                                 message: Optional[str] = None):
        """Update job progress in database."""
        query = """
        UPDATE processing_jobs SET
            progress = :progress
        WHERE job_id = :job_id
        """
        
        async with self.db_manager.get_postgres_session() as session:
            await session.execute(text(query), {"job_id": job_id, "progress": progress})
            await session.commit()
        
        # Also update in Redis
        await task_queue.update_job_progress(job_id, progress, message)
    
    async def get_jobs_by_status(self, status: JobStatus, limit: int = 100) -> List[Dict[str, Any]]:
        """Get jobs by status."""
        query = """
        SELECT * FROM processing_jobs 
        WHERE status = :status
        ORDER BY created_at DESC
        LIMIT :limit
        """
        
        async with self.db_manager.get_postgres_session() as session:
            result = await session.execute(text(query), {"status": status.value, "limit": limit})
            rows = result.fetchall()
            
            jobs = []
            for row in rows:
                job_dict = dict(row._mapping)
                job_dict["payload"] = self._safe_json_parse(job_dict.get("payload"), "payload")
                job_dict["result"] = self._safe_json_parse(job_dict.get("result"), "result")
                jobs.append(job_dict)
            
            return jobs
    
    async def get_user_jobs(self, user_id: Optional[str] = None, 
                           limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get jobs for a specific user (or all jobs if no user specified)."""
        if user_id:
            query = """
            SELECT * FROM processing_jobs 
            WHERE payload::jsonb ->> 'user_id' = :user_id
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
            """
            params = {"user_id": user_id, "limit": limit, "offset": offset}
        else:
            query = """
            SELECT * FROM processing_jobs 
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
            """
            params = {"limit": limit, "offset": offset}
        
        async with self.db_manager.get_postgres_session() as session:
            result = await session.execute(text(query), params)
            rows = result.fetchall()
            
            jobs = []
            for row in rows:
                job_dict = dict(row._mapping)
                job_dict["payload"] = self._safe_json_parse(job_dict.get("payload"), "payload")
                job_dict["result"] = self._safe_json_parse(job_dict.get("result"), "result")
                jobs.append(job_dict)
            
            return jobs
    
    async def get_job_statistics(self) -> Dict[str, Any]:
        """Get job processing statistics."""
        query = """
        SELECT 
            status,
            COUNT(*) as count,
            AVG(actual_duration) as avg_duration,
            MAX(actual_duration) as max_duration,
            MIN(actual_duration) as min_duration
        FROM processing_jobs 
        WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
        GROUP BY status
        """
        
        async with self.db_manager.get_postgres_session() as session:
            result = await session.execute(text(query))
            rows = result.fetchall()
            
            stats = {
                "total_24h": 0,
                "status_counts": {},
                "processing_times": {}
            }
            
            for row in rows:
                status = row[0]
                count = row[1]
                avg_duration = row[2]
                max_duration = row[3]
                min_duration = row[4]
                
                stats["total_24h"] += count
                stats["status_counts"][status] = count
                
                if avg_duration:
                    stats["processing_times"][status] = {
                        "avg_seconds": float(avg_duration),
                        "max_seconds": float(max_duration) if max_duration else 0,
                        "min_seconds": float(min_duration) if min_duration else 0
                    }
            
            return stats
    
    async def delete_job(self, job_id: str):
        """Delete a job from database."""
        query = """
        DELETE FROM processing_jobs WHERE job_id = :job_id
        """
        
        async with self.db_manager.get_postgres_session() as session:
            result = await session.execute(text(query), {"job_id": job_id})
            await session.commit()
            return result.rowcount > 0
    
    async def cleanup_old_jobs(self, days_old: int = 7):
        """Clean up old completed/failed jobs."""
        query = """
        DELETE FROM processing_jobs 
        WHERE status IN ('completed', 'failed', 'cancelled')
        AND created_at < CURRENT_TIMESTAMP - INTERVAL '%s days'
        """
        
        async with self.db_manager.get_postgres_session() as session:
            result = await session.execute(text(query % days_old))
            await session.commit()
            deleted_count = result.rowcount
            
            logger.info(f"Cleaned up {deleted_count} old jobs older than {days_old} days")
            return deleted_count
    
    async def get_failed_jobs_for_retry(self) -> List[Dict[str, Any]]:
        """Get failed jobs that can be retried."""
        query = """
        SELECT * FROM processing_jobs 
        WHERE status = 'failed'
        AND retry_count < :max_retries
        AND created_at >= CURRENT_TIMESTAMP - INTERVAL '1 day'
        ORDER BY created_at DESC
        """
        
        async with self.db_manager.get_postgres_session() as session:
            result = await session.execute(text(query), {"max_retries": 3})  # From settings
            rows = result.fetchall()
            
            jobs = []
            for row in rows:
                job_dict = dict(row._mapping)
                job_dict["payload"] = self._safe_json_parse(job_dict.get("payload"), "payload")
                job_dict["result"] = self._safe_json_parse(job_dict.get("result"), "result")
                jobs.append(job_dict)
            
            return jobs


# Global job manager instance
job_manager = JobManager()