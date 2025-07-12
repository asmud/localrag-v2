"""
Redis-based async task queue for background document processing.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as redis
from loguru import logger
from pydantic import BaseModel

from .config import settings


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(str, Enum):
    """Job priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskType(str, Enum):
    """Task type enumeration."""
    SINGLE_DOCUMENT = "single_document"
    DIRECTORY_BATCH = "directory_batch"
    DOCUMENT_REPROCESS = "document_reprocess"


class JobData(BaseModel):
    """Job data model."""
    job_id: str
    task_type: TaskType
    priority: JobPriority
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0
    payload: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class TaskQueue:
    """Redis-based async task queue using modern redis-py."""
    
    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None
        self.queue_name = settings.task_queue_name
        self.job_prefix = f"{self.queue_name}:job:"
        self.status_prefix = f"{self.queue_name}:status:"
        self.progress_prefix = f"{self.queue_name}:progress:"
        
        # Priority queue names
        self.priority_queues = {
            JobPriority.HIGH: f"{self.queue_name}:high",
            JobPriority.MEDIUM: f"{self.queue_name}:medium", 
            JobPriority.LOW: f"{self.queue_name}:low"
        }
    
    @property
    def redis(self) -> redis.Redis:
        """Get Redis connection, creating it if necessary."""
        if self._redis is None:
            raise RuntimeError("Redis connection not initialized. Call connect() first.")
        return self._redis
    
    async def connect(self):
        """Connect to Redis using modern redis-py."""
        if self._redis is not None:
            return  # Already connected
            
        try:
            # Parse Redis URL to get connection parameters
            redis_url = settings.get_redis_url
            
            # Create connection pool for better resource management
            self._connection_pool = redis.ConnectionPool.from_url(
                redis_url,
                encoding='utf-8',
                decode_responses=True,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Create Redis client with connection pool
            self._redis = redis.Redis(
                connection_pool=self._connection_pool,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            
            # Test connection
            await self._redis.ping()
            logger.info("Connected to Redis task queue")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            await self.disconnect()  # Clean up on failure
            raise
    
    async def disconnect(self):
        """Disconnect from Redis and clean up resources."""
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception as e:
                logger.warning(f"Error during Redis disconnect: {e}")
            finally:
                self._redis = None
        
        if self._connection_pool:
            try:
                await self._connection_pool.aclose()
            except Exception as e:
                logger.warning(f"Error closing connection pool: {e}")
            finally:
                self._connection_pool = None
        
        logger.info("Disconnected from Redis task queue")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def enqueue(self, 
                     task_type: TaskType,
                     payload: Dict[str, Any],
                     priority: JobPriority = JobPriority.MEDIUM) -> str:
        """Enqueue a new job."""
        await self.connect()
        
        job_id = str(uuid.uuid4())
        job_data = JobData(
            job_id=job_id,
            task_type=task_type,
            priority=priority,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            payload=payload
        )
        
        # Store job data
        job_key = f"{self.job_prefix}{job_id}"
        await self.redis.set(
            job_key,
            job_data.model_dump_json(),
            ex=settings.task_job_retention
        )
        
        # Add to priority queue
        queue_name = self.priority_queues[priority]
        await self.redis.lpush(queue_name, job_id)
        
        logger.info(f"Enqueued job {job_id} with {priority} priority")
        return job_id
    
    async def dequeue(self, priority: JobPriority, timeout: int = 5) -> Optional[JobData]:
        """Dequeue a job from the specified priority queue."""
        await self.connect()
        
        queue_name = self.priority_queues[priority]
        result = await self.redis.brpop(queue_name, timeout=timeout)
        
        if not result:
            return None
        
        # Modern redis-py returns tuple of (queue_name, job_id)
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            queue_name_result, job_id = result
        else:
            job_id = result
        
        # redis-py with decode_responses=True returns strings, no need to decode
        job_data = await self.get_job(job_id)
        
        if job_data:
            # Update status to processing
            await self.update_job_status(job_id, JobStatus.PROCESSING)
            job_data.status = JobStatus.PROCESSING
            job_data.started_at = datetime.now()
        
        return job_data
    
    async def dequeue_any_priority(self, timeout: int = 5) -> Optional[JobData]:
        """Dequeue from any priority queue (high priority first)."""
        await self.connect()
        
        # Try high priority first, then medium, then low
        for priority in [JobPriority.HIGH, JobPriority.MEDIUM, JobPriority.LOW]:
            job_data = await self.dequeue(priority, timeout=1)
            if job_data:
                return job_data
        
        return None
    
    async def get_job(self, job_id: str) -> Optional[JobData]:
        """Get job data by ID."""
        await self.connect()
        
        job_key = f"{self.job_prefix}{job_id}"
        job_json = await self.redis.get(job_key)
        
        if not job_json:
            return None
        
        # redis-py with decode_responses=True returns strings directly
        try:
            job_dict = json.loads(job_json)
            return JobData(**job_dict)
        except Exception as e:
            logger.error(f"Failed to parse job data for {job_id}: {e}")
            return None
    
    async def update_job_status(self, job_id: str, status: JobStatus, 
                               error_message: Optional[str] = None,
                               result: Optional[Dict[str, Any]] = None):
        """Update job status."""
        await self.connect()
        
        job_data = await self.get_job(job_id)
        if not job_data:
            logger.warning(f"Job {job_id} not found for status update")
            return
        
        job_data.status = status
        if error_message:
            job_data.error_message = error_message
        if result:
            job_data.result = result
        if status == JobStatus.COMPLETED or status == JobStatus.FAILED:
            job_data.completed_at = datetime.now()
        
        # Update job data
        job_key = f"{self.job_prefix}{job_id}"
        await self.redis.set(
            job_key,
            job_data.model_dump_json(),
            ex=settings.task_job_retention
        )
        
        # Update status key for quick access
        status_key = f"{self.status_prefix}{job_id}"
        await self.redis.set(status_key, status.value, ex=settings.task_job_retention)
        
        logger.info(f"Updated job {job_id} status to {status}")
    
    async def update_job_progress(self, job_id: str, progress: float, 
                                 message: Optional[str] = None):
        """Update job progress."""
        await self.connect()
        
        progress_data = {
            "progress": min(100.0, max(0.0, progress)),
            "message": message or "",
            "updated_at": datetime.now().isoformat()
        }
        
        progress_key = f"{self.progress_prefix}{job_id}"
        await self.redis.set(
            progress_key,
            json.dumps(progress_data),
            ex=settings.task_job_retention
        )
        
        # Also update the job data
        job_data = await self.get_job(job_id)
        if job_data:
            job_data.progress = progress_data["progress"]
            job_key = f"{self.job_prefix}{job_id}"
            await self.redis.set(
                job_key,
                job_data.model_dump_json(),
                ex=settings.task_job_retention
            )
    
    async def get_job_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job progress."""
        await self.connect()
        
        progress_key = f"{self.progress_prefix}{job_id}"
        progress_json = await self.redis.get(progress_key)
        
        if not progress_json:
            return None
        
        # redis-py with decode_responses=True returns strings directly
        try:
            return json.loads(progress_json)
        except Exception as e:
            logger.error(f"Failed to parse progress data for {job_id}: {e}")
            return None
    
    async def requeue_job(self, job_id: str) -> bool:
        """Requeue a failed job for retry."""
        await self.connect()
        
        job_data = await self.get_job(job_id)
        if not job_data:
            return False
        
        if job_data.retry_count >= settings.task_max_retries:
            logger.warning(f"Job {job_id} exceeded max retries ({settings.task_max_retries})")
            return False
        
        # Increment retry count
        job_data.retry_count += 1
        job_data.status = JobStatus.RETRYING
        job_data.error_message = None
        
        # Update job data
        job_key = f"{self.job_prefix}{job_id}"
        await self.redis.set(
            job_key,
            job_data.model_dump_json(),
            ex=settings.task_job_retention
        )
        
        # Re-add to queue with delay
        await asyncio.sleep(settings.task_retry_delay)
        queue_name = self.priority_queues[job_data.priority]
        await self.redis.lpush(queue_name, job_id)
        
        logger.info(f"Requeued job {job_id} for retry ({job_data.retry_count}/{settings.task_max_retries})")
        return True
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or processing job."""
        await self.connect()
        
        job_data = await self.get_job(job_id)
        if not job_data:
            return False
        
        if job_data.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False
        
        # Remove from queue if pending
        if job_data.status == JobStatus.PENDING:
            queue_name = self.priority_queues[job_data.priority]
            await self.redis.lrem(queue_name, 1, job_id)
        
        # Update status
        await self.update_job_status(job_id, JobStatus.CANCELLED)
        logger.info(f"Cancelled job {job_id}")
        return True
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        await self.connect()
        
        stats = {
            "queue_lengths": {},
            "total_pending": 0
        }
        
        for priority, queue_name in self.priority_queues.items():
            length = await self.redis.llen(queue_name)
            stats["queue_lengths"][priority.value] = length
            stats["total_pending"] += length
        
        return stats
    
    async def remove_job(self, job_id: str) -> bool:
        """Remove a job from Redis queue and data."""
        await self.connect()
        
        job_data = await self.get_job(job_id)
        if not job_data:
            return False
        
        # Remove from queue if still pending
        if job_data.status == JobStatus.PENDING:
            queue_name = self.priority_queues[job_data.priority]
            await self.redis.lrem(queue_name, 1, job_id)
        
        # Remove job data
        await self.redis.delete(f"job:{job_id}")
        await self.redis.delete(f"job:{job_id}:progress")
        
        logger.info(f"Removed job {job_id} from queue")
        return True
    
    async def cleanup_expired_jobs(self):
        """Clean up expired job data."""
        await self.connect()
        
        # This is handled automatically by Redis TTL
        # But we can implement additional cleanup logic here if needed


# Global task queue instance
task_queue = TaskQueue()