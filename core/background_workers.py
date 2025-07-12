"""
Background worker processes for document ingestion tasks.
"""

import asyncio
import json
import os
import signal
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from loguru import logger

from .config import settings
from .task_queue import (
    JobData, JobStatus, JobPriority, TaskType, 
    task_queue
)
from .job_manager import job_manager
from .ingestion import ingestion_pipeline


def serialize_datetime_objects(obj: Any) -> Any:
    """Recursively convert datetime objects to ISO format strings for JSON serialization."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_datetime_objects(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime_objects(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_datetime_objects(item) for item in obj)
    else:
        return obj


class WorkerProcess:
    """Base worker process for handling background tasks."""
    
    def __init__(self, worker_id: Optional[str] = None, 
                 priority: Optional[JobPriority] = None):
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.priority = priority
        self.is_running = False
        self.current_job: Optional[JobData] = None
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = None
    
    async def start(self):
        """Start the worker process."""
        self.is_running = True
        self.start_time = datetime.now()
        
        logger.info(f"Starting worker {self.worker_id} with priority {self.priority}")
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        try:
            await self._main_loop()
        except Exception as e:
            logger.error(f"Worker {self.worker_id} crashed: {e}")
            logger.error(traceback.format_exc())
        finally:
            await self.stop()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Worker {self.worker_id} received signal {signum}, shutting down...")
        self.is_running = False
    
    async def stop(self):
        """Stop the worker process."""
        self.is_running = False
        
        # Wait for current job to complete if any
        if self.current_job:
            logger.info(f"Worker {self.worker_id} waiting for current job to complete...")
            # Could implement job cancellation here
        
        runtime = datetime.now() - self.start_time if self.start_time else None
        logger.info(f"Worker {self.worker_id} stopped. Processed: {self.processed_count}, "
                   f"Failed: {self.failed_count}, Runtime: {runtime}")
    
    async def _main_loop(self):
        """Main worker loop."""
        while self.is_running:
            try:
                # Dequeue job based on priority
                if self.priority:
                    job_data = await task_queue.dequeue(self.priority, timeout=5)
                else:
                    job_data = await task_queue.dequeue_any_priority(timeout=5)
                
                if not job_data:
                    continue
                
                self.current_job = job_data
                logger.info(f"Worker {self.worker_id} processing job {job_data.job_id}")
                
                # Save job to database with worker ID
                await job_manager.save_job(job_data, self.worker_id)
                
                # Process the job
                success = await self._process_job(job_data)
                
                if success:
                    self.processed_count += 1
                else:
                    self.failed_count += 1
                
                self.current_job = None
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                logger.error(traceback.format_exc())
                
                if self.current_job:
                    await self._handle_job_error(self.current_job, str(e))
                    self.current_job = None
                
                # Brief pause before continuing
                await asyncio.sleep(1)
    
    async def _process_job(self, job_data: JobData) -> bool:
        """Process a job based on its type."""
        try:
            if job_data.task_type == TaskType.SINGLE_DOCUMENT:
                return await self._process_single_document(job_data)
            elif job_data.task_type == TaskType.DIRECTORY_BATCH:
                return await self._process_directory_batch(job_data)
            elif job_data.task_type == TaskType.DOCUMENT_REPROCESS:
                return await self._process_document_reprocess(job_data)
            else:
                logger.error(f"Unknown task type: {job_data.task_type}")
                return False
                
        except Exception as e:
            logger.error(f"Job processing failed: {e}")
            await self._handle_job_error(job_data, str(e))
            return False
    
    async def _process_single_document(self, job_data: JobData) -> bool:
        """Process a single document upload."""
        try:
            file_path = job_data.payload.get("file_path")
            if not file_path:
                raise ValueError("No file_path in job payload")
            
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Update progress
            await job_manager.update_job_progress(job_data.job_id, 10.0, "Starting document processing")
            
            # Process the document
            result = await ingestion_pipeline.ingest_document(file_path)
            
            # Update progress
            await job_manager.update_job_progress(job_data.job_id, 90.0, "Finalizing processing")
            
            # Mark as completed (serialize datetime objects)
            serialized_result = serialize_datetime_objects(result)
            await job_manager.update_job_status(
                job_data.job_id, 
                JobStatus.COMPLETED,
                result=serialized_result,
                worker_id=self.worker_id
            )
            
            await job_manager.update_job_progress(job_data.job_id, 100.0, "Processing completed")
            
            logger.info(f"Successfully processed document: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Single document processing failed: {e}")
            await self._handle_job_error(job_data, str(e))
            return False
    
    async def _process_directory_batch(self, job_data: JobData) -> bool:
        """Process a directory of documents in batches."""
        try:
            directory_path = job_data.payload.get("directory_path")
            file_patterns = job_data.payload.get("file_patterns")
            batch_size = job_data.payload.get("batch_size", settings.task_batch_size)
            
            if not directory_path:
                raise ValueError("No directory_path in job payload")
            
            directory_path = Path(directory_path)
            if not directory_path.exists() or not directory_path.is_dir():
                raise ValueError(f"Invalid directory: {directory_path}")
            
            # Update progress
            await job_manager.update_job_progress(job_data.job_id, 5.0, "Scanning directory")
            
            # Get list of files to process
            file_patterns = file_patterns or ["*.txt", "*.md", "*.pdf", "*.docx", "*.html"]
            files_to_process = []
            for pattern in file_patterns:
                files_to_process.extend(directory_path.glob(f"**/{pattern}"))
            
            if not files_to_process:
                no_files_result = {"message": "No files found to process", "files_processed": 0}
                serialized_result = serialize_datetime_objects(no_files_result)
                await job_manager.update_job_status(
                    job_data.job_id,
                    JobStatus.COMPLETED,
                    result=serialized_result,
                    worker_id=self.worker_id
                )
                return True
            
            # Process files in batches
            total_files = len(files_to_process)
            processed_files = 0
            successful_files = 0
            skipped_duplicates = 0
            failed_files = []
            results = []
            
            for i in range(0, total_files, batch_size):
                batch_files = files_to_process[i:i + batch_size]
                
                # Process batch
                for file_path in batch_files:
                    try:
                        result = await ingestion_pipeline.ingest_document(file_path)
                        results.append(result)
                        
                        # Count different result types
                        if result.get("status") == "skipped_duplicate":
                            skipped_duplicates += 1
                            logger.info(f"Skipped duplicate: {file_path}")
                        else:
                            successful_files += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")
                        failed_files.append({"file": str(file_path), "error": str(e)})
                        results.append({
                            "file_path": str(file_path),
                            "status": "error",
                            "error": str(e)
                        })
                    
                    processed_files += 1
                    
                    # Update progress with more detailed status
                    progress = 10.0 + (processed_files / total_files) * 85.0
                    status_msg = f"Processed {processed_files}/{total_files} files"
                    if skipped_duplicates > 0:
                        status_msg += f" ({skipped_duplicates} duplicates skipped)"
                    
                    await job_manager.update_job_progress(
                        job_data.job_id, 
                        progress,
                        status_msg
                    )
                
                # Brief pause between batches
                await asyncio.sleep(0.1)
            
            # Compile final results with duplicate information
            final_result = {
                "total_files": total_files,
                "successful": successful_files,
                "skipped_duplicates": skipped_duplicates,
                "failed": len(failed_files),
                "failed_files": failed_files,
                "results": results,
                "summary": f"Processed {total_files} files: {successful_files} new, {skipped_duplicates} duplicates skipped, {len(failed_files)} failed"
            }
            
            # Mark as completed (serialize datetime objects)
            serialized_result = serialize_datetime_objects(final_result)
            await job_manager.update_job_status(
                job_data.job_id,
                JobStatus.COMPLETED,
                result=serialized_result,
                worker_id=self.worker_id
            )
            
            await job_manager.update_job_progress(job_data.job_id, 100.0, "Batch processing completed")
            
            logger.info(f"Successfully processed directory: {directory_path} "
                       f"({successful_files}/{total_files} files)")
            return True
            
        except Exception as e:
            logger.error(f"Directory batch processing failed: {e}")
            await self._handle_job_error(job_data, str(e))
            return False
    
    async def _process_document_reprocess(self, job_data: JobData) -> bool:
        """Reprocess an existing document."""
        try:
            # Similar to single document but with additional cleanup logic
            return await self._process_single_document(job_data)
        except Exception as e:
            logger.error(f"Document reprocessing failed: {e}")
            await self._handle_job_error(job_data, str(e))
            return False
    
    async def _handle_job_error(self, job_data: JobData, error_message: str):
        """Handle job processing errors."""
        logger.error(f"Job {job_data.job_id} failed: {error_message}")
        
        # Check if job should be retried
        if job_data.retry_count < settings.task_max_retries:
            # Requeue for retry
            requeued = await task_queue.requeue_job(job_data.job_id)
            if requeued:
                await job_manager.update_job_status(
                    job_data.job_id,
                    JobStatus.RETRYING,
                    error_message=f"Attempt {job_data.retry_count + 1}: {error_message}",
                    worker_id=self.worker_id
                )
                return
        
        # Mark as permanently failed
        await job_manager.update_job_status(
            job_data.job_id,
            JobStatus.FAILED,
            error_message=error_message,
            worker_id=self.worker_id
        )


class WorkerManager:
    """Manages multiple worker processes."""
    
    def __init__(self):
        self.workers: List[WorkerProcess] = []
        self.worker_tasks: List[asyncio.Task] = []
        self.is_running = False
    
    async def start_workers(self):
        """Start all configured workers."""
        if self.is_running:
            logger.warning("Workers already running")
            return
        
        self.is_running = True
        
        # Start high priority workers
        for i in range(settings.task_high_priority_workers):
            worker = WorkerProcess(
                worker_id=f"high-priority-{i+1}",
                priority=JobPriority.HIGH
            )
            self.workers.append(worker)
            task = asyncio.create_task(worker.start())
            self.worker_tasks.append(task)
        
        # Start general workers (handle medium and low priority)
        remaining_workers = max(0, settings.task_max_workers - settings.task_high_priority_workers)
        for i in range(remaining_workers):
            worker = WorkerProcess(
                worker_id=f"general-{i+1}",
                priority=None  # Handle any priority
            )
            self.workers.append(worker)
            task = asyncio.create_task(worker.start())
            self.worker_tasks.append(task)
        
        logger.info(f"Started {len(self.workers)} workers "
                   f"({settings.task_high_priority_workers} high-priority, "
                   f"{remaining_workers} general)")
        
        # Give workers a moment to initialize
        await asyncio.sleep(0.1)
    
    async def stop_workers(self):
        """Stop all workers gracefully."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        logger.info("Stopping all workers...")
        
        # Signal all workers to stop
        for worker in self.workers:
            worker.is_running = False
        
        # Wait for all tasks to complete with timeout
        if self.worker_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.worker_tasks, return_exceptions=True),
                    timeout=30.0  # 30 second timeout for graceful shutdown
                )
            except asyncio.TimeoutError:
                logger.warning("Worker shutdown timeout, some tasks may have been cancelled")
                # Cancel remaining tasks
                for task in self.worker_tasks:
                    if not task.done():
                        task.cancel()
        
        self.workers.clear()
        self.worker_tasks.clear()
        
        # Disconnect shared task queue Redis connection
        try:
            await task_queue.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting task queue during worker shutdown: {e}")
        
        logger.info("All workers stopped")
    
    async def get_worker_status(self) -> List[Dict[str, Any]]:
        """Get status of all workers."""
        status = []
        for worker in self.workers:
            worker_status = {
                "worker_id": worker.worker_id,
                "priority": worker.priority.value if worker.priority else "any",
                "is_running": worker.is_running,
                "processed_count": worker.processed_count,
                "failed_count": worker.failed_count,
                "current_job": worker.current_job.job_id if worker.current_job else None,
                "start_time": worker.start_time.isoformat() if worker.start_time else None
            }
            status.append(worker_status)
        
        return status


# Global worker manager instance
worker_manager = WorkerManager()


# Convenience function to start worker as a separate process
async def run_worker_process():
    """Run worker process (can be called from CLI or separate process)."""
    try:
        await worker_manager.start_workers()
        
        # Keep running until stopped
        while worker_manager.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await worker_manager.stop_workers()