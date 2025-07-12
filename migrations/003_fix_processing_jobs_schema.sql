-- Processing Jobs Schema Cleanup Migration
-- This migration serves as a safety net to ensure all required columns and indexes exist
-- Should be mostly redundant after migration 002, but handles edge cases

-- Ensure all required columns exist (safety net)
DO $$
BEGIN
    -- These should all exist after migration 002, but just in case...
    
    -- Ensure job_id column exists
    IF NOT EXISTS (SELECT FROM information_schema.columns WHERE table_name = 'processing_jobs' AND column_name = 'job_id') THEN
        ALTER TABLE processing_jobs ADD COLUMN job_id VARCHAR(36);
        UPDATE processing_jobs SET job_id = CAST(id AS VARCHAR) WHERE job_id IS NULL;
        ALTER TABLE processing_jobs ALTER COLUMN job_id SET NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_processing_jobs_job_id ON processing_jobs(job_id);
    END IF;
    
    -- Ensure all other required columns exist
    IF NOT EXISTS (SELECT FROM information_schema.columns WHERE table_name = 'processing_jobs' AND column_name = 'priority') THEN
        ALTER TABLE processing_jobs ADD COLUMN priority VARCHAR(10) DEFAULT 'medium';
        ALTER TABLE processing_jobs ALTER COLUMN priority SET NOT NULL;
    END IF;
    
    IF NOT EXISTS (SELECT FROM information_schema.columns WHERE table_name = 'processing_jobs' AND column_name = 'progress') THEN
        ALTER TABLE processing_jobs ADD COLUMN progress FLOAT DEFAULT 0.0;
    END IF;
    
    IF NOT EXISTS (SELECT FROM information_schema.columns WHERE table_name = 'processing_jobs' AND column_name = 'retry_count') THEN
        ALTER TABLE processing_jobs ADD COLUMN retry_count INTEGER DEFAULT 0;
    END IF;
    
    IF NOT EXISTS (SELECT FROM information_schema.columns WHERE table_name = 'processing_jobs' AND column_name = 'worker_id') THEN
        ALTER TABLE processing_jobs ADD COLUMN worker_id VARCHAR(50);
    END IF;
    
    IF NOT EXISTS (SELECT FROM information_schema.columns WHERE table_name = 'processing_jobs' AND column_name = 'estimated_duration') THEN
        ALTER TABLE processing_jobs ADD COLUMN estimated_duration INTEGER;
    END IF;
    
    IF NOT EXISTS (SELECT FROM information_schema.columns WHERE table_name = 'processing_jobs' AND column_name = 'actual_duration') THEN
        ALTER TABLE processing_jobs ADD COLUMN actual_duration INTEGER;
    END IF;
    
END $$;

-- Ensure all indexes exist (some might be missing from migration 002)
CREATE INDEX IF NOT EXISTS idx_jobs_priority ON processing_jobs(priority);
CREATE INDEX IF NOT EXISTS idx_jobs_task_type ON processing_jobs(task_type);
CREATE INDEX IF NOT EXISTS idx_jobs_worker_id ON processing_jobs(worker_id);
CREATE INDEX IF NOT EXISTS idx_jobs_payload_user_id ON processing_jobs USING GIN (payload jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at_status ON processing_jobs(created_at, status);
CREATE INDEX IF NOT EXISTS idx_jobs_status_retry_count ON processing_jobs(status, retry_count);