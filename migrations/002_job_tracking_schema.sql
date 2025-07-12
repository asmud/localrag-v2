-- Now create indexes on the properly migrated table
CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON processing_jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_priority ON processing_jobs(priority);
CREATE INDEX IF NOT EXISTS idx_jobs_task_type ON processing_jobs(task_type);
CREATE INDEX IF NOT EXISTS idx_jobs_worker_id ON processing_jobs(worker_id);
CREATE INDEX IF NOT EXISTS idx_jobs_payload_user_id ON processing_jobs USING GIN (payload jsonb_path_ops);

-- Create job_statistics table for aggregated job metrics
CREATE TABLE IF NOT EXISTS job_statistics (
    id SERIAL PRIMARY KEY,
    date DATE DEFAULT CURRENT_DATE,
    total_jobs INTEGER DEFAULT 0,
    completed_jobs INTEGER DEFAULT 0,
    failed_jobs INTEGER DEFAULT 0,
    cancelled_jobs INTEGER DEFAULT 0,
    avg_processing_time FLOAT DEFAULT 0.0,
    max_processing_time FLOAT DEFAULT 0.0,
    total_processing_time FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create unique constraint on date for job_statistics
CREATE UNIQUE INDEX IF NOT EXISTS idx_job_stats_date ON job_statistics(date);

-- Function to update job statistics
CREATE OR REPLACE FUNCTION update_job_statistics()
RETURNS TRIGGER AS $$
BEGIN
    -- Only update statistics when job is completed, failed, or cancelled
    IF NEW.status IN ('completed', 'failed', 'cancelled') AND 
       OLD.status NOT IN ('completed', 'failed', 'cancelled') THEN
        
        INSERT INTO job_statistics (
            date,
            total_jobs,
            completed_jobs,
            failed_jobs,
            cancelled_jobs,
            avg_processing_time,
            max_processing_time,
            total_processing_time
        )
        SELECT 
            CURRENT_DATE,
            COUNT(*),
            COUNT(*) FILTER (WHERE status = 'completed'),
            COUNT(*) FILTER (WHERE status = 'failed'),
            COUNT(*) FILTER (WHERE status = 'cancelled'),
            AVG(actual_duration) FILTER (WHERE actual_duration IS NOT NULL),
            MAX(actual_duration) FILTER (WHERE actual_duration IS NOT NULL),
            SUM(actual_duration) FILTER (WHERE actual_duration IS NOT NULL)
        FROM processing_jobs
        WHERE DATE(created_at) = CURRENT_DATE
        ON CONFLICT (date) DO UPDATE SET
            total_jobs = EXCLUDED.total_jobs,
            completed_jobs = EXCLUDED.completed_jobs,
            failed_jobs = EXCLUDED.failed_jobs,
            cancelled_jobs = EXCLUDED.cancelled_jobs,
            avg_processing_time = EXCLUDED.avg_processing_time,
            max_processing_time = EXCLUDED.max_processing_time,
            total_processing_time = EXCLUDED.total_processing_time,
            updated_at = CURRENT_TIMESTAMP;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update job statistics
DROP TRIGGER IF EXISTS trigger_update_job_statistics ON processing_jobs;
CREATE TRIGGER trigger_update_job_statistics
    AFTER UPDATE ON processing_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_job_statistics();

-- Create function to clean up old job records
CREATE OR REPLACE FUNCTION cleanup_old_jobs(days_old INTEGER DEFAULT 7)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM processing_jobs 
    WHERE status IN ('completed', 'failed', 'cancelled')
    AND created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * days_old;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to get job queue metrics
CREATE OR REPLACE FUNCTION get_job_queue_metrics()
RETURNS TABLE (
    status VARCHAR(20),
    count BIGINT,
    avg_duration NUMERIC,
    min_duration NUMERIC,
    max_duration NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        j.status,
        COUNT(*) as count,
        AVG(j.actual_duration) as avg_duration,
        MIN(j.actual_duration) as min_duration,
        MAX(j.actual_duration) as max_duration
    FROM processing_jobs j
    WHERE j.created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
    GROUP BY j.status
    ORDER BY count DESC;
END;
$$ LANGUAGE plpgsql;

-- Create indexes for better performance on common queries
CREATE INDEX IF NOT EXISTS idx_jobs_created_at_status ON processing_jobs(created_at, status);
CREATE INDEX IF NOT EXISTS idx_jobs_status_retry_count ON processing_jobs(status, retry_count);

-- Add comments to tables for documentation
COMMENT ON TABLE processing_jobs IS 'Background job tracking for document processing tasks';
COMMENT ON COLUMN processing_jobs.job_id IS 'Unique job identifier (UUID)';
COMMENT ON COLUMN processing_jobs.task_type IS 'Type of task: single_document, directory_batch, document_reprocess';
COMMENT ON COLUMN processing_jobs.priority IS 'Job priority: high, medium, low';
COMMENT ON COLUMN processing_jobs.status IS 'Job status: pending, processing, completed, failed, cancelled, retrying';
COMMENT ON COLUMN processing_jobs.progress IS 'Job completion percentage (0.0 to 100.0)';
COMMENT ON COLUMN processing_jobs.payload IS 'Job input parameters as JSON';
COMMENT ON COLUMN processing_jobs.result IS 'Job output result as JSON';
COMMENT ON COLUMN processing_jobs.worker_id IS 'ID of the worker process that handled this job';
COMMENT ON COLUMN processing_jobs.actual_duration IS 'Actual processing time in seconds';

COMMENT ON TABLE job_statistics IS 'Daily aggregated job processing statistics';
COMMENT ON FUNCTION update_job_statistics() IS 'Trigger function to update daily job statistics';
COMMENT ON FUNCTION cleanup_old_jobs(INTEGER) IS 'Function to clean up old completed/failed job records';
COMMENT ON FUNCTION get_job_queue_metrics() IS 'Function to get current job queue performance metrics';