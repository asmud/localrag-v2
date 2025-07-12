-- Migration: 005_fix_job_status_constraint.sql
-- Purpose: Fix processing_jobs status constraint to allow all JobStatus enum values
-- Issue: Database constraint only allows ['pending', 'running', 'completed', 'failed']
-- But application uses ['pending', 'processing', 'completed', 'failed', 'cancelled', 'retrying']

-- Drop the existing constraint that's too restrictive
ALTER TABLE processing_jobs DROP CONSTRAINT IF EXISTS processing_jobs_status_check;

-- Add the correct constraint that matches the JobStatus enum
ALTER TABLE processing_jobs ADD CONSTRAINT processing_jobs_status_check 
    CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled', 'retrying'));

-- Update any existing 'running' status to 'processing' for consistency
UPDATE processing_jobs SET status = 'processing' WHERE status = 'running';

-- Add comment for clarity
COMMENT ON CONSTRAINT processing_jobs_status_check ON processing_jobs 
    IS 'Ensures status matches JobStatus enum values: pending, processing, completed, failed, cancelled, retrying';