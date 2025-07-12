-- Chat Threading Migration
-- PostgreSQL Migration 004
-- Add support for message threading and conversation branching

-- Add threading columns to chat_messages table
ALTER TABLE chat_messages 
ADD COLUMN IF NOT EXISTS parent_message_id UUID REFERENCES chat_messages(id) ON DELETE SET NULL,
ADD COLUMN IF NOT EXISTS thread_id UUID DEFAULT gen_random_uuid();

-- Create index for parent-child message relationships
CREATE INDEX IF NOT EXISTS idx_chat_messages_parent_id ON chat_messages(parent_message_id);

-- Create index for thread grouping
CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_id ON chat_messages(thread_id);

-- Create index for session + thread combination queries
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_thread ON chat_messages(session_id, thread_id);

-- Add constraint to prevent self-referencing messages
ALTER TABLE chat_messages
ADD CONSTRAINT chk_no_self_reference 
CHECK (parent_message_id != id);

-- Update existing messages to have their own thread_id (each message becomes root of its own thread initially)
UPDATE chat_messages 
SET thread_id = id 
WHERE thread_id IS NULL;

-- Create view for thread structure analysis
CREATE OR REPLACE VIEW message_threads AS
WITH RECURSIVE thread_tree AS (
    -- Base case: root messages (no parent)
    SELECT 
        id,
        session_id,
        thread_id,
        parent_message_id,
        role,
        content,
        created_at,
        0 as depth,
        ARRAY[id] as path,
        id as root_message_id
    FROM chat_messages 
    WHERE parent_message_id IS NULL
    
    UNION ALL
    
    -- Recursive case: child messages
    SELECT 
        cm.id,
        cm.session_id,
        cm.thread_id,
        cm.parent_message_id,
        cm.role,
        cm.content,
        cm.created_at,
        tt.depth + 1,
        tt.path || cm.id,
        tt.root_message_id
    FROM chat_messages cm
    JOIN thread_tree tt ON cm.parent_message_id = tt.id
)
SELECT 
    id,
    session_id,
    thread_id,
    parent_message_id,
    role,
    content,
    created_at,
    depth,
    path,
    root_message_id,
    cardinality(path) as thread_length
FROM thread_tree
ORDER BY session_id, thread_id, depth, created_at;

-- Create function to get thread summary
CREATE OR REPLACE FUNCTION get_thread_summary(p_thread_id UUID)
RETURNS TABLE (
    thread_id UUID,
    session_id UUID,
    message_count BIGINT,
    max_depth INT,
    created_at TIMESTAMP WITH TIME ZONE,
    last_message_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p_thread_id as thread_id,
        cm.session_id,
        COUNT(*) as message_count,
        MAX(mt.depth) as max_depth,
        MIN(cm.created_at) as created_at,
        MAX(cm.created_at) as last_message_at
    FROM chat_messages cm
    JOIN message_threads mt ON cm.id = mt.id
    WHERE cm.thread_id = p_thread_id
    GROUP BY cm.session_id;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions for the view and function
GRANT SELECT ON message_threads TO PUBLIC;

COMMENT ON TABLE chat_messages IS 'Chat messages with threading support for conversation branching';
COMMENT ON COLUMN chat_messages.parent_message_id IS 'Reference to parent message for threading';
COMMENT ON COLUMN chat_messages.thread_id IS 'Thread identifier for grouping related messages';
COMMENT ON VIEW message_threads IS 'Hierarchical view of message threads with depth and path information';
COMMENT ON FUNCTION get_thread_summary(UUID) IS 'Get summary statistics for a specific thread';