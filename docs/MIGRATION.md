# Database Migration Guide

This guide covers the complete database migration process for LocalRAG, including setup, execution, and troubleshooting.

## Overview

LocalRAG uses PostgreSQL as the primary database with pgvector extension for embeddings, along with Neo4j for knowledge graphs and Redis for caching. The migration system handles:

- Initial schema creation
- Vector embeddings setup (VECTOR(768) for multilingual-e5-base)
- Chat sessions and message management
- Document and chunk storage with metadata
- Background job processing tracking

## Prerequisites

### Required Extensions
- **PostgreSQL 14+** with the following extensions:
  - `uuid-ossp` - UUID generation
  - `vector` - pgvector for embeddings (required for similarity search)

### Database Access
- PostgreSQL database created and accessible
- User with CREATE, ALTER, and INSERT permissions
- Network connectivity from LocalRAG application

## Migration Commands

### Using LocalRAG Configuration (Recommended)

```bash
# Check database health and extensions
python utils/postgres_migrate.py health

# View migration status
python utils/postgres_migrate.py status

# Run all pending migrations
python utils/postgres_migrate.py migrate

# Create new migration file
python utils/postgres_migrate.py create migration_name

# For Neo4j migrations
python utils/neo4j_migrate.py status
python utils/neo4j_migrate.py migrate
```

### Manual Migration (For Initial Setup)

Use this when LocalRAG configuration is not available or during initial setup:

```bash
# Run PostgreSQL migrations with manual database connection
python utils/postgres_migrate.py migrate-manual \
  --host=localhost \
  --port=5432 \
  --database=localrag \
  --user=localrag

# Run Neo4j migrations with manual connection
python utils/neo4j_migrate.py migrate-manual \
  --uri=bolt://localhost:7687 \
  --user=neo4j
# Password will be prompted securely
```

### Docker Container Migration

```bash
# From within running LocalRAG container
docker exec localrag-localrag-1 python utils/postgres_migrate.py migrate

# Or run migration tool directly
docker run --rm -it \
  --network=host \
  -v $(pwd):/app \
  localrag:latest \
  python utils/migrate.py migrate-manual --host=localhost --database=localrag --user=localrag
```

## Migration Files

### Current Migrations

1. **001_initial_schema.sql** - Complete initial schema
   - Documents table with metadata
   - Chunks table with vector embeddings
   - Chat sessions and messages
   - Processing jobs for background tasks
   - All indexes and triggers

2. **002_job_tracking_schema.sql** - Background job processing
   - Processing jobs table with status tracking
   - Job metadata and error handling
   - Performance monitoring

3. **003_fix_processing_jobs_schema.sql** - Job schema fixes
   - Status constraint fixes
   - Index optimizations
   - Data integrity improvements

4. **004_chat_threading.sql** - Chat system enhancements
   - Message threading support
   - Session management improvements
   - Conversation history tracking

5. **005_fix_job_status_constraint.sql** - Job status validation
   - Enhanced status constraints
   - Better error handling
   - Status transition validation

6. **006_foundation.cypher** - Neo4j knowledge graph setup
   - Initial graph schema
   - Indonesian entity constraints
   - Relationship definitions

### Schema Created

#### Documents Table
```sql
documents (
    id SERIAL PRIMARY KEY,
    file_path VARCHAR(1000) NOT NULL UNIQUE,
    file_name VARCHAR(255) NOT NULL,
    file_size BIGINT NOT NULL,
    content_hash VARCHAR(64),
    chunk_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
)
```

#### Chunks Table with Vector Embeddings
```sql
chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    start_index INTEGER NOT NULL,
    end_index INTEGER NOT NULL,
    word_count INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding VECTOR(768), -- For multilingual-e5-base model
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
)
```

#### Chat Management
```sql
chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb
)

chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
)
```

## Troubleshooting

### Common Issues

#### 1. Extension Not Found
```
ERROR: extension "vector" is not available
```
**Solution**: Install pgvector extension
```bash
# Ubuntu/Debian
sudo apt install postgresql-14-pgvector

# Or compile from source
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make && sudo make install
```

#### 2. Connection Issues
```
socket.gaierror: [Errno -2] Name or service not known
```
**Solutions**:
- Check hostname resolution: use IP address instead of hostname
- For Docker: ensure containers are on same network
- Verify SSL settings: add `?ssl=disable` to connection string
- Check firewall and network connectivity

#### 3. Permission Denied
```
permission denied to create extension "vector"
```
**Solution**: Use superuser or grant permissions
```sql
-- As superuser
CREATE EXTENSION IF NOT EXISTS vector;
GRANT USAGE ON SCHEMA public TO localrag;
```

#### 4. SSL Connection Issues
```
SSL connection failed
```
**Solutions**:
- Add `POSTGRES_SSLMODE=disable` to .env
- Use manual migration with `--host=IP_ADDRESS`
- Configure PostgreSQL to accept SSL connections

#### 5. Dollar-Quoted Function Errors
```
unterminated dollar-quoted string
```
This is handled automatically by the improved migration parser. If you encounter this:
- Use the latest migration tool
- Check for manual SQL modifications
- Verify migration file format

### Verification Commands

```bash
# Check all tables created
docker exec postgres psql -U user -d database -c "
SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;
"

# Verify vector extension
docker exec postgres psql -U user -d database -c "
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
"

# Check indexes
docker exec postgres psql -U user -d database -c "
SELECT schemaname, indexname FROM pg_indexes 
WHERE schemaname = 'public' AND indexname LIKE 'idx_%';
"

# View migration history
docker exec postgres psql -U user -d database -c "
SELECT filename, applied_at FROM migrations ORDER BY id;
"
```

### Migration Status Checking

```bash
# Using LocalRAG migration tool
python utils/migrate.py status

# Manual database check
psql -h localhost -U user -d database -c "
SELECT 
  (SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'public') as tables,
  (SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public') as indexes,
  (SELECT COUNT(*) FROM migrations) as applied_migrations;
"
```

## Advanced Usage

### Creating Custom Migrations

```bash
# Create new migration file
python utils/migrate.py create add_user_preferences

# Edit the generated file
vim migrations/004_add_user_preferences.sql
```

### Rollback Process

Migrations are designed to be forward-only. For rollback:

1. **Manual Rollback**: Use the rollback instructions in each migration file
2. **Database Restore**: Restore from backup before migration
3. **Schema Recreation**: Drop tables and re-run migrations

### Environment-Specific Migrations

For different environments, you can:

```bash
# Development
python utils/migrate.py migrate-manual --database=localrag_dev

# Testing  
python utils/migrate.py migrate-manual --database=localrag_test

# Production
python utils/migrate.py migrate-manual --database=localrag_prod
```

## Best Practices

1. **Always backup** before running migrations in production
2. **Test migrations** in development environment first
3. **Check migration status** before and after running
4. **Monitor application logs** during migration
5. **Verify data integrity** after migration completion
6. **Use manual migration** for initial setup or troubleshooting
7. **Keep migration files** in version control
8. **Document custom migrations** with clear rollback instructions

## Integration with LocalRAG

After successful migration:

1. **Restart LocalRAG** application
2. **Check health endpoint**: `curl http://localhost:8080/api/v1/health`
3. **Verify database status**: All databases should show `true`
4. **Test document upload** and embedding creation
5. **Validate chat functionality** and session persistence

For configuration details, see [CONFIG.md](CONFIG.md).