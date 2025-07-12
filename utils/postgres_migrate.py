#!/usr/bin/env python3
"""
Database migration utility for LocalRAG PostgreSQL.
This script is designed to be robust, clean, and easy to use.
"""

import asyncio
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict

import asyncpg
import click
from loguru import logger

# --- Configuration ---

# Attempt to import settings from the core application, but allow standalone operation.
try:
    # Add parent directory to path to allow importing 'core'
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.config import settings
    logger.info("Loaded configuration from core.config settings.")
except ImportError:
    settings = None
    logger.info("core.config not found. Running in standalone mode using environment variables.")

# --- Data Structures ---

class MigrationResult:
    """Represents the result of a single migration file execution."""
    def __init__(self, filename: str):
        self.filename = filename
        self.success = False
        self.error = None
        self.execution_time = 0.0

    def set_success(self, duration: float):
        self.success = True
        self.execution_time = duration

    def set_failure(self, error: Exception, duration: float):
        self.success = False
        self.error = error
        self.execution_time = duration

# --- Core Migration Logic ---

class PostgresMigrationManager:
    """Manages the entire PostgreSQL migration process."""

    def __init__(self):
        self.migrations_dir = Path("migrations")
        self.database_url = self._get_database_url()
        self._log_database_url()

    def _get_database_url(self) -> str:
        """
        Gets the database URL from settings or environment variables
        and ensures it's compatible with asyncpg.
        """
        raw_url = ""
        if settings and hasattr(settings, 'get_database_url'):
            raw_url = settings.get_database_url
        
        if not raw_url:
            # Fallback to environment variables
            raw_url = os.getenv('DATABASE_URL')
        
        if not raw_url:
            # Construct from individual parts if DATABASE_URL is not set
            user = os.getenv('POSTGRES_USER', 'localrag')
            password = os.getenv('POSTGRES_PASSWORD', 'localrag_password')
            host = os.getenv('POSTGRES_HOST', 'localhost')
            port = os.getenv('POSTGRES_PORT', '5432')
            db = os.getenv('POSTGRES_DB', 'localrag')
            ssl = os.getenv('POSTGRES_SSLMODE', 'prefer')
            raw_url = f"postgresql://{user}:{password}@{host}:{port}/{db}?ssl={ssl}"

        # Ensure the scheme is correct for asyncpg, removing the '+asyncpg' dialect part.
        return raw_url.replace("postgresql+asyncpg://", "postgresql://")

    def _log_database_url(self):
        """Logs the database URL, redacting the password for security."""
        if self.database_url:
            safe_url = re.sub(r':([^@]+)@', r':***@', self.database_url)
            logger.debug(f"PostgreSQL connection target: {safe_url}")
        else:
            logger.error("No database URL is configured. Please set DATABASE_URL or configure settings.")

    async def get_connection(self) -> asyncpg.Connection:
        """Establishes and returns a database connection with retry logic."""
        if not self.database_url:
            raise ConnectionError("Database URL not configured.")

        for attempt in range(3):
            try:
                logger.info(f"Connecting to PostgreSQL (attempt {attempt + 1}/3)...")
                conn = await asyncpg.connect(self.database_url, timeout=15)
                logger.success("PostgreSQL connection successful.")
                return conn
            except (asyncpg.PostgresError, OSError) as e:
                logger.warning(f"Connection attempt failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(2)
                else:
                    logger.error("Failed to connect to PostgreSQL after 3 attempts.")
                    raise ConnectionError("Could not connect to the database.") from e

    async def _ensure_migrations_table(self, conn: asyncpg.Connection):
        """Creates the migrations tracking table if it doesn't exist."""
        logger.debug("Ensuring 'migrations' table exists.")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) NOT NULL UNIQUE,
                applied_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            );
        """)

    async def get_applied_migrations(self, conn: asyncpg.Connection) -> List[str]:
        """Retrieves the list of already applied migration filenames."""
        await self._ensure_migrations_table(conn)
        rows = await conn.fetch("SELECT filename FROM migrations ORDER BY id;")
        applied = [row['filename'] for row in rows]
        logger.debug(f"Found {len(applied)} applied migrations in the database.")
        return applied

    def get_all_migration_files(self) -> List[Path]:
        """Gets all .sql migration files from the migrations directory, sorted numerically."""
        if not self.migrations_dir.is_dir():
            logger.warning(f"Migrations directory not found at '{self.migrations_dir}'.")
            return []
        
        files = sorted(self.migrations_dir.glob("*.sql"))
        logger.debug(f"Found {len(files)} total .sql migration files.")
        return files

    def _parse_migration_file(self, content: str) -> Dict[str, List[str]]:
        """
        Parses SQL content, separating statements that must run outside a transaction
        and correctly handling dollar-quoted strings.
        """
        # Remove comments first
        content = re.sub(r'--.*', '', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        statements = []
        current_statement = ""
        in_dollar_quote = False
        dollar_tag = None

        # Use a more robust regex to find potential dollar quote tags
        dollar_quote_regex = re.compile(r'(\$[a-zA-Z_]*\$)')

        # Split content by semicolons to handle most statements
        potential_statements = content.split(';')

        # Re-join statements that were incorrectly split inside dollar quotes
        rejoined_statements = []
        buffer = ""
        for part in potential_statements:
            buffer += part
            # Count opening and closing dollar quotes in the buffer
            open_tags = dollar_quote_regex.findall(buffer)
            if len(open_tags) % 2 == 0:
                rejoined_statements.append(buffer.strip())
                buffer = ""
            else:
                buffer += ';' # Add the semicolon back
        if buffer:
            rejoined_statements.append(buffer.strip())

        # Final filtering for empty statements
        statements = [s for s in rejoined_statements if s]

        transactional_stmts = []
        non_transactional_stmts = []

        non_transactional_pattern = re.compile(
            r'CREATE\s+INDEX\s+CONCURRENTLY|DROP\s+INDEX\s+CONCURRENTLY|VACUUM',
            re.IGNORECASE
        )

        for stmt in statements:
            if non_transactional_pattern.search(stmt):
                non_transactional_stmts.append(stmt)
            else:
                transactional_stmts.append(stmt)
        
        return {
            'transactional': transactional_stmts,
            'non_transactional': non_transactional_stmts
        }

    async def apply_migration(self, conn: asyncpg.Connection, migration_file: Path) -> MigrationResult:
        """Applies a single migration file with robust transaction handling."""
        result = MigrationResult(migration_file.name)
        start_time = time.time()
        logger.info(f"Applying migration: {migration_file.name}...")

        try:
            content = migration_file.read_text()
            if not content.strip():
                logger.warning("Migration file is empty, skipping.")
                result.set_success(time.time() - start_time)
                return result

            parsed_stmts = self._parse_migration_file(content)
            transactional = parsed_stmts['transactional']
            non_transactional = parsed_stmts['non_transactional']

            # Execute transactional statements
            if transactional:
                logger.info(f"Executing {len(transactional)} statements within a transaction.")
                async with conn.transaction():
                    for stmt in transactional:
                        logger.debug(f"Executing: {stmt[:100]}...")
                        await conn.execute(stmt)
            
            # Execute non-transactional statements
            if non_transactional:
                logger.info(f"Executing {len(non_transactional)} statements individually.")
                for stmt in non_transactional:
                    logger.debug(f"Executing (non-transactional): {stmt[:100]}...")
                    await conn.execute(stmt)

            # Record migration success in the database
            await conn.execute(
                "INSERT INTO migrations (filename) VALUES ($1) ON CONFLICT (filename) DO NOTHING;",
                migration_file.name
            )
            
            duration = time.time() - start_time
            result.set_success(duration)
            logger.success(f"Successfully applied {migration_file.name} in {duration:.2f}s.")

        except (asyncpg.PostgresError, OSError) as e:
            duration = time.time() - start_time
            result.set_failure(e, duration)
            logger.error(f"Failed to apply {migration_file.name} after {duration:.2f}s.")
            logger.error(f"Error: {e}")
        
        return result

    async def run_all_pending(self):
        """Runs all pending migrations in sequence."""
        logger.info("Starting PostgreSQL migration process.")
        all_files = self.get_all_migration_files()
        if not all_files:
            logger.info("No .sql migration files found. Nothing to do.")
            return

        try:
            conn = await self.get_connection()
            applied_files = await self.get_applied_migrations(conn)
            
            pending_files = [f for f in all_files if f.name not in applied_files]

            if not pending_files:
                logger.success("Database is up to date. No pending migrations.")
                return

            logger.info(f"Found {len(pending_files)} pending migrations.")
            
            results = []
            for migration_file in pending_files:
                result = await self.apply_migration(conn, migration_file)
                results.append(result)
                if not result.success:
                    logger.error("Stopping migration process due to a failure.")
                    break
            
            self._generate_summary(results)

        except ConnectionError as e:
            logger.error(f"Could not run migrations: {e}")
        finally:
            if 'conn' in locals() and conn and not conn.is_closed():
                await conn.close()
                logger.info("PostgreSQL connection closed.")

    def _generate_summary(self, results: List[MigrationResult]):
        """Generates and logs a summary of the migration run."""
        if not results:
            return

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        total_time = sum(r.execution_time for r in results)

        logger.info("\n" + "="*50)
        logger.info("PostgreSQL Migration Summary")
        logger.info("="*50)
        logger.info(f"Total migrations processed: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Total execution time: {total_time:.2f} seconds")

        if failed:
            logger.error("\nFailed migrations:")
            for r in failed:
                logger.error(f"  - {r.filename}: {r.error}")
        
        if not failed and successful:
            logger.success("\nAll pending migrations applied successfully!")
        logger.info("="*50)

# --- CLI Commands ---

@click.group()
def cli():
    """A robust migration tool for PostgreSQL."""
    # Configure Loguru for clean CLI output
    logger.remove()
    logger.add(sys.stderr, format="<level>{level: <8}</level> | <level>{message}</level>")
    pass

@cli.command()
def clean():
    """Clean all migrated schema. WARNING! This will destroy tour data too."""
    logger.warning("⚠️  WARNING: This will completely destroy all data in your PostgreSQL database!")
    logger.warning("⚠️  This action cannot be undone!")
    
    if not click.confirm("Are you absolutely sure you want to continue?"):
        logger.info("Operation cancelled.")
        return
    
    async def _clean():
        logger.info("Starting PostgreSQL database cleanup...")
        manager = PostgresMigrationManager()
        rollback_script = Path("migrations/rollback/001_cleansing.sql")
        
        if not rollback_script.exists():
            logger.error(f"Rollback script not found: {rollback_script}")
            return
        
        try:
            conn = await manager.get_connection()
            content = rollback_script.read_text()
            
            if not content.strip():
                logger.warning("Rollback script is empty. Nothing to clean.")
                return
            
            # Parse and execute the rollback script
            parsed_stmts = manager._parse_migration_file(content)
            transactional = parsed_stmts['transactional']
            non_transactional = parsed_stmts['non_transactional']
            
            start_time = time.time()
            
            # Execute transactional statements
            if transactional:
                logger.info(f"Executing {len(transactional)} cleanup statements within a transaction...")
                async with conn.transaction():
                    for stmt in transactional:
                        logger.debug(f"Executing: {stmt[:100]}...")
                        await conn.execute(stmt)
            
            # Execute non-transactional statements
            if non_transactional:
                logger.info(f"Executing {len(non_transactional)} cleanup statements individually...")
                for stmt in non_transactional:
                    logger.debug(f"Executing (non-transactional): {stmt[:100]}...")
                    await conn.execute(stmt)
            
            # Clean up the migrations tracking table
            logger.info("Cleaning up migrations tracking table...")
            await conn.execute("DELETE FROM migrations;")
            
            duration = time.time() - start_time
            logger.success(f"Database cleanup completed successfully in {duration:.2f} seconds.")
            logger.info("All tables, indexes, and migration history have been removed.")
            
        except ConnectionError as e:
            logger.error(f"Could not connect to database: {e}")
        except Exception as e:
            logger.error(f"Database cleanup failed: {e}")
        finally:
            if 'conn' in locals() and conn and not conn.is_closed():
                await conn.close()
                logger.info("PostgreSQL connection closed.")
    
    asyncio.run(_clean())

@cli.command()
def migrate():
    """Applies all pending .sql migrations."""
    manager = PostgresMigrationManager()
    asyncio.run(manager.run_all_pending())

@cli.command()
def status():
    """Checks the status of all .sql migrations."""
    async def _status():
        logger.info("Checking PostgreSQL migration status...")
        manager = PostgresMigrationManager()
        try:
            conn = await manager.get_connection()
            all_files = manager.get_all_migration_files()
            applied = await manager.get_applied_migrations(conn)
            
            logger.info("\n" + "="*50)
            logger.info("Migration Status")
            logger.info("="*50)
            for f in all_files:
                status_icon = "✅" if f.name in applied else "⏳"
                logger.info(f" {status_icon} {f.name}")
            logger.info("="*50)
            logger.info(f"Total: {len(all_files)}, Applied: {len(applied)}, Pending: {len(all_files) - len(applied)}")

        except ConnectionError as e:
            logger.error(f"Could not check status: {e}")
        finally:
            if 'conn' in locals() and conn and not conn.is_closed():
                await conn.close()

    asyncio.run(_status())

if __name__ == "__main__":
    cli()