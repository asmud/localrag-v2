#!/usr/bin/env python3
"""
Simple Neo4j migration utility for LocalRAG.
Designed for smooth first-time user experience.
"""

import asyncio
import os
import re
import time
import sys
from pathlib import Path
from typing import List

import click
from loguru import logger

# --- Configuration ---

try:
    from neo4j import AsyncGraphDatabase, exceptions as neo4j_exceptions
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.config import settings
    logger.info("Loaded configuration from core.config settings.")
except ImportError:
    settings = None
    logger.info("Using environment variables for configuration.")

# --- Data Structures ---

class MigrationResult:
    """Simple migration result tracking."""
    def __init__(self, filename: str):
        self.filename = filename
        self.success = False
        self.error = None
        self.execution_time = 0.0
        self.statements_executed = 0

    def set_success(self, duration: float, statements: int = 0):
        self.success = True
        self.execution_time = duration
        self.statements_executed = statements

    def set_failure(self, error: Exception, duration: float):
        self.success = False
        self.error = error
        self.execution_time = duration

# --- Core Migration Logic ---

class Neo4jMigrationManager:
    """Simple Neo4j migration manager for first-time users."""

    def __init__(self):
        if not NEO4J_AVAILABLE:
            logger.error("Neo4j driver not installed. Please run: pip install neo4j")
            sys.exit(1)
        
        self.migrations_dir = Path("migrations")
        self.neo4j_uri, self.neo4j_auth = self._get_neo4j_credentials()
        self._log_connection_info()

    def _get_neo4j_credentials(self) -> tuple:
        """Get Neo4j credentials from settings or environment."""
        if settings:
            uri = settings.neo4j_uri
            auth = (settings.neo4j_user, settings.neo4j_password)
            return uri, auth
        
        # Fallback to environment variables
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        user = os.getenv('NEO4J_USER', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'password')
        return uri, (user, password)

    def _log_connection_info(self):
        """Log connection information for user."""
        logger.info(f"Neo4j URI: {self.neo4j_uri}")
        logger.info(f"Neo4j User: {self.neo4j_auth[0]}")

    async def test_connection(self) -> bool:
        """Test Neo4j connection."""
        try:
            logger.info("Testing Neo4j connection...")
            driver = AsyncGraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth)
            await driver.verify_connectivity()
            await driver.close()
            logger.success("Neo4j connection successful!")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            logger.error("Please check your Neo4j server is running and credentials are correct.")
            return False

    async def get_driver(self) -> AsyncGraphDatabase.driver:
        """Get Neo4j driver."""
        if not self.neo4j_uri:
            raise ConnectionError("Neo4j URI not configured.")
        
        try:
            driver = AsyncGraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth)
            await driver.verify_connectivity()
            return driver
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise ConnectionError("Could not connect to Neo4j.") from e

    def get_migration_files(self) -> List[Path]:
        """Get all .cypher migration files."""
        if not self.migrations_dir.is_dir():
            logger.warning(f"Migrations directory not found: {self.migrations_dir}")
            return []
        
        files = sorted(self.migrations_dir.glob("*.cypher"))
        logger.info(f"Found {len(files)} migration files")
        return files

    def _parse_statements(self, content: str) -> List[str]:
        """Parse Cypher content into statements."""
        # Remove comments and empty lines
        lines = []
        for line in content.split('\n'):
            line = re.sub(r'//.*$', '', line).strip()
            if line:
                lines.append(line)
        
        # Join lines and split by semicolon
        clean_content = '\n'.join(lines)
        statements = [stmt.strip() for stmt in clean_content.split(';') if stmt.strip()]
        
        return statements

    async def apply_migration(self, driver: AsyncGraphDatabase.driver, migration_file: Path) -> MigrationResult:
        """Apply a single migration file."""
        result = MigrationResult(migration_file.name)
        start_time = time.time()
        
        logger.info(f"Applying migration: {migration_file.name}")

        try:
            content = migration_file.read_text(encoding='utf-8')
            if not content.strip():
                logger.warning("Migration file is empty, skipping")
                result.set_success(time.time() - start_time)
                return result

            statements = self._parse_statements(content)
            logger.info(f"Executing {len(statements)} statements")

            async with driver.session() as session:
                for i, stmt in enumerate(statements, 1):
                    logger.debug(f"Executing statement {i}/{len(statements)}")
                    try:
                        await session.run(stmt)
                    except Exception as e:
                        logger.error(f"Statement {i} failed: {e}")
                        logger.error(f"Statement: {stmt[:200]}...")
                        raise
            
            duration = time.time() - start_time
            result.set_success(duration, len(statements))
            logger.success(f"Migration {migration_file.name} completed in {duration:.2f}s")

        except Exception as e:
            duration = time.time() - start_time
            result.set_failure(e, duration)
            logger.error(f"Migration {migration_file.name} failed: {e}")
        
        return result

    async def run_migrations(self):
        """Run all pending migrations."""
        logger.info("Starting Neo4j migrations...")
        
        # Test connection first
        if not await self.test_connection():
            return
        
        migration_files = self.get_migration_files()
        if not migration_files:
            logger.info("No migration files found")
            return

        try:
            driver = await self.get_driver()
            results = []
            
            for migration_file in migration_files:
                result = await self.apply_migration(driver, migration_file)
                results.append(result)
                
                if not result.success:
                    logger.error("Migration failed, stopping")
                    break
            
            await driver.close()
            self._print_summary(results)

        except Exception as e:
            logger.error(f"Migration process failed: {e}")

    def _print_summary(self, results: List[MigrationResult]):
        """Print migration summary."""
        if not results:
            return

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        total_time = sum(r.execution_time for r in results)
        total_statements = sum(r.statements_executed for r in results)

        logger.info("\n" + "="*50)
        logger.info("Migration Summary")
        logger.info("="*50)
        logger.info(f"Migrations processed: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Total statements executed: {total_statements}")
        logger.info(f"Total time: {total_time:.2f} seconds")

        if failed:
            logger.error("\nFailed migrations:")
            for r in failed:
                logger.error(f"  - {r.filename}: {r.error}")
        
        if successful and not failed:
            logger.success("\nAll migrations completed successfully!")
        
        logger.info("="*50)

# --- CLI Commands ---

@click.group()
def cli():
    """Simple Neo4j migration tool for LocalRAG."""
    logger.remove()
    logger.add(sys.stderr, format="<level>{level: <8}</level> | <level>{message}</level>")

@cli.command()
def clean():
    """Clean all migrated schema. WARNING! This will destroy tour data too."""
    logger.warning("⚠️  WARNING: This will completely destroy all data in your Neo4j database!")
    logger.warning("⚠️  This action cannot be undone!")
    
    if not click.confirm("Are you absolutely sure you want to continue?"):
        logger.info("Operation cancelled.")
        return
    
    async def _clean():
        logger.info("Starting Neo4j database cleanup...")
        manager = Neo4jMigrationManager()
        rollback_script = Path("migrations/rollback/002_cleansing.cypher")
        
        if not rollback_script.exists():
            logger.error(f"Rollback script not found: {rollback_script}")
            return
        
        # Test connection first
        if not await manager.test_connection():
            return
        
        try:
            driver = await manager.get_driver()
            content = rollback_script.read_text(encoding='utf-8')
            
            if not content.strip():
                logger.warning("Rollback script is empty. Nothing to clean.")
                return
            
            statements = manager._parse_statements(content)
            logger.info(f"Executing {len(statements)} cleanup statements...")
            
            start_time = time.time()
            
            async with driver.session() as session:
                for i, stmt in enumerate(statements, 1):
                    logger.debug(f"Executing cleanup statement {i}/{len(statements)}")
                    try:
                        await session.run(stmt)
                    except Exception as e:
                        logger.warning(f"Statement {i} failed (may be expected): {e}")
                        logger.debug(f"Statement: {stmt[:200]}...")
                        # Continue with other statements even if some fail
                        # (expected for DROP operations on non-existent objects)
                        continue
            
            # Additional cleanup - remove all remaining nodes and relationships
            logger.info("Performing final cleanup - removing all remaining nodes and relationships...")
            async with driver.session() as session:
                # Delete all relationships first
                await session.run("MATCH ()-[r]-() DELETE r")
                # Delete all nodes
                await session.run("MATCH (n) DELETE n")
            
            await driver.close()
            
            duration = time.time() - start_time
            logger.success(f"Neo4j cleanup completed successfully in {duration:.2f} seconds.")
            logger.info("All nodes, relationships, constraints, and indexes have been removed.")
            
        except Exception as e:
            logger.error(f"Neo4j cleanup failed: {e}")
    
    asyncio.run(_clean())

@cli.command()
def migrate():
    """Run all pending migrations."""
    manager = Neo4jMigrationManager()
    asyncio.run(manager.run_migrations())

@cli.command()
def status():
    """Show migration status."""
    manager = Neo4jMigrationManager()
    files = manager.get_migration_files()
    
    if not files:
        logger.info("No migration files found")
        return
    
    logger.info(f"Found {len(files)} migration files:")
    for f in files:
        logger.info(f"  - {f.name}")

if __name__ == "__main__":
    cli()