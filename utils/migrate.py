#!/usr/bin/env python3
"""
Main migration script dispatcher for LocalRAG.

This script dispatches migration commands to the appropriate
PostgreSQL or Neo4j migration scripts.
"""

import sys
import subprocess
import click

@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.argument('db_type', type=click.Choice(['postgres', 'neo4j']))
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
def cli(db_type, args):
    """
    LocalRAG Migration Tool Dispatcher.

    DB_TYPE: The database type (postgres or neo4j).
    ARGS: The command and arguments for the specific migration script.
    """
    if db_type == 'postgres':
        script_path = 'utils/postgres_migrate.py'
    elif db_type == 'neo4j':
        script_path = 'utils/neo4j_migrate.py'
    else:
        # This case is handled by click.Choice, but as a safeguard:
        click.echo(f"Invalid database type: {db_type}", err=True)
        sys.exit(1)

    command = [sys.executable, script_path] + list(args)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except FileNotFoundError:
        click.echo(f"Error: Script not found at {script_path}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    cli()