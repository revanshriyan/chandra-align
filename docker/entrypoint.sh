#!/bin/bash
# Entrypoint script for Chandra-Align Docker container
# Sets up runtime environment and routes to batch CLI

set -e

# Set GDAL/PROJ environment variables
export GDAL_DATA=${GDAL_DATA:-/usr/share/gdal}
export PROJ_LIB=${PROJ_LIB:-/usr/share/proj}

# Set Python path
export PYTHONPATH="/install/lib/python3.12/site-packages:${PYTHONPATH}:/app"

# Ensure proper permissions
export HOME=/home/appuser

# Handle first argument - if it's a known command, run it directly
# Otherwise, pass through to batch CLI
case "$1" in
    --help|-h|batch|run|register|health)
        exec python3.12 -m chandra_align.cli.batch "$@"
        ;;
    *)
        # Default to batch CLI with all args
        exec python3.12 -m chandra_align.cli.batch "$@"
        ;;
esac