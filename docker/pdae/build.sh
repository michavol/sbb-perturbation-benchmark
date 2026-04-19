#!/bin/bash

set -e

echo "Building pDAE Docker container..."
docker build -f docker/pdae/Dockerfile -t cellsimbench/pdae:latest .
echo "Docker image built successfully: cellsimbench/pdae:latest"
