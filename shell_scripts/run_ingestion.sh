#!/bin/bash
set -e  # stop if any script fails

echo "Running data ingestion pipeline..."

python src/data_ingestion.py
python src/data_processing.py
python src/chunking.py
python src/embedding_ingestion.py
python src/upload_qdrant.py

echo "Ingestion complete!"
