#!/bin/bash
# Wrapper script to run the BigQuery executor with the correct Python environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment and run the Python script
cd "$SCRIPT_DIR"
source venv/bin/activate
python bigquery_ai_gen_exec.py "$@"
