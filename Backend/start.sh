#!/bin/bash

# Activate virtual environment (only if exists)
if [ -d ".real" ]; then
    source .real/Scripts/activate  # Windows Git Bash / WSL
elif [ -d "venv" ]; then
    source venv/bin/activate  # Linux/Mac
fi

# Use PORT from environment (Render) or default to 8000 (local)
PORT=${PORT:-8000}

# Run uvicorn
exec uvicorn main:app --host 0.0.0.0 --port $PORT
