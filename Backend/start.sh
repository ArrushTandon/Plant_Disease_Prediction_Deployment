#!/bin/bash

# Use PORT from environment (Render) or default to 8000 (local)
PORT=${PORT:-8000}

# Start Uvicorn
exec uvicorn main:app --host 0.0.0.0 --port $PORT
