FROM python:3.11-slim

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps FIRST for layer caching
# All versions pinned in requirements.txt for reproducible builds
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (separate layer — changes more often)
COPY models.py graders.py baseline.py client.py __init__.py inference.py ui.py ./
COPY openenv.yaml pyproject.toml README.md ./
COPY server/ ./server/
COPY tasks/ ./tasks/
COPY sandbox/ ./sandbox/

# Create sandbox workspace for live execution (non-root writable)
RUN mkdir -p /tmp/sandbox && chmod 777 /tmp/sandbox

# Set path
ENV PYTHONPATH="/app"

# Enforce non-root execution (Required for HF Spaces)
RUN useradd -m -u 1000 appuser
USER 1000

# HF Spaces port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start server — single worker, fast startup
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
