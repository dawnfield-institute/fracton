# Dockerfile for Fracton Chatbot Example

FROM python:3.11-slim

LABEL maintainer="Dawn Field Institute"
LABEL description="Fracton Agentic Chatbot Example"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional chatbot dependencies
RUN pip install --no-cache-dir \
    openai \
    anthropic \
    fastapi \
    uvicorn \
    websockets

# Copy Fracton source
COPY fracton/ /app/fracton/
COPY setup.py /app/
COPY README.md /app/

# Install Fracton
RUN pip install -e .

# Copy chatbot example
COPY examples/chatbot/ /app/chatbot/

# Create directories
RUN mkdir -p /app/data /app/cache /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FRACTON_DATA_DIR=/app/data
ENV FRACTON_CACHE_DIR=/app/cache
ENV FRACTON_LOG_DIR=/app/logs
ENV DEVICE=cpu

# Expose port for web interface (future)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import fracton; print('OK')" || exit 1

# Default command: run chatbot
CMD ["python", "/app/chatbot/chatbot.py"]
