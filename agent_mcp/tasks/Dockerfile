# Dockerfile for sandboxed agentic eval tasks
# Used by inspect-ai's docker sandbox for running agent evals
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    sqlite3 \
    netcat-openbsd \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir requests

WORKDIR /home/user
RUN useradd -m user || true

CMD ["tail", "-f", "/dev/null"]
