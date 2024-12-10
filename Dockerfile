# Stage 1: Build environment with Poetry and dependencies
FROM python:3.10.15-slim as builder

LABEL maintainer="Soutrik soutrik1991@gmail.com" \
      description="Docker image for running a Python app with dependencies managed by Poetry."

# Install AWS Lambda Web Adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Install Poetry and necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add Poetry to the PATH explicitly
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory to /app
WORKDIR /app

# Copy pyproject.toml and poetry.lock to install dependencies
COPY pyproject.toml poetry.lock /app/

# Configure Poetry environment
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install dependencies without installing the package itself
RUN --mount=type=cache,target=/tmp/poetry_cache poetry install --only main --no-root

# Additional steps: Uninstall and re-add cryptography
RUN poetry run pip uninstall -y cryptography && \
    poetry add cryptography --lock

# Stage 2: Runtime environment
FROM python:3.10.15-slim as runner

# Install curl for health check script
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy.env file to set environment variables
COPY .env /app/.env

# Copy application source code and necessary files
COPY src /app/src
COPY app.py /app/app.py

# Copy virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# copy checkpoints from local to docker
COPY checkpoints /app/checkpoints

# Set the working directory to /app
WORKDIR /app

# Set the environment path to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Expose port 8080 for documentation purposes
EXPOSE 8080

CMD ["python", "-m", "app"]
