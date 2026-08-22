# ============================================================
# Dockerfile for Credit Card Churn Prediction API
# ============================================================
# Build:  docker build -t churn-predictor .
# Run:    docker run -p 8000:8000 churn-predictor
# ============================================================

# Step 1: Start from an official, minimal Python base image.
# 'slim' means a stripped-down Debian Linux with just Python --
# no unnecessary OS packages, keeping the image small.
FROM python:3.12-slim

# Step 2: Set the working directory INSIDE the container.
# All subsequent commands run from here, and our app files
# will live here. /app is a standard convention.
WORKDIR /app

# Step 3: Copy requirements.txt FIRST (before copying code).
# This is a deliberate Docker optimization called layer caching:
# Docker builds in layers, one per instruction. If requirements.txt
# hasn't changed, Docker reuses the cached pip install layer on
# the next build -- even if your code changed. This makes
# rebuilds significantly faster during development.
COPY requirements.txt .

# Step 4: Install dependencies inside the container.
# --no-cache-dir: don't store pip's download cache (saves space).
# --upgrade pip: ensures we're using a modern pip version.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the application code and model artifacts.
# Order matters for layer caching -- copy things that change
# frequently (code) AFTER things that change rarely (dependencies).
COPY src/ ./src/
COPY models/ ./models/
COPY main.py .

# Step 6: Expose port 8000 so the outside world can reach the API.
# This is documentation as much as configuration -- it tells Docker
# (and anyone reading this file) which port the app listens on.
EXPOSE 8000

# Step 7: The command that runs when the container starts.
# --host 0.0.0.0: listen on ALL network interfaces inside the
# container (not just localhost), so traffic from outside the
# container can actually reach it.
# --port 8000: match the EXPOSE above.
# No --reload: that's for development only, not for containers.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]