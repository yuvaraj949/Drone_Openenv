FROM python:3.11-slim

# == Metadata ==============================================================
LABEL maintainer="drone-traffic-control"
LABEL description="Autonomous Drone Traffic Control - OpenEnv Environment"
LABEL version="1.0"

# == System deps ===========================================================
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc curl && \
    rm -rf /var/lib/apt/lists/*

# == Working directory =====================================================
WORKDIR /app

# == Copy round1_submission directory =======================================
COPY . .

# == Python dependencies (cached layer) ===================================
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# == Environment variables =================================================
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# Disable matplotlib GUI backend (headless)
ENV MPLBACKEND=Agg

# == Healthcheck ===========================================================
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# == Ports ================================================================
# OpenEnv Server
EXPOSE 7860

# == Entry point ===========================================================
# Default: launch OpenEnv server
# Entrypoint is empty to allow the evaluator to run 'sh inference.py' or other commands
CMD ["python", "app.py"]

