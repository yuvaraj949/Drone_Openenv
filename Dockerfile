FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────────────
LABEL maintainer="drone-traffic-control"
LABEL description="Autonomous Drone Traffic Control — OpenEnv Environment"
LABEL version="1.0"

# ── System deps ───────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Copy round1_submission directory ───────────────────────────────────────
COPY round1_submission/ .

# ── Python dependencies (cached layer) ───────────────────────────────────
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Environment variables ─────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# Disable matplotlib GUI backend (headless)
ENV MPLBACKEND=Agg

# ── Healthcheck ───────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "from environment.drone_env import DroneTrafficEnv; DroneTrafficEnv('easy').reset(); print('ok')" || exit 1

# ── Ports ────────────────────────────────────────────────────────────────
# OpenEnv Server
EXPOSE 7860

# ── Entry point ───────────────────────────────────────────────────────────
# Default: launch OpenEnv server
ENTRYPOINT ["python"]
CMD ["-m", "server.app"]

