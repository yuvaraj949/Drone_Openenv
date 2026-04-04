FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────────────
LABEL maintainer="drone-traffic-control"
LABEL description="Autonomous Drone Traffic Control — OpenEnv Environment"
LABEL version="0.2"

# ── System deps ───────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ───────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────
COPY . .

# ── Environment variables ─────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# Disable matplotlib GUI backend (headless)
ENV MPLBACKEND=Agg

# ── Healthcheck ───────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "from environment.drone_env import DroneTrafficEnv; DroneTrafficEnv('easy').reset(); print('ok')" || exit 1

# ── Ports ────────────────────────────────────────────────────────────────
# Gradio UI
EXPOSE 7860

# ── Entry point ───────────────────────────────────────────────────────────
# Default: launch Gradio UI
# Override for CLI: docker run drone-traffic python inference.py --task hard --seed 7
ENTRYPOINT ["python"]
CMD ["app.py"]
