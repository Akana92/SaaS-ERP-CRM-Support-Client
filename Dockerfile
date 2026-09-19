FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    LANGCHAIN_TRACING_V2=false LANGSMITH_TRACING=false TOKENIZERS_PARALLELISM=false
COPY requirements-docker.txt ./
RUN python -m pip install --no-cache-dir -r requirements-docker.txt
# Triton, imported by bitsandbytes, compiles its small CUDA driver binding.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY src/support/ ./src/support/
COPY scripts/live_demo.py scripts/training_control.py scripts/fetch_assets.py ./scripts/
COPY configs/models.json configs/asset-sources.json ./configs/
COPY data/policy/employee-telecom-v3.json ./data/policy/
COPY data/erp/live-audiences.json data/erp/live-objects.json ./data/erp/
COPY artifacts/stage5/quality90-v1/reports/candidate-f-semantic-round06/aggregate-summary.json ./artifacts/stage5/quality90-v1/reports/candidate-f-semantic-round06/
COPY artifacts/stage5/quality90-v1/verification/candidate-f-final-assessment-verification.json ./artifacts/stage5/quality90-v1/verification/
COPY docs/CANDIDATE_F_RESULTS.md ./docs/
RUN mkdir -p /app/artifacts/stage5/live /app/artifacts/stage4
EXPOSE 7860
CMD ["python", "-u", "scripts/live_demo.py", "serve", "--host", "0.0.0.0", "--precision", "bf16"]
