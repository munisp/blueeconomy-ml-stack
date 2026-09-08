FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
# requirements.txt's own comment claims "CPU-only default (CPU wheel by
# default on PyPI)" - that's false on Linux: PyPI's default torch/
# torch-geometric wheels for linux_x86_64 bundle full CUDA support and pull
# in ~2GB of nvidia_* runtime libraries (cublas, cudnn, nccl, etc.) that this
# cluster has no GPU nodes to use. Installing torch/torch-geometric from
# PyTorch's own CPU-only index first avoids that; everything else installs
# from PyPI normally afterward.
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY inference ./inference
COPY monitoring ./monitoring
COPY models ./models

ENV BEML_MODELS_ROOT=/app/models
ENV BEML_AB_CONFIG=/app/inference/ab_config.yaml
ENV BEML_LATENCY_BUDGET_MS=50

RUN useradd --system --create-home --uid 10001 app
USER app

EXPOSE 8100
CMD ["uvicorn", "inference.service:app", "--host", "0.0.0.0", "--port", "8100"]
