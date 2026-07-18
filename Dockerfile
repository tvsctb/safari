FROM quay.io/vessl-ai/torch:2.3.1-cuda12.1-r5

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /workspace/safari

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        rsync \
        tmux \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-aux.txt /tmp/requirements-aux.txt
RUN python -m pip install --no-compile -r /tmp/requirements-aux.txt \
    && python -c 'import datasets, einops, hydra, matplotlib, numpy, opt_einsum, pandas, PIL, pytorch_lightning, rich, scipy, sklearn, timm, torch, torchmetrics, torchtext, torchvision, tqdm, transformers, wandb; assert torch.__version__.startswith("2.3.1"), torch.__version__; assert torchvision.__version__.startswith("0.18.1"), torchvision.__version__; assert numpy.__version__ == "1.24.4", numpy.__version__' \
    && rm /tmp/requirements-aux.txt

CMD ["python", "-m", "train", "--help"]
