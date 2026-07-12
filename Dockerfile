FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /workspace/safari

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        rsync \
        tmux \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-aux.txt /tmp/requirements-aux.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements-aux.txt

CMD ["python", "-m", "train", "--help"]
