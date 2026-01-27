# syntax=docker/dockerfile:1.6

ARG TORCH=2.7.0
ARG CUDA_TAG=cu128

# -------- Builder (hat Compiler/NVCC Toolchain) --------
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel AS build

ARG TORCH
ARG CUDA_TAG
RUN echo "TORCH=${TORCH} CUDA_TAG=${CUDA_TAG}"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/spt
# Wichtig: .dockerignore nutzen (data/, .git/, outputs/, notebooks/ etc.)
COPY . /opt/spt

# venv, damit wir im Runtime-Image nur /opt/venv rüberkopieren
RUN python -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Basispins (du hattest numpy+setuptools gepinnt)
RUN python -m pip install -U pip wheel "setuptools<81" "numpy==1.26.4"

# Prüfen, dass Torch/CUDA passt
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# --- Minimalset für dein predict.py + SPT typischer Inferenz-Pfad ---
# (bewusst ohne jupyter/plotly/wandb/matplotlib/ipywidgets/…)
RUN pip install \
    hydra-core hydra-colorlog pyrootutils "rich<=14.0" \
    pytorch-lightning torchmetrics==0.11.4 \
    laspy plyfile h5py colorhash numba

# PyG Wheels passend zu Torch/CUDA
# PyG bietet Wheels für Torch 2.7.0 + cu128. :contentReference[oaicite:1]{index=1}
RUN pip install pyg_lib torch_scatter torch_cluster \
      -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA_TAG}.html \
 && pip install torch_geometric

# SPT-spezifische Pakete (häufig für Partition/Graph/Features)
RUN pip install pgeof pycut-pursuit pygrid-graph torch-graph-components torch-ransac3d

# # FRNN (CUDA extension) bauen
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"
# ONLY RTX 5060 TI:
# ARG TORCH_CUDA_ARCH_LIST="12.0+PTX" 

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=8

RUN python -m pip install --no-build-isolation -v src/dependencies/FRNN/external/prefix_sum \
 && python -m pip install --no-build-isolation -v src/dependencies/FRNN

# -------- Runtime (kein Compiler/NVCC) --------
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /opt/spt
COPY . /opt/spt
ENV PYTHONPATH=/opt/spt

CMD ["sleep", "infinity"]
