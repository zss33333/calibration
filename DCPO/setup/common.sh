#!/usr/bin/env bash
# Source this file to use project-local caches and direct connections.
DCPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DCPO_WORKSPACE="$(dirname "$DCPO_ROOT")"
DCPO_ENV="/home2/zhousishan/miniconda3/envs/DCPO"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
export TMPDIR="$DCPO_WORKSPACE/.cache/DCPO/tmp"
export PIP_CACHE_DIR="$DCPO_WORKSPACE/.cache/DCPO/pip"
export CONDA_PKGS_DIRS="$DCPO_WORKSPACE/.cache/DCPO/conda"
export HF_HOME="$DCPO_WORKSPACE/.cache/DCPO/huggingface"
export MODELSCOPE_CACHE="$DCPO_WORKSPACE/.cache/DCPO/modelscope"
export TORCH_EXTENSIONS_DIR="$DCPO_WORKSPACE/.cache/DCPO/torch_extensions"
export TRITON_CACHE_DIR="$DCPO_WORKSPACE/.cache/DCPO/triton"
export CUDA_CACHE_PATH="$DCPO_WORKSPACE/.cache/DCPO/cuda"
export VLLM_CACHE_ROOT="$DCPO_WORKSPACE/.cache/DCPO/vllm"
export MPLCONFIGDIR="$DCPO_WORKSPACE/.cache/DCPO/matplotlib"
export XDG_CACHE_HOME="$DCPO_WORKSPACE/.cache/DCPO/xdg"
export PYTHONPATH="$DCPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
unset PIP_EXTRA_INDEX_URL
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME" \
  "$MODELSCOPE_CACHE" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$VLLM_CACHE_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
