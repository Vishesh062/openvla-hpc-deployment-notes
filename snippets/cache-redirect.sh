#!/usr/bin/env bash
# cache-redirect.sh
#
# Redirect every cache that an ML stack uses onto a non-root partition.
# Run this BEFORE any pip install, conda activate, or git clone on a shared
# HPC server with a constrained root partition.
#
# Add the export lines to ~/.bashrc, source it, verify with `echo $HF_HOME`.
#
# Replace /Disk1/vish/ with your actual writable partition path.

# ---------------------------------------------------------------
# 1. Set cache locations
# ---------------------------------------------------------------
CACHE_ROOT="/Disk1/vish"   # <-- change this to your writable partition

export HF_HOME="${CACHE_ROOT}/hf_cache"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/hf_cache"   # deprecated, still respected
export PIP_CACHE_DIR="${CACHE_ROOT}/pip_cache"
export TMPDIR="${CACHE_ROOT}/tmp"
export CONDA_PKGS_DIRS="${CACHE_ROOT}/conda_pkgs"

# ---------------------------------------------------------------
# 2. Create the directories
# ---------------------------------------------------------------
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$CONDA_PKGS_DIRS"

# ---------------------------------------------------------------
# 3. Verify
# ---------------------------------------------------------------
echo "HF_HOME            = $HF_HOME"
echo "TRANSFORMERS_CACHE = $TRANSFORMERS_CACHE"
echo "PIP_CACHE_DIR      = $PIP_CACHE_DIR"
echo "TMPDIR             = $TMPDIR"
echo "CONDA_PKGS_DIRS    = $CONDA_PKGS_DIRS"
echo
echo "Disk usage on cache partition:"
df -h "$CACHE_ROOT"

# ---------------------------------------------------------------
# 4. (Optional) Create a Conda env directly under the cache partition,
#    avoiding ~/miniconda3/envs entirely
# ---------------------------------------------------------------
# conda create -p "${CACHE_ROOT}/envs/openvla" python=3.10 -y
# conda activate "${CACHE_ROOT}/envs/openvla"

# ---------------------------------------------------------------
# Why each variable matters:
#
#   HF_HOME             - Final destination for HuggingFace model weights,
#                         tokenizers, processor configs. ~13.6 GB for OpenVLA-7B.
#
#   TRANSFORMERS_CACHE  - Older name for the same thing. Deprecated but still
#                         read by some libraries. Set both.
#
#   PIP_CACHE_DIR       - pip's wheel cache. A single torch install is ~2 GB.
#
#   TMPDIR              - Python tempfile ops, partial downloads. HuggingFace
#                         downloads into TMPDIR FIRST and only moves to HF_HOME
#                         once verified. Miss this and you'll still fill /tmp
#                         even with HF_HOME set correctly.
#
#   CONDA_PKGS_DIRS     - Conda's downloaded package cache. Avoids the
#                         ~/miniconda3/pkgs default.
# ---------------------------------------------------------------
