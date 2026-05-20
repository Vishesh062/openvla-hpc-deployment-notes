#!/usr/bin/env bash
# flashattn-build.sh
#
# Build FlashAttention 2 for an NVIDIA A100 against a pre-installed PyTorch.
#
# IMPORTANT: a successful build does NOT mean a working install. If the PyTorch
# you're building against was compiled with a different CUDA toolkit or C++ ABI
# than what's currently on the system, this build will succeed, `import flash_attn`
# will work, and then forward/backward passes will fail intermittently with
# symbol resolution errors against libtorch.
#
# See docs/05-flashattention-debug.md for the full story.
# Use this when you control the PyTorch + CUDA versions. Otherwise, prefer a
# pinned Docker image with all three components built together.

# ---------------------------------------------------------------
# 1. Target the right compute capability
# ---------------------------------------------------------------
#  A100  -> 8.0
#  H100  -> 9.0
#  RTX 3090 / A6000 -> 8.6
#  RTX 4090 -> 8.9
export TORCH_CUDA_ARCH_LIST="8.0"

# ---------------------------------------------------------------
# 2. Cap parallel build jobs to avoid OOM
# ---------------------------------------------------------------
#  The build needs ~2-3 GB RAM per job. On a shared server with limited RAM,
#  cap this; on a workstation with 64+ GB, you can raise it.
export MAX_JOBS=4

# ---------------------------------------------------------------
# 3. (Optional) point ninja at a temp dir on a non-root partition
# ---------------------------------------------------------------
# export TMPDIR=/path/to/your/disk/tmp

# ---------------------------------------------------------------
# 4. Verify your PyTorch and CUDA before building
# ---------------------------------------------------------------
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvcc --version

# ---------------------------------------------------------------
# 5. Build
# ---------------------------------------------------------------
#  --no-build-isolation is REQUIRED. It tells pip to build against the
#  installed torch rather than fetching a fresh one in an isolated env.
pip install flash-attn --no-build-isolation

# ---------------------------------------------------------------
# 6. Verify build (NOT runtime - that requires forward passes)
# ---------------------------------------------------------------
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"

# ---------------------------------------------------------------
# 7. Smoke test (THIS is what catches ABI mismatches)
# ---------------------------------------------------------------
#  If this crashes with an unresolved symbol from libtorch, your PyTorch and
#  FlashAttention were built against incompatible ABIs. The fix is to rebuild
#  PyTorch from source against the same CUDA toolkit you're building
#  FlashAttention against -- which usually requires sudo and a clean env.
python <<'PY'
import torch
from flash_attn import flash_attn_func

q = torch.randn(1, 8, 16, 64, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1, 8, 16, 64, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1, 8, 16, 64, device='cuda', dtype=torch.bfloat16)

out = flash_attn_func(q, k, v)
print('forward OK, shape:', out.shape)

# Backward exercises more of the C++ extension surface. If forward passes
# but backward fails or produces NaNs, the ABI mismatch is real.
out.sum().backward()
print('backward OK')
PY
