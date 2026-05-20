# 07 — Lessons Learned

Four things I'd tell myself on day one.

## 1. Redirect every cache before your first install

Before you run a single `pip install`, `conda activate`, or `git clone` on a shared server:

```bash
export HF_HOME=/your/disk/hf_cache
export TRANSFORMERS_CACHE=/your/disk/hf_cache
export PIP_CACHE_DIR=/your/disk/pip_cache
export TMPDIR=/your/disk/tmp
export CONDA_PKGS_DIRS=/your/disk/conda_pkgs
```

Add to `~/.bashrc`, source it, verify with `echo $HF_HOME`, then start work.

Sounds trivial. Cost me about a week of partial downloads, mysterious "no space left" errors, and one afternoon where the root partition hit 100% and locked out other users. Five minutes to do correctly. A week to recover from doing it wrong.

Full pattern in `docs/04-environment-setup.md`.

## 2. Treat PyTorch + CUDA + FlashAttention as a single version-locked triple

These three have ABI dependencies on each other that aren't separable. If any one of them drifts:

- Your wheel builds will succeed
- Your imports will work
- Your forward passes may even succeed
- Your backward passes will silently corrupt gradients

Pin all three at the start of the project. Document the exact versions in a `requirements.txt` or `environment.yml`. If your sysadmin upgrades CUDA mid-project, your FlashAttention build is dead and you may not realise until a training step produces NaNs.

Operational discipline that ML coursework doesn't teach because the courses run in clean Docker containers someone else maintained. Real deployment doesn't look like that.

Full story in `docs/05-flashattention-debug.md`.

## 3. The hard part of deploying a research model isn't the model

Every VLA paper I read during the internship described its model in fifty pages and its infrastructure in zero. Real deployment is the opposite ratio.

Where my time actually went:

- ~10% reading the OpenVLA paper and understanding the architecture
- ~5% writing the actual inference driver script
- ~25% on cache and disk space issues
- ~30% on PyTorch / CUDA / FlashAttention compatibility
- ~20% on tokenizer, model loader, and config patches (the MiniVLA work)
- ~10% on dataset preparation

The model is the small part. If your project timeline assumes the infrastructure is solved, you don't have a timeline, you have a hope.

## 4. A failed fine-tune is not a failed project

The MiniVLA fine-tune didn't complete. What the project did produce:

- A working OpenVLA-7B inference pipeline on dual A100s
- A FastAPI + ROS 2 backend running TinyVLA end-to-end locally
- A patched MiniVLA training scaffold (the patches in `docs/06`)
- A comparative analysis of eight VLA frameworks (`docs/02`)
- A hybrid TinyVLA + RoboMamba architecture proposal (`docs/03`)
- A documented set of dependency failures with workarounds (this repo)
- Build instructions for non-sudo ROS 2 (sketched in `docs/04`)

The thing that didn't finish is also the thing I learned the most from. Documenting the failure modes — the FlashAttention symbol issues, the MiniVLA scaffolding gaps, the cache management problems — is what makes this useful for anyone deploying VLAs on similar infrastructure.

That last point is worth restating. **The workarounds, the patches, the documented dead-ends are the deliverable.** If you join a research project expecting every line of code you write to lead to a working system, you'll be disappointed often. If you join expecting the documented dead-ends to be valuable themselves, you'll be right more often than wrong.

## The thing under all of these

Production deployment of research models is its own engineering discipline. Distinct from ML modelling. Almost entirely absent from coursework. Cache management, version pinning, ABI compatibility, build orchestration, environment reproducibility — none of it shows up in a Coursera deep learning class. All of it determines whether a paper's model actually runs on your hardware.

Get burned by this stuff in a low-stakes setting — a student project, a personal hack — rather than learning it for the first time at a company under deadline. Lessons are cheaper now.

## See also

- The Medium post: [Deploying OpenVLA-7B on a Shared HPC Server Without Sudo: Notes from a Failed Fine-Tune](https://medium.com/@vishesh062/deploying-openvla-7b-on-a-shared-hpc-server-without-sudo-notes-from-a-failed-fine-tune-40d6ba370657)
- `08-future-work.md` — open threads from this project
