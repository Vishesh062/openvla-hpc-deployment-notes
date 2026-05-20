# 01 — Project Overview

## The host organisation

The Centre for Applied Artificial Intelligence (CAAI) at Macquarie University is a research centre within the Faculty of Science and Engineering, based at the Wallumattagal campus in North Ryde, Sydney. The centre's emphasis is on applied AI work — things that end in a system, not only in a paper. Active research streams include secure and privacy-preserving AI, human-centred and explainable AI, AI for social good, and robotics and automation.

I sat with the robotics group. Their flagship hardware is **ALOHA**, a low-cost bimanual teleoperated robot originally designed at Stanford. CAAI's interest is in adapting Vision-Language-Action (VLA) models (TinyVLA, OpenVLA, MiniVLA, and others) to run on ALOHA, eventually moving toward affordable robotic learning systems that work outside controlled lab conditions.

## The brief

Three deliverables, in priority order:

1. Get **OpenVLA-7B** running on the centre's GPU server
2. Prepare a custom dataset of cleaned ALOHA teleop demonstrations (`aloha_clean_dish`)
3. Fine-tune the model on that dataset for comparison against MiniVLA

The model weights were public ([OpenVLA on Hugging Face](https://huggingface.co/openvla/openvla-7b)). The codebase was open ([openvla/openvla](https://github.com/openvla/openvla)). The hardware existed. The work was applied engineering, not novel ML research: take a published model, adapt it to constrained infrastructure, fine-tune it on a custom dataset, deliver a working pipeline.

## Timeline

The internship ran for 15 weeks from 28 February to mid-June 2025, part-time, alongside coursework.

| Weeks | Focus |
|---|---|
| 1–2 | Literature review — "Show and Tell," then a broader sweep of VLA papers |
| 3 | Internal presentation and technical discussion |
| 4 | Comparative study of MiniVLA, TinyVLA, CogACT, OpenVLA |
| 5 | Hybrid TinyVLA + RoboMamba pipeline proposal |
| 6 | ALOHA documentation review; reading the TinyVLA codebase |
| 7 | Local environment setup; FastAPI + ROS 2 backend skeleton |
| 8 | First end-to-end local TinyVLA inference; remote server access via VPN |
| 9 | Migration to the GPU server; cache redirection; OpenVLA-7B loaded |
| 10 | OpenVLA evaluation; first FlashAttention build attempts |
| 11 | FlashAttention symbol-resolution debugging; MiniVLA fine-tune launched |
| 12 | Final testing; model compatibility troubleshooting; documentation |

## Working environment

Mostly remote: VPN into Macquarie's network, SSH into the GPU server, no GUI. Occasional in-person lab visits to actually see ALOHA. Weekly meetings with the supervisor and team.

The server (`aloha-server` throughout these docs) had two constraints that shaped the whole project:

- **No sudo access.** The root filesystem was managed by someone I never met. PyTorch and CUDA versions were installed system-wide and weren't getting upgraded mid-project.
- **A 95%-full root partition.** Default cache locations (`~/.cache`, `/tmp`, `~/miniconda3/pkgs`) all lived on root. A secondary 1 TB volume (`/Disk1`) had space but nothing used it by default.

These two facts drive most of `docs/04`. If you've worked on a university HPC cluster, none of this surprises you. If you've only worked on Colab or your own laptop, this is what "production" looks like in a lot of research environments.

## Team

- **Supervisor:** [Yuankai Qi](https://yuankaiqi.github.io/) — research fellow at CAAI, project framing, weekly review, unblocking.
- **Project teammates:** Prajwal Chaudhary, Md Rownak Islam Dip.
- **My role:** the OpenVLA / MiniVLA deployment track — environment setup, model loading, fine-tune scaffolding, dataset prep, the FlashAttention debug work. The hybrid pipeline proposal in `docs/03` was also mine.

## What got done, honestly

- A working OpenVLA-7B inference pipeline on dual A100s
- A patched MiniVLA training scaffold
- A documented set of dependency failures with workarounds
- A comparative analysis of eight VLA frameworks

The fine-tune itself didn't complete. System-level CUDA / FlashAttention mismatches intermittently corrupted gradients during the backward pass, and with one week left I chose to focus on reproducibility documentation rather than chase a flaky training loop. The thing that didn't finish is the thing I learned the most from. The rest of these docs are why.
