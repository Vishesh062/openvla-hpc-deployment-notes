# 08 — Future Work

Open threads from the internship. Roughly the order I'd tackle them if I picked the project up again.

## 1. Finish the MiniVLA fine-tune

The training pipeline was scaffolded, the dataset prepared, the patches in `docs/06` applied, the loop launched. What's needed:

- A clean PyTorch + CUDA + FlashAttention triple in a Docker container or fresh environment, sidestepping the ABI issue from `docs/05` entirely
- Resume the training launch and run for at least one full epoch on `aloha_clean_dish`
- Evaluate against the same dataset's held-out split
- Compare action-prediction accuracy against the OpenVLA-7B baseline from week 10

Estimated time given a clean environment: about two weeks. Without a clean environment: indefinite.

## 2. Build the hybrid TinyVLA + RoboMamba pipeline

Architecture is in `docs/03`. What it would take to implement:

- A coordinator ROS 2 node subscribing to both `/actions/fast` and `/plan/subgoals`, publishing blended commands on `/cmd/action`
- A RoboMamba inference node, initially returning dummy subgoals on a timer, then the real model once the coordinator is validated
- QoS configuration: best-effort low-latency on the fast topic, reliable-delivery on the planner topic
- Integration testing against PyBullet or Gazebo before going anywhere near real hardware

Four to six weeks for a working simulator demo. Longer for real-robot deployment.

## 3. Test TinyVLA inside a physics simulator

Scoped during the internship, never executed. Setup involves:

- Building a URDF file for ALOHA (likely available from the open-source ALOHA project)
- Creating a PyBullet or Gazebo world with manipulable objects
- Wiring the existing TinyVLA FastAPI backend to publish actions into the simulator's control interface
- Running closed-loop tests where the simulator provides observations and the model provides actions

Lowest-risk way to validate the perception-to-action pipeline before any real hardware involvement. Also cheap visual variation for training-time augmentation.

Two to three weeks.

## 4. Write up the non-sudo ROS 2 build process

The Medium post and `docs/04` both reference a "separate adventure" — building ROS 2 Humble from source in user space, with patched `ros2.repos`, dependency loop resolution, and segmented `colcon` builds. That writeup doesn't exist yet. It would be useful to anyone trying to run ROS 2 on an HPC cluster.

One weekend to draft.

## 5. Distil the diffusion policy for faster inference

TinyVLA's diffusion policy involves multi-step sampling, which is expensive per inference. Two techniques from the literature would help:

- **Policy distillation** — compressing the multi-step diffusion into a single-step student model. Identified as a potential enhancement during the internship but never attempted.
- **Improved noise scheduling** — reducing the number of denoising steps without harming output quality.

Either would meaningfully improve fast-tier latency in the hybrid pipeline (`docs/03`).

Three to four weeks, depending on how much is novel vs. adapting published techniques.

## 6. Quantify the OpenVLA inference baseline

The 7-dim action vector OpenVLA-7B produced in week 10 was the first concrete output, but it was a single inference on a single image. A useful follow-up:

- Run OpenVLA-7B on a held-out portion of `aloha_clean_dish`
- Compare predicted action vectors against the dataset's ground-truth teleop commands
- Report mean per-dimension error, success rate at various thresholds, inference latency distribution

This gives a defensible baseline for any future fine-tuning work. Without it, "we fine-tuned and improved performance" has no anchor.

One week. Mostly evaluation infrastructure.

## Status

I'm not currently working on these. The internship ended, I no longer have access to `aloha-server`, and the source code lives there. Most likely paths for the work to continue:

- A future intern at CAAI picks up the project, using this repo as background
- Someone else interested in VLA deployment redoes the infrastructure work in a clean environment (cloud GPU, container-based)

If any of this looks worth collaborating on, find me via the links in the README. I'm not precious about the design. The goal was always for the work to be useful past the end of the internship, not to be mine.
