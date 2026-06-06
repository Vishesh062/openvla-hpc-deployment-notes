# OpenVLA on a Shared HPC Server: Deployment Notes

Case study from a 15-week research internship at Macquarie University's [Centre for Applied Artificial Intelligence (CAAI)](https://www.mq.edu.au/research/research-centres-groups-and-facilities/groups/centre-for-applied-artificial-intelligence), deploying Vision-Language-Action (VLA) models for the ALOHA bimanual robot.

The companion writeup is on Medium: **[Deploying OpenVLA-7B on a Shared HPC Server Without Sudo: Notes from a Failed Fine-Tune](https://medium.com/@vishesh062/deploying-openvla-7b-on-a-shared-hpc-server-without-sudo-notes-from-a-failed-fine-tune-40d6ba370657)**. Read that first if you only have ten minutes. This repo is the structured version with the model comparison, the architecture proposal, and the debugging trail broken out into separate docs.

![OpenVLA-7B inference: source script and the 7-dim action vector it produces](screenshots/01-openvla-inference-source.png)
*`run_openvla.py` and its output. A 7-dimensional action vector — six end-effector deltas and a gripper command, generated from an image and a natural-language instruction. About six weeks of infrastructure work for those seven numbers.*

## Status

The inference script and dataset explorer are now in `scripts/`, verified working as of June 2026. The notes, model comparison, and debugging trail are in `docs/` and `snippets/` as before.

What's here:

- `scripts/run_openvla.py` — the working inference script, confirmed producing correct output on an RTX A6000
- `scripts/dataview.py` — ALOHA HDF5 dataset explorer
- `scripts/environment.yml` — pinned conda environment (see the June 2026 update below for why this matters)
- A comparative analysis across eight VLA models
- The hybrid TinyVLA + RoboMamba architecture I proposed
- The infrastructure debugging trail: cache redirection, FlashAttention symbol failures, MiniVLA patches
- Lessons that generalise to anyone deploying a 7B model on a constrained shared server

The fine-tune didn't complete inside the internship window. I'm honest about that in `docs/07-lessons-learned.md`. Training did run: checkpoints exist at steps 70 through 12,366 (roughly 6% of the planned 200k-step run) before the internship ended.

## June 2026 update: inference verified, code recovered

I regained SSH access to the server a year after the internship ended and retrieved the scripts. Running `run_openvla.py` on the current environment hit three separate failures before producing output. All three are worth knowing about if you work with this stack.

**NumPy 2.x ABI break.** TensorFlow in the environment was compiled against NumPy 1.x. NumPy had drifted to 2.1.2 over the year. Fix: `pip install "numpy<2" --force-reinstall`.

**PyTorch / torchvision version mismatch.** PyTorch had updated to 2.7.1 but torchvision was still pinned to an older release, causing `torchvision::nms does not exist` on import. Fix: `pip install torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126`.

**LLaMA attention mask off-by-one on PyTorch 2.7.1.** PyTorch 2.7.1 changed how attention weights are shaped during generation, producing a tensor size mismatch (276 vs 275) in the causal mask addition. The fix is a two-line patch to `transformers/models/llama/modeling_llama.py`:

```python
# Replace: attn_weights = attn_weights + causal_mask
# With:
min_len = min(attn_weights.size(-1), causal_mask.size(-1))
attn_weights = attn_weights[..., :min_len] + causal_mask[..., :min_len]
```

Hardware note: the server now has RTX 5090s (Blackwell, sm_120). PyTorch 2.7.1 tops out at sm_90. The 5090s show up in `nvidia-smi` but PyTorch cannot use them. Use an Ampere or Ada GPU instead (`cuda:1` on this machine).

After those fixes:

```
Predicted Action: [-0.0035639  -0.0030142  -0.00959495 -0.01801952  0.02132494 -0.05164425  0.99607843]
```

Seven numbers. Six end-effector deltas and a gripper command (0.996 = open). Correct output format for `bridge_orig`.

## What we were trying to do

OpenVLA is a 7-billion-parameter Vision-Language-Action model. A `dinosiglip-224px` vision backbone feeds a Llama-2-7B language model, with action tokenisation on top. Give it an RGB image and an instruction like "pick up the red block," and it returns a sequence of action tokens that decode into robot end-effector commands.

The brief had three parts:

1. Get OpenVLA-7B running on the centre's dual-A100 server
2. Prepare a custom dataset of cleaned ALOHA teleop demonstrations (`aloha_clean_dish`)
3. Fine-tune the model on that dataset and compare against MiniVLA

Model weights were public. Codebase was open. Hardware was there. The interesting constraints were elsewhere.

## What I built and what I learned

| Phase | Output |
|---|---|
| Weeks 1–6: Research | Comparative analysis of 8 VLA models; hybrid TinyVLA + RoboMamba architecture proposal |
| Weeks 7–8: Local prototype | FastAPI server wrapping TinyVLA inference, wired to ROS 2; image → action token loop validated locally |
| Weeks 9–10: HPC migration | OpenVLA-7B loaded on dual-A100s; cache-redirection pattern for non-sudo HPC environments |
| Weeks 11–12: Fine-tune attempt | MiniVLA patches for Qwen 0.5B's non-standard hidden size; FlashAttention 2 build with symbol-resolution failures; training launched but not completed |

If you only take one thing from this repo, take the cache-redirection pattern in `snippets/cache-redirect.sh`. Five minutes of setup saves a week of partial downloads on a 95%-full root partition.

## Quick start

```bash
git clone https://github.com/Vishesh062/openvla-hpc-deployment-notes.git
cd openvla-hpc-deployment-notes
conda env create -f scripts/environment.yml
conda activate openvla
cd scripts
python run_openvla.py
```

Requires a CUDA GPU with at least 15 GB free, sm_50 through sm_90. If `cuda:0` is a Blackwell card, set `DEVICE = "cuda:1"` at the top of the script.

## Repository contents

```
.
├── README.md
├── scripts/
│   ├── run_openvla.py             verified inference script
│   ├── dataview.py                ALOHA HDF5 dataset explorer
│   ├── environment.yml            pinned conda environment (June 2026)
│   └── exampleimage.jpeg          example input image for run_openvla.py
├── docs/
│   ├── 01-project-overview.md             scope, organisation, team
│   ├── 02-vla-model-comparison.md         the 8-model comparative analysis
│   ├── 03-hybrid-pipeline-design.md       TinyVLA + RoboMamba proposal
│   ├── 04-environment-setup.md            cache redirection on a 95%-full root
│   ├── 05-flashattention-debug.md         the symbol-resolution saga
│   ├── 06-minivla-patches.md              base_llm.py and qwen25.py for Qwen 0.5B
│   ├── 07-lessons-learned.md              what generalises
│   └── 08-future-work.md                  open threads
├── snippets/
│   ├── cache-redirect.sh                  HF_HOME, TRANSFORMERS_CACHE, etc.
│   ├── flashattn-build.sh                 the TORCH_CUDA_ARCH_LIST invocation
│   └── load-openvla.py                    AutoProcessor / AutoModelForVision2Seq
├── diagrams/
│   └── hybrid-pipeline.png                the proposed architecture
└── screenshots/
    ├── 01-openvla-inference-source.png    run_openvla.py + 7-dim action vector
    ├── 02-openvla-7b-shards-loading.png   the 13.6 GB model load
    ├── 03-minivla-positional-embedding-resize.png  the (37,37)→(16,16) patch in action
    ├── 04-minivla-backbone-debugging.png  qwen_backbone.py debug session
    ├── 05-flashattention-warnings.png     the FA2 warning state
    └── 06-fastapi-real-image-request.png  the backend processing a real image
```

## Reading order

Ten minutes: the Medium post.

An hour, in order:

1. `02-vla-model-comparison.md` — the landscape, why OpenVLA was the right target
2. `04-environment-setup.md` — the cache redirection pattern, the single most reusable thing
3. `05-flashattention-debug.md` — where the project went sideways
4. `06-minivla-patches.md` — what broke when MiniVLA's Qwen 0.5B met OpenVLA's training scaffolding
5. `07-lessons-learned.md` — what generalises

## Acknowledgements

Supervised by [Yuankai Qi](https://yuankaiqi.github.io/) at Macquarie's CAAI. Project teammates: Prajwal Chaudhary and Md Rownak Islam Dip.

## About me

I'm looking for ML engineering and applied AI roles in Sydney. Find me on **[GitHub](https://github.com/Vishesh062)**, **[LinkedIn](https://linkedin.com/in/visheshsingh062)**, or **[Medium](https://medium.com/@vishesh062)**.

## License

Writeups in this repository are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code snippets under MIT — see `LICENSE`.
