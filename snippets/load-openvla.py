"""
load-openvla.py
---------------
Minimal driver to load OpenVLA-7B and run a single inference.

Prerequisites:
  - Cache variables already redirected (see snippets/cache-redirect.sh)
  - PyTorch with CUDA support installed
  - transformers >= 4.40

Run:
  python load-openvla.py path/to/image.jpg "pick up the red block"

Expected first-run behaviour: ~2.5 minutes to load three shards totalling
13.6 GB, ~15 GB GPU memory footprint on an A100 once everything is on device.

About flash_attention_2:
  The screenshots in this repo show the model loaded with
  attn_implementation="flash_attention_2". That's the version I actually ran.
  If your PyTorch + CUDA + flash-attn aren't a clean ABI-compatible triple,
  this will load fine and then crash mid-forward-pass with a flash_attn_2_cuda
  symbol error. See docs/05-flashattention-debug.md. The default below uses
  standard attention to keep this snippet portable. Switch to flash_attention_2
  when you've verified the build with snippets/flashattn-build.sh.
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


MODEL_ID = "openvla/openvla-7b"


def load_model_and_processor(device: str = "cuda:0"):
    """Load the processor and model.

    trust_remote_code=True is required because OpenVLA ships its own
    modeling code (modeling_prismatic.py, configuration_prismatic.py,
    processing_prismatic.py) auto-downloaded from the Hub.
    Pin to a revision if you need reproducibility.
    """
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    # bfloat16 is the right dtype on A100 / H100. fp16 also works.
    # Skip torch_dtype and the model loads in fp32 and OOMs.
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        # attn_implementation="flash_attention_2",   # see note at top of file
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    model.eval()
    return processor, model


def run_inference(processor, model, image_path: Path, instruction: str, device: str = "cuda:0"):
    """Single-step inference: image + instruction -> 7-dim action vector.

    Returns a tensor of shape (7,): 6 end-effector deltas (dx, dy, dz,
    droll, dpitch, dyaw) and 1 gripper command.
    """
    image = Image.open(image_path).convert("RGB")
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    action = model.predict_action(
        **inputs,
        unnorm_key="bridge_orig",   # swap for your dataset's normalisation key
        do_sample=False,
    )
    return action


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    image_path = Path(sys.argv[1])
    instruction = sys.argv[2]

    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {MODEL_ID} (first run: ~2.5 min)...")
    processor, model = load_model_and_processor()

    print(f"Running inference on {image_path}: {instruction!r}")
    action = run_inference(processor, model, image_path, instruction)

    print("Predicted Action:", action)


if __name__ == "__main__":
    main()
