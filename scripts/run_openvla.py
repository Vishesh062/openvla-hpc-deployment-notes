from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

# On this server cuda:0 is an RTX 5090 (sm_120), incompatible with PyTorch 2.7.1
# which supports up to sm_90. Switched to cuda:1 (A6000, sm_86).
# On most setups, cuda:0 is correct.
DEVICE = "cuda:0"

# Load the processor and VLA model from Hugging Face
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(DEVICE)

# Load image (use one in your images/ folder)
image = Image.open("images/exampleimage.jpeg").convert("RGB")

# Create a prompt
prompt = "In: What action should the robot take to pick up the bottle?\nOut:"

# Process the input
inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

# Run inference
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print("Predicted Action:", action)
