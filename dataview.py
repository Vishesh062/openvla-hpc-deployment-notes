import h5py
import os

# Path to your ALOHA dataset episode file.
# Update this to point to your local copy of the dataset.
# Example: /path/to/aloha_clean_dish/episode_0.hdf5
SAMPLE_PATH = os.environ.get(
    "ALOHA_SAMPLE_PATH",
    "/path/to/aloha_dataset/episode_0.hdf5"
)

def print_tree(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name} : shape {obj.shape}, dtype {obj.dtype}")
    else:
        print(f"{name}/")

with h5py.File(SAMPLE_PATH, "r") as f:
    print("\n--- Dataset Tree ---")
    f.visititems(print_tree)
