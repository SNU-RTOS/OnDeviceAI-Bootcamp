"""
 * Filename: prepare_test_data.py
 *
 * @Author: Namcheol Lee
 * @Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * @Created: 10/06/25
 * @Contact: {nclee}@redwood.snu.ac.kr
 *
 * @Description: Convert test data from .gz to .png
 *
 """

import json
from pathlib import Path

from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image

# Find the base directory (the one that contains "data/")
def find_base_dir() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data").is_dir():
            return p
    raise RuntimeError("Base directory not found (expected a 'data/' folder).")

# Set up directories
BASE_DIR = find_base_dir()  # e.g., ~/OnDeviceAI-Bootcamp
root = BASE_DIR / "data"                   # OnDeviceAI-Bootcamp/data
out_dir = BASE_DIR / "data" / "MNIST" / "test"
out_dir.mkdir(parents=True, exist_ok=True)

# Load the test dataset
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(
    root=str(root), train=False, download=True, transform=transform
)

# Save images as .png and create a JSON file for labels
labels = {}
for i in range(len(test_dataset)):
    img, label = test_dataset[i]  # img: 1x28x28 tensor in [0,1]
    name = f"{i:05d}.png"
    to_pil_image(img).save(out_dir / name)
    labels[name] = int(label)

# Save labels to a JSON file
with open(out_dir / "data_labels.json", "w") as f:
    json.dump(labels, f, indent=2)

# Print summary
print("Wrote", len(labels), "images to", out_dir)
print("Labels:", out_dir / "data_labels.json")