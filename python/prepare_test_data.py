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

import os
import json

from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image

root = "./data"                       # same root you used
out_dir = "./data/MNIST/test"         # where to dump PNGs
os.makedirs(out_dir, exist_ok=True)

transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

labels = {}
for i in range(len(test_dataset)):               
    img, label = test_dataset[i]                 # img: 1x28x28 tensor in [0,1]
    name = f"{i:05d}.png"
    to_pil_image(img).save(os.path.join(out_dir, name))
    labels[name] = int(label)

with open(os.path.join(out_dir, "data_labels.json"), "w") as f:
    json.dump(labels, f, indent=2)

print("Wrote", len(labels), "images to", out_dir)
print("Labels:", os.path.join(out_dir, "data_labels.json"))
