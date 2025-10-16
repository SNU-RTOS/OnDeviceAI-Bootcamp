"""
 * Filename: prepare_test_data.py
 *
 * @Author: Namcheol Lee
 * @Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * @Created: 10/06/25
 * @Modified by: Namcheol Lee, Taehyun Kim on 10/16/25
 * @Contact: nclee@redwood.snu.ac.kr
 *
 * @Description: Convert MNIST test dataset from .gz to .png
 *
 """

import os
import json

from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image

root = "./data"                       
out_dir = "./data/MNIST/test"         
os.makedirs(out_dir, exist_ok=True)

transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

labels = {}
for i in range(len(test_dataset)):               
    img, _ = test_dataset[i]                            # img: 1x28x28 tensor in [0,1]
    name = f"{i:05d}.png"                               # f"{i:05d}" turns i into a 5-digit string with zero padding, e.g., 0 -> "00000"
    to_pil_image(img).save(os.path.join(out_dir, name)) # Convert tensor to PIL image and save as PNG

print("Saved MNIST test dataset images to", out_dir)
