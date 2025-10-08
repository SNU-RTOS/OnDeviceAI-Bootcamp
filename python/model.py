"""
 * Filename: model.py
 *
 * @Author: Namcheol Lee
 * @Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * @Created: 10/06/25
 * @Contact: {nclee}@redwood.snu.ac.kr
 *
 * @Description: Model definition
 *
 """

# Import PyTorch, nn
import torch
import torch.nn as nn

# Define simple DNN
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)   # logits before either hardmax or softmax

        )

    def forward(self, x):
        return self.model(x)
