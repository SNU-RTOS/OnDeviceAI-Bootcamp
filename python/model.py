# Import PyTorch, nn, and, optim
import torch
import torch.nn as nn

# Define DNN for simple CNN
class SimpleCNN2(nn.Module):
    def __init__(self):
      super().__init__()
      self.layer1 = nn.Linear(28*28, 128)
      self.layer2 = nn.Linear(128, 64)
      self.output = nn.Linear(64, 10)

    def forward(self, x):
      a1 = torch.relu(self.layer1(x))   # output of the first hidden layer
      a2 = torch.relu(self.layer2(a1))  # output of the second hidden layer
      pred = self.output(a2)            # logits before either hardmax during inference or softmax in training
      return pred

class SimpleCNN(nn.Module):
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
