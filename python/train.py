"""
 * Filename: train.py
 *
 * @Author: Namcheol Lee
 * @Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * @Created: 10/06/25
 * @Contact: {nclee}@redwood.snu.ac.kr
 *
 * @Description: Trains a SimpleDNN and exports it into a .onnx file
 *
 """
# Import warnings to ignore DeprecationWarning from torch.onnx
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message=".*legacy TorchScript-based ONNX export.*") 

# Import PyTorch, nn, and, optim
import torch
import torch.nn as nn
import torch.optim as optim

# Load dataset from torchvision.datasets
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Make directory for exported ONNX model
import os

transform = transforms.ToTensor()
train_data = datasets.MNIST(root = '../data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True) # An iterable of (batch of inputs, answers)
test_data = datasets.MNIST(root = '../data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True) # An iterable of (batch of inputs, answers)

# Create a model object
from model import SimpleClassifier

model = SimpleClassifier()

# Create objects for backpropagation and gradient descent
criterion = nn.CrossEntropyLoss() # Cross Entropy includes softmax
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# Train
epochs = 1
for epoch in range(epochs):
  for images, labels in train_loader:    # 60000 / 64 iterations in 1 epoch
    optimizer.zero_grad()
    pred = model(images.view(images.size(0), -1)) # Flatten the input image
    loss = criterion(pred, labels)       # softmax is included in the loss function
    loss.backward()
    optimizer.step()

  if loss.item() < 0.5:
    break

  print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
  
# Evaluation
model.eval()
correct = 0
total = len(test_loader.dataset)

for images, labels in test_loader:
    pred = model(images.view(images.size(0), -1))

    # Postprocessing: hardmax is used for inference
    _, y = torch.max(pred, 1)

    # Accumulate the number of correct predictions
    correct += (y == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}% "
      f"({correct}/{total} correct)")
  
# Save the model
# Using ONNX
# Dummy input (batch_size=1, 784 features for MNIST)
dummy_input = torch.randn(1, 784) # Generalize 할 수 있나?

# Export to ONNX
os.makedirs("./models", exist_ok=True)
torch.onnx.export(
    model, 
    dummy_input, 
    "./models/simple_classifier.onnx",        # output filename
    input_names=['input'], 
    output_names=['output']
)