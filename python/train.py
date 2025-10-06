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

# Import PyTorch, nn, and, optim
import torch
import torch.nn as nn
import torch.optim as optim

# Load dataset from torchvision.datasets
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()
train_data = datasets.MNIST(root = '../data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True) # An iterable of (batch of inputs, answers)
test_data = datasets.MNIST(root = '../data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True) # An iterable of (batch of inputs, answers)

# Create a model object
from model import SimpleDNN

model = SimpleDNN()

# Create objects for backpropagation and gradient descent
criterion = nn.CrossEntropyLoss() # Cross Entropy includes softmax
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# Train
epochs = 20
for epoch in range(epochs):
  for images, labels in train_loader:    # 60000 / 64 iterations in 1 epoch
    optimizer.zero_grad()
    pred = model(images.view(-1, 28*28)) # Flatten the input image
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
    pred = model(images.view(-1, 28*28))

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
dummy_input = torch.randn(1, 784)

# Export to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "../models/simplednn.onnx",        # output filename
    export_params=True,      # store trained parameter weights
    opset_version=11,        # ONNX opset version
    do_constant_folding=True, # optimize constant ops
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # allow variable batch
)

# Using torch pt
# torch.save(model, "../models/simplednn.pt")
# Using torch state_dict, only saved weights and biases
# torch.save(model.state_dict, "../models/simplednn_state_dict.pt") 