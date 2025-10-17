"""
 * Filename: train.py
 *
 * @Author: Namcheol Lee
 * @Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * @Created: 10/06/25
 * @Modified by: Namcheol Lee, Taehyun Kim on 10/16/25
 * @Contact: nclee@redwood.snu.ac.kr
 *
 * @Description: Trains a SimpleDNN and exports it into a .onnx file
 *
 """

# Import os to create directories
import os

# Import warnings to ignore DeprecationWarning
import warnings  
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import PyTorch, nn, and, optim


# Import the model


# Download MNIST dataset from torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# Create a model object
model

# Create objects for backpropagation and gradient descent
criterion = 
optimizer = 

# Train the model
epochs = 20
for epoch in range(epochs):
    for images, labels in train_loader:               # 60000 / 64 iterations in 1 epoch
        # Write code for training
        ## Hint: use ".view(images.size(0), -1)" to flatten the input images
        
        
        

    if loss.item() < 0.5:
        break

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate the model with the test dataset
model.eval()
correct = 0
total = len(test_loader.dataset)

for images, labels in test_loader:
    pred = 

    # Postprocessing: hardmax is used for inference
    _, y = 

    # Accumulate the number of correct predictions
    correct += (y == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}% "
      f"({correct}/{total} correct)")
  
# Prepare a dummy input for ONNX export
sample_image, _ =
num_features =               ## Hint: use ".numel()" to get the total number of features
dummy_input = 

# Export to ONNX
# Call torch.onnx.export







print("Successfully saved simple_classifier.onnx in ./models")