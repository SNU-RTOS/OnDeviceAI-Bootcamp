# Exporting to .tflite using ai-edge-torch
import torch
from model import SimpleCNN
from ai_edge_torch import convert

# Dummy input for error checking during conversion
example_input = (torch.randn(1, 1, 784),)

model = SimpleCNN()
state = torch.load("../models/simplecnn.pt", map_location=torch.device('cpu'))
model.load_state_dict(state)
output_path = "../models/simplecnn.tflite"
edge_model = convert(model.eval(), example_input)
edge_model.export(output_path)

print(f"TFLite model saved to: {output_path}")