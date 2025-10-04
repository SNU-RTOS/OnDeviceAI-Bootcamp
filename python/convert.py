# Exporting to .tflite using ai-edge-torch
import torch
from model import SimpleCNN
from ai_edge_torch import convert
import argparse


def main():
    # Dummy input for error checking during conversion
    example_input = (torch.randn(1, 1, 784),)

    # model = torch.load("../models/simplecnn.pt", map_location=torch.device('cpu'))
    # print(dir(model))

    # Alternatively, load the model using state_dict
    model = SimpleCNN()
    state = torch.load("../models/simplecnn_state_dict.pt", map_location=torch.device('cpu'))
    model.load_state_dict(state)
    # print(dir(model))
    
    output_path = "../models/simplecnn2.tflite"
    edge_model = convert(model.eval(), example_input)
    edge_model.export(output_path)

    print(f"TFLite model saved to: {output_path}")
    
if __name__ == "__main__":
    main()