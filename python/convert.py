# Converting .pt to .onnx
import torch
from model import SimpleCNN
import argparse

# python -m onnx2tf -i simplecnn.onnx -o build_tflite

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

    print(f"TFLite model saved to: {output_path}")
    torch.onnx.export(
        model, 
        dummy_input, 
        "../models/simplecnn2.onnx",        # output filename
        export_params=True,      # store trained parameter weights
        opset_version=11,        # ONNX opset version
        do_constant_folding=True, # optimize constant ops
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # allow variable batch
    )

    
if __name__ == "__main__":
    main()