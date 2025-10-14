"""
 * Filename: convert.py
 *
 * @Author: Namcheol Lee
 * @Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * @Created: 10/06/25
 * @Contact: {nclee}@redwood.snu.ac.kr
 *
 * @Description: Converts .onnx to .tflite
 *
 """

import argparse
from onnx2tf import convert

def convert_onnx_to_litert(onnx_path, output_dir):
    convert(
        input_onnx_file_path=args.onnx_path,
        output_folder_path=args.output_dir,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=False,                              # Enable detailed output logs
        not_use_onnxsim=True,                           # Disable onnx-simplifier tool
        not_use_opname_auto_generate=True               # Disable automatic operation name generation
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .onnx ONNX model to .tflite.")
    parser.add_argument("--onnx-path", type=str , default="./models/simple_classifier.onnx", help="Path to the input .onnx model")
    parser.add_argument("--output-dir", type=str , default="./models/litert", help="Directory to save LiteRT model")
    args = parser.parse_args()

    convert_onnx_to_litert(args.onnx_path, args.output_dir)