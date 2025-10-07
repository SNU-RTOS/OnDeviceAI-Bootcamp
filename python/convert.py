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
# Equivalent commnad (Under OnDeviceAI-Bootcamp/models): 
# python -m onnx2tf -i ../models/simple_classifier.onnx -o ../models/tflite

from onnx2tf import convert

convert(
    input_onnx_file_path="../models/simple_classifier.onnx",
    output_folder_path="../models",
    copy_onnx_input_output_names_to_tflite=True,
    non_verbose=False,
)