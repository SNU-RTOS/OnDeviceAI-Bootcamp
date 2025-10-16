#!/bin/bash

#-----------------------------------------------------------------------------------------------
# Filename: run_inference_driver.sh
#
# @Author: Namcheol Lee
# @Affiliation: Real-Time Operating System Laboratory, Seoul National University
# @Created: 07/23/25
# @Contact: {ghpark,thkim,nclee}@redwood.snu.ac.kr
#
# @Description: Script to run inference driver for Ondevice AI Bootcamp.
# 
#-----------------------------------------------------------------------------------------------

set -e  # Exit if any command fails

# --------- CONFIGURATION ---------
executable="./bin/inference_driver"
model="./models/litert/simple_classifier_float32.tflite"
gpu_usage="true"
class_labels="class_labels.json"
image_dir="./data/MNIST/test"
total_inputs=1000
# ---------------------------------

# Sanity check for files and directories
if [ ! -f "$model" ]; then
    echo "ERROR: Model file not found: $model"
    exit 1
fi

if [ ! -d "$image_dir" ]; then
    echo "ERROR: Image directory not found: $image_dir"
    exit 1
fi

# Build array of image paths
image_paths=()
echo "Generating array of the first $total_inputs image paths..."
for ((i=0; i<total_inputs; i++)); do

    # Format the filename with leading zeros
    filename=$(printf "%05d.png" $i)
    image_path="$image_dir/$filename"
    
    # Check if the image file exists
    if [ ! -f "$image_path" ]; then
        echo "ERROR: Image file not found: $image_path"
        echo "Please check if 'total_inputs' exceeds the number of available images."
        exit 1
    fi
    image_paths+=("$image_path")
done

# Run
echo "Starting inference..."
"$executable" "$model" "$gpu_usage" "$class_labels" "${image_paths[@]}"
echo "Inference finished."