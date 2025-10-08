#!/bin/bash

#-----------------------------------------------------------------------------------------------
# Filename: run_inference_driver.sh
#
# @Author: Namcheol Lee
# @Affiliation: Real-Time Operating System Laboratory, Seoul National University
# @Created: 07/23/25
# @Contact: {ghpark,thkim,nclee}@redwood.snu.ac.kr
#
# @Description: Script to run inference driver for RTCSA25 tutorial
# 
#-----------------------------------------------------------------------------------------------

set -e  # Exit if any command fails

# --------- CONFIGURATION ---------
executable="./bin/inference_driver"
model="./models/tflite/simple_classifier_float32.tflite"
gpu_usage="true"
class_labels="class_labels.json"
base_dir="./data/MNIST/test"    
input_period_ms=0
total_inputs=100                
# ---------------------------------

# Sanity check for files and directories
if [ ! -f "$model" ]; then
    echo "ERROR: Model file not found: $model"
    exit 1
fi

if [ ! -d "$base_dir" ]; then
    echo "ERROR: Image directory not found: $base_dir"
    exit 1
fi

# Build image list dynamically
images=()
echo "Generating image list for the first $total_inputs images..."
for ((i=0; i<total_inputs; i++)); do

    # Format the filename with leading zeros
    filename=$(printf "%05d.png" $i)
    image_path="$base_dir/$filename"
    
    # Check if the image file exists
    if [ ! -f "$image_path" ]; then
        echo "ERROR: Image file not found: $image_path"
        echo "Please check if 'total_inputs' exceeds the number of available images."
        exit 1
    fi
    images+=("$image_path")
done

# Build input-period argument
period_arg="--input-period=$input_period_ms"

# Show the command
# echo "Running: $executable $model $gpu_usage $class_labels ${images[@]} $period_arg"

# Run
echo "Starting inference..."
taskset -c 7-7 "$executable" "$model" "$gpu_usage" "$class_labels" "${images[@]}" "$period_arg"
echo "Inference finished."