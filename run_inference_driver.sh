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
base_images=(
    "./data/MNIST/test/00000.png"
    "./data/MNIST/test/00001.png"
    "./data/MNIST/test/00002.png"
    "./data/MNIST/test/00003.png"
    "./data/MNIST/test/00004.png"
    "./data/MNIST/test/00005.png"
    "./data/MNIST/test/00006.png"
    "./data/MNIST/test/00007.png"
    "./data/MNIST/test/00008.png"
    "./data/MNIST/test/00009.png"
)
input_period_ms=0
total_inputs=100 # adjust as needed
# ---------------------------------

# Sanity check for files
for f in "$model" "${base_images[@]}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

# Build repeated image list (round-robin)
images=()
for ((i=0; i<total_inputs; i++)); do
    index=$(( i % ${#base_images[@]} ))
    images+=("${base_images[$index]}")
done

# Build input-period argument
period_arg="--input-period=$input_period_ms"

# Show the command
# echo "Running: $executable $model $gpu_usage $class_labels ${images[@]} $period_arg"

# Run
taskset -c 7-7 "$executable" "$model" "$gpu_usage" "$class_labels" "${images[@]}" "$period_arg"
