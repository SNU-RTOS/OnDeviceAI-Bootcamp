/*
 * Filename: util.hpp
 *
 * @Author: GeonhaPark
 * @Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * @Created: 07/23/25
 * @Original Work: Based on minimal-litert-c repository (https://github.com/SNU-RTOS/minimal-litert-c)
 * @Modified by: Namcheol Lee, Taehyun Kim on 10/16/25
 * @Contact: nclee@redwood.snu.ac.kr
 *
 * @Description: Utility functions
 * 
 */

#ifndef _UTIL_H_
#define _UTIL_H_

#include <unordered_map>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <string>
#include <thread>
#include <jsoncpp/json/json.h>
#include <opencv2/opencv.hpp> //opencv
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model.h"

namespace util
{
    // Alias for high-resolution clock and time point
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    // Struct to store start and end times with indices
    struct TimerResult
    {
        TimePoint start;
        TimePoint end;
        int start_index;
        int stop_index;
    };

    // Global timer map to store timing results
    static std::unordered_map<std::string, TimerResult> timer_map;

    // Global index for identifying timer start/stop order
    static int global_index = 0;

    // Start timing for a given label
    void timer_start(const std::string &label);

    // Stop timing for a given label
    void timer_stop(const std::string &label);

    // Calculate and print average latency for a given label
    void print_average_latency(const std::string &label);

    // Calculate and print throughput for a given label
    void print_throughput(const std::string &label, size_t num_inputs);

    // Load class labels from a JSON file
    std::unordered_map<int, std::string> load_class_labels(const std::string &json_path);

    // Get indices of top-K elements from a float vector
    std::vector<int> get_topK_indices(const std::vector<float> &data, int k);

    // Compute softmax probabilities from logits
    void softmax(const float *logits, std::vector<float> &probs, int size);

    // Preprocess input image to match model input size (normalization, resize, etc.)
    cv::Mat preprocess_image(cv::Mat &image, int target_height, int target_width);

} // namespace util

#endif // _UTIL_H_
