/*
 * Filename: util.hpp
 *
 * @Author: GeonhaPark
 * @Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * @Created: 07/23/25
 * @Original Work: Based on minimal-litert-c repository (https://github.com/SNU-RTOS/minimal-litert-c)
 * @Modified by: Namcheol Lee, Taehyun Kim on 10/16/25
 * @Contact: {nclee,ghpark,thkim}@redwood.snu.ac.kr
 *
 * @Description: Headers of utility functions
 * 
 */

#include "util.hpp"

// Get indices of top-K elements from a float vector
std::vector<int> util::get_topK_indices(const std::vector<float> &data, int k) {
    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(), indices.begin() + k, indices.end(),
        [&data](int a, int b)
        { return data[a] > data[b]; });
    indices.resize(k);
    return indices;
}

// Start timing for a given label
void util::timer_start(const std::string &label) {
    util::timer_map[label] = util::TimerResult{util::Clock::now(), util::TimePoint{}, util::global_index++};
}

// Stop timing for a given label
void util::timer_stop(const std::string &label) {
    auto it = util::timer_map.find(label);
    if (it != timer_map.end()) {
        it->second.end = Clock::now();
        it->second.stop_index = global_index++;
    } else {
        std::cerr << "[WARN] No active timer for label: " << label << std::endl;
    }
}

// Calculate and print average latency for a given label
void util::print_average_latency(const std::string &label) {
    std::vector<long long> latencies;

    for (const auto &[key, record] : util::timer_map) {
        if (key.find(label) != std::string::npos && record.end != util::TimePoint{}) {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(record.end - record.start).count();
            latencies.push_back(ms);
        }
    }

    if (!latencies.empty()) {
        long long sum = std::accumulate(latencies.begin(), latencies.end(), 0LL);
        double avg = static_cast<double>(sum) / latencies.size();
        std::cout << "\n[INFO] Average " << label << " latency (" 
                  << latencies.size() << " runs): " << avg << " ms" << std::endl;
    } else {
        std::cout << "\n[WARN] No measurements found for label containing \"" << label << "\"" << std::endl;
    }
}

// Calculate and print throughput for a given label
void util::print_throughput(const std::string &label, size_t num_inputs) {
    long long total_latency = -1;

    for (const auto &[key, record] : util::timer_map) {
        if (key == label && record.end != util::TimePoint{}) {
            total_latency = std::chrono::duration_cast<std::chrono::milliseconds>(record.end - record.start).count();
            break; // Assuming only one total latency measurement
        }
    }

    if (total_latency > 0 && num_inputs > 0) {
        double seconds = total_latency / 1000.0;
        double throughput = num_inputs / seconds;
        std::cout << "\n[INFO] Throughput: " 
                  << throughput << " items/sec (" << num_inputs 
                  << " items in " << total_latency << " ms)" << std::endl;
    } else {
        std::cout << "\n[WARN] Cannot calculate throughput for label \"" << label << "\"" << std::endl;
    }
}

// Load label file from JSON and return index → label map
std::unordered_map<int, std::string> util::load_class_labels(const std::string &json_path) {
    std::ifstream ifs(json_path, std::ifstream::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open label file: " + json_path);
    }
    
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errs;

    if (!Json::parseFromStream(builder, ifs, &root, &errs)) {
        throw std::runtime_error("Failed to parse JSON: " + errs);
    }

    std::unordered_map<int, std::string> label_map;

    for (const auto &key : root.getMemberNames()) {
        int idx = std::stoi(key);
        if (root[key].isArray() && root[key].size() >= 2) {
            label_map[idx] = root[key][1].asString(); // label = second element
        }
    }

    return label_map;
}

// Preprocess input image to match model input size (normalization, resize, etc.)
cv::Mat util::preprocess_image(cv::Mat &image, int target_height, int target_width) {
    // 1) Grayscale
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }                       

    // 2) Resize
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(target_width, target_height), 0, 0, cv::INTER_AREA);

    // 3) To float32 in [0,1]
    cv::Mat float_image;
    resized.convertTo(float_image, CV_32F, 1.0 / 255.0);

    // Ensure contiguous memory
    if (!float_image.isContinuous()) {
        float_image = float_image.clone();            // CV_32FC1, HxW
    }

    return float_image;
}

// Compute softmax probabilities from logits
void util::softmax(const float *logits, std::vector<float> &probs, int size) {
    float max_val = *std::max_element(logits, logits + size);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    if (sum > 0.0f) {
        for (int i = 0; i < size; ++i) {
            probs[i] /= sum;
        }
    }
}

