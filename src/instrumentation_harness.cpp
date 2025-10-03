/*
 * Filename: instrumentation_harness.cpp
 *
 * @Author: Namcheol Lee
 * @Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * @Created: 07/27/25
 * @Contact: nclee@redwood.snu.ac.kr
 *
 * @Description: Inference driver internals
 * 
 */

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/delegates/gpu/delegate.h"
#include "tflite/interpreter_builder.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "util.hpp"
#include "instrumentation_harness_utils.hpp"

/* ============ Function Naming Convention of LiteRT ============
 * Public C++ class methods: UpperCamelCase (e.g., BuildFromFile)
 * Internal helpers: snake_case (e.g., typed_input_tensor)
 * ============================================================ */

int main(int argc, char *argv[])
{
    /* Receive user input */
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] 
                << "<model_path> [gpu_usage]" // mandatory arguments
                << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];

    bool gpu_usage = false; // If true, GPU delegate is applied
    if(argc == 3) {
        const std::string gpu_usage_str = argv[2];
        if(gpu_usage_str == "true"){
            gpu_usage = true;
        }
    }

    /* Load .tflite model */
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model)
    {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    instrumentation::inspect_model_loading();

    /* Build interpreter */
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    if (!interpreter)
    {
        std::cerr << "Failed to Initialize Interpreter" << std::endl;
        return 1;
    }
    instrumentation::inspect_interpreter_instantiation(model.get(), resolver, interpreter.get());
    instrumentation::inspect_interpreter(interpreter.get());

    /* Apply either XNNPACK delegate or GPU delegate */
    TfLiteDelegate* xnn_delegate = TfLiteXNNPackDelegateCreate(nullptr);
    TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateV2Create(nullptr);
    bool delegate_applied = false;
    if(gpu_usage) {
        if (interpreter->ModifyGraphWithDelegate(gpu_delegate) == kTfLiteOk)
        {
            delegate_applied = true;
            // Delete unused delegate
            if(xnn_delegate) TfLiteXNNPackDelegateDelete(xnn_delegate);
        } else {
            std::cerr << "Failed to Apply GPU Delegate" << std::endl;
        }
    } else {
        if (interpreter->ModifyGraphWithDelegate(xnn_delegate) == kTfLiteOk)
        {
            delegate_applied = true;
            // Delete unused delegate
            if(gpu_delegate) TfLiteGpuDelegateV2Delete(gpu_delegate);
        } else {
            std::cerr << "Failed to Apply XNNPACK Delegate" << std::endl;
        }
    }
    instrumentation::inspect_interpreter_with_delegate(interpreter.get());

    /* Allocate Tensor */
    instrumentation::inspect_tensors(interpreter.get(), "Before Allocate Tensors");
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to Allocate Tensors" << std::endl;
        return 1;
    }
    instrumentation::inspect_tensors(interpreter.get(), "After Allocate Tensors");

    /* Preprocessing */
    // We skip the actual preprocessing step here, as it is not the focus of this code

    /* Inference */
    // We skip the actual inference step here, as it is not the focus of this code
    instrumentation::inspect_inference(interpreter.get());

    /* PostProcessing */
    // We skip the actual postprocessing step here, as it is not the focus of this code

    return 0;
}