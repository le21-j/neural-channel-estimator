// Build (from ONNX) or deserialize (from cache) a TensorRT engine.
// Precision: FP32 default, --fp16 / --int8 flags. TensorRT 10.x API.
#pragma once
#include <NvInfer.h>

#include <memory>
#include <string>

struct Logger : nvinfer1::ILogger {
    void log(Severity sev, const char* msg) noexcept override;
};

class TrtEngine {
  public:
    // Builds from onnx_path (and caches to engine_path) unless engine_path
    // already exists, in which case it deserializes. calib_npy is required
    // only when int8 = true.
    TrtEngine(const std::string& onnx_path, const std::string& engine_path,
              bool fp16, bool int8, const std::string& calib_npy);

    nvinfer1::ICudaEngine* engine() { return engine_.get(); }
    nvinfer1::IExecutionContext* context() { return context_.get(); }

    std::string input_name;    // "ls_grid"
    std::string output_name;   // "h_refined"
    size_t input_bytes = 0;
    size_t output_bytes = 0;

  private:
    void build(const std::string& onnx_path, const std::string& engine_path,
               bool fp16, bool int8, const std::string& calib_npy);
    void load(const std::string& engine_path);
    void inspect();

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
};
