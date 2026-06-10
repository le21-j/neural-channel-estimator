#include "trt_engine.h"

#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "calibrator.h"

void Logger::log(Severity sev, const char* msg) noexcept {
    if (sev <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
}

static bool file_exists(const std::string& p) {
    return std::ifstream(p).good();
}

TrtEngine::TrtEngine(const std::string& onnx_path,
                     const std::string& engine_path, bool fp16, bool int8,
                     const std::string& calib_npy) {
    if (file_exists(engine_path)) {
        std::cout << "deserializing cached engine " << engine_path << "\n";
        load(engine_path);
    } else {
        build(onnx_path, engine_path, fp16, int8, calib_npy);
    }
    inspect();
}

void TrtEngine::build(const std::string& onnx_path,
                      const std::string& engine_path, bool fp16, bool int8,
                      const std::string& calib_npy) {
    using namespace nvinfer1;
    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(logger_));
    auto network = std::unique_ptr<INetworkDefinition>(
        builder->createNetworkV2(0));  // explicit batch is default in TRT 10
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger_));
    if (!parser->parseFromFile(onnx_path.c_str(),
                               int(ILogger::Severity::kWARNING)))
        throw std::runtime_error("ONNX parse failed: " + onnx_path);

    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    std::unique_ptr<NpyEntropyCalibrator> calib;
    if (fp16) config->setFlag(BuilderFlag::kFP16);
    if (int8) {
        if (calib_npy.empty())
            throw std::runtime_error("--int8 requires --calib <file.npy>");
        config->setFlag(BuilderFlag::kINT8);
        calib = std::make_unique<NpyEntropyCalibrator>(
            calib_npy, network->getInput(0)->getName(), "calib.cache");
        config->setInt8Calibrator(calib.get());
    }

    auto blob = std::unique_ptr<IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!blob) throw std::runtime_error("engine build failed");

    std::ofstream out(engine_path, std::ios::binary);
    out.write(static_cast<const char*>(blob->data()),
              static_cast<std::streamsize>(blob->size()));
    std::cout << "built + cached engine -> " << engine_path << " ("
              << blob->size() / (1024.0 * 1024.0) << " MiB)\n";

    runtime_.reset(createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(blob->data(), blob->size()));
    context_.reset(engine_->createExecutionContext());
}

void TrtEngine::load(const std::string& engine_path) {
    std::ifstream f(engine_path, std::ios::binary | std::ios::ate);
    auto size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<char> blob(size);
    f.read(blob.data(), static_cast<std::streamsize>(size));

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), size));
    if (!engine_) throw std::runtime_error("engine deserialize failed");
    context_.reset(engine_->createExecutionContext());
}

void TrtEngine::inspect() {
    using namespace nvinfer1;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        Dims d = engine_->getTensorShape(name);
        size_t bytes = sizeof(float);
        for (int j = 0; j < d.nbDims; ++j)
            bytes *= static_cast<size_t>(d.d[j]);
        if (engine_->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            input_name = name;
            input_bytes = bytes;
        } else {
            output_name = name;
            output_bytes = bytes;
        }
    }
    if (input_bytes == 0 || output_bytes == 0)
        throw std::runtime_error("engine IO inspection failed");
}
