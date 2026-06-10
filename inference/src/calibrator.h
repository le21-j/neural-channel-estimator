// INT8 entropy calibrator fed from a .npy calibration set
// (shape (N, 2, 14, 64) float32 — LS grids from the training distribution).
#pragma once
#include <NvInfer.h>

#include <memory>
#include <string>
#include <vector>

#include "buffers.h"
#include "npy.h"

class NpyEntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
  public:
    NpyEntropyCalibrator(const std::string& npy_path,
                         const std::string& input_name,
                         const std::string& cache_path);

    int getBatchSize() const noexcept override { return 1; }
    bool getBatch(void* bindings[], const char* names[],
                  int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void writeCalibrationCache(const void* cache,
                               size_t length) noexcept override;

  private:
    NpyArray data_;
    std::string input_name_;
    std::string cache_path_;
    std::vector<char> cache_;
    size_t sample_floats_ = 0;
    size_t next_ = 0;
    std::unique_ptr<DeviceBuffer> buf_;
};
