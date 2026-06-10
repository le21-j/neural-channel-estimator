#include "calibrator.h"

#include <cstring>
#include <fstream>
#include <iostream>

NpyEntropyCalibrator::NpyEntropyCalibrator(const std::string& npy_path,
                                           const std::string& input_name,
                                           const std::string& cache_path)
    : data_(npy_load(npy_path)),
      input_name_(input_name),
      cache_path_(cache_path) {
    sample_floats_ = data_.size() / data_.shape[0];
    buf_ = std::make_unique<DeviceBuffer>(sample_floats_ * sizeof(float));
    std::cout << "calibrator: " << data_.shape[0] << " samples, "
              << sample_floats_ << " floats each\n";
}

bool NpyEntropyCalibrator::getBatch(void* bindings[], const char* names[],
                                    int nbBindings) noexcept {
    if (next_ >= data_.shape[0]) return false;  // calibration set exhausted
    std::memcpy(buf_->host(), data_.data.data() + next_ * sample_floats_,
                sample_floats_ * sizeof(float));
    buf_->h2d(nullptr);
    cudaStreamSynchronize(nullptr);
    for (int i = 0; i < nbBindings; ++i) {
        if (input_name_ == names[i]) bindings[i] = buf_->device();
    }
    ++next_;
    return true;
}

const void* NpyEntropyCalibrator::readCalibrationCache(
    size_t& length) noexcept {
    cache_.clear();
    std::ifstream f(cache_path_, std::ios::binary | std::ios::ate);
    if (!f) {
        length = 0;
        return nullptr;
    }
    cache_.resize(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(cache_.data(), static_cast<std::streamsize>(cache_.size()));
    length = cache_.size();
    std::cout << "calibrator: reusing cache " << cache_path_ << "\n";
    return cache_.data();
}

void NpyEntropyCalibrator::writeCalibrationCache(const void* cache,
                                                 size_t length) noexcept {
    std::ofstream f(cache_path_, std::ios::binary);
    f.write(static_cast<const char*>(cache),
            static_cast<std::streamsize>(length));
}
