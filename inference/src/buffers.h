// RAII pinned-host + device buffer pair with explicit H2D / D2H copies.
// Pinned memory keeps cudaMemcpyAsync truly async and measurable.
#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess)                                              \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(err__) + " at " +      \
                                     __FILE__ + ":" + std::to_string(__LINE__)); \
    } while (0)

class DeviceBuffer {
  public:
    explicit DeviceBuffer(size_t bytes);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void h2d(cudaStream_t stream);   // async pinned-host -> device
    void d2h(cudaStream_t stream);   // async device -> pinned-host

    float* host() { return static_cast<float*>(host_); }
    void* device() { return dev_; }
    size_t bytes() const { return bytes_; }

  private:
    size_t bytes_;
    void* host_ = nullptr;
    void* dev_ = nullptr;
};
