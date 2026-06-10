#include "buffers.h"

DeviceBuffer::DeviceBuffer(size_t bytes) : bytes_(bytes) {
    CUDA_CHECK(cudaHostAlloc(&host_, bytes_, cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&dev_, bytes_));
}

DeviceBuffer::~DeviceBuffer() {
    cudaFreeHost(host_);
    cudaFree(dev_);
}

void DeviceBuffer::h2d(cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dev_, host_, bytes_, cudaMemcpyHostToDevice,
                               stream));
}

void DeviceBuffer::d2h(cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(host_, dev_, bytes_, cudaMemcpyDeviceToHost,
                               stream));
}
