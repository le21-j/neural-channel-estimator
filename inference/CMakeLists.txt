# Build (remote Linux + CUDA + TensorRT 10.x host ONLY — see CLAUDE.md GPU gates):
#   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
cmake_minimum_required(VERSION 3.20)
project(estimator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# TensorRT: honor -DTENSORRT_ROOT=/path if non-standard install
find_path(TENSORRT_INCLUDE NvInfer.h
          HINTS ${TENSORRT_ROOT} /usr/include/x86_64-linux-gnu /usr/local/TensorRT
          PATH_SUFFIXES include)
find_library(TENSORRT_LIB nvinfer
             HINTS ${TENSORRT_ROOT} /usr/lib/x86_64-linux-gnu /usr/local/TensorRT
             PATH_SUFFIXES lib lib64)
find_library(TENSORRT_ONNX_LIB nvonnxparser
             HINTS ${TENSORRT_ROOT} /usr/lib/x86_64-linux-gnu /usr/local/TensorRT
             PATH_SUFFIXES lib lib64)
if(NOT TENSORRT_INCLUDE OR NOT TENSORRT_LIB OR NOT TENSORRT_ONNX_LIB)
  message(FATAL_ERROR "TensorRT not found — set -DTENSORRT_ROOT=/path/to/TensorRT")
endif()

add_executable(estimator
  src/main.cpp
  src/trt_engine.cpp
  src/buffers.cpp
  src/calibrator.cpp)
target_include_directories(estimator PRIVATE ${TENSORRT_INCLUDE})
target_link_libraries(estimator PRIVATE
  ${TENSORRT_LIB} ${TENSORRT_ONNX_LIB} CUDA::cudart)
