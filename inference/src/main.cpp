// Standalone TensorRT channel-estimator benchmark.
//   ./estimator --onnx model.onnx --input sample_input.npy [--fp16|--int8
//   --calib calib.npy] [--iters 1000] [--out trt_out.npy]
// Reports per-stage latency (H2D / enqueue / D2H) and end-to-end mean/p99.
#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "buffers.h"
#include "npy.h"
#include "trt_engine.h"

struct Args {
    std::string onnx = "model.onnx";
    std::string engine;
    std::string input = "sample_input.npy";
    std::string calib;
    std::string out = "trt_out.npy";
    bool fp16 = false;
    bool int8 = false;
    int iters = 1000;
};

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&] { return std::string(argv[++i]); };
        if (s == "--onnx") a.onnx = next();
        else if (s == "--engine") a.engine = next();
        else if (s == "--input") a.input = next();
        else if (s == "--calib") a.calib = next();
        else if (s == "--out") a.out = next();
        else if (s == "--fp16") a.fp16 = true;
        else if (s == "--int8") a.int8 = true;
        else if (s == "--iters") a.iters = std::stoi(next());
        else { std::cerr << "unknown arg " << s << "\n"; std::exit(2); }
    }
    if (a.engine.empty())
        a.engine = a.int8 ? "estimator_int8.engine"
                  : a.fp16 ? "estimator_fp16.engine"
                           : "estimator_fp32.engine";
    return a;
}

struct Stat {
    std::vector<float> v;
    void add(float x) { v.push_back(x); }
    float mean() const {
        return std::accumulate(v.begin(), v.end(), 0.0f) / float(v.size());
    }
    float p99() {
        std::sort(v.begin(), v.end());
        return v[size_t(0.99 * double(v.size() - 1))];
    }
};

int main(int argc, char** argv) {
    Args args = parse(argc, argv);
    TrtEngine trt(args.onnx, args.engine, args.fp16, args.int8, args.calib);

    NpyArray in = npy_load(args.input);
    if (in.size() * sizeof(float) != trt.input_bytes) {
        std::cerr << "input size mismatch: npy " << in.size() << " floats vs engine "
                  << trt.input_bytes / sizeof(float) << "\n";
        return 1;
    }

    DeviceBuffer din(trt.input_bytes), dout(trt.output_bytes);
    std::memcpy(din.host(), in.data.data(), trt.input_bytes);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    trt.context()->setTensorAddress(trt.input_name.c_str(), din.device());
    trt.context()->setTensorAddress(trt.output_name.c_str(), dout.device());

    cudaEvent_t e0, e1, e2, e3;
    for (auto* e : {&e0, &e1, &e2, &e3}) CUDA_CHECK(cudaEventCreate(e));

    for (int i = 0; i < 50; ++i) {  // warmup
        din.h2d(stream);
        trt.context()->enqueueV3(stream);
        dout.d2h(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Stat h2d, krn, d2h, e2e;
    for (int i = 0; i < args.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(e0, stream));
        din.h2d(stream);
        CUDA_CHECK(cudaEventRecord(e1, stream));
        trt.context()->enqueueV3(stream);
        CUDA_CHECK(cudaEventRecord(e2, stream));
        dout.d2h(stream);
        CUDA_CHECK(cudaEventRecord(e3, stream));
        CUDA_CHECK(cudaEventSynchronize(e3));
        float a, b, c, d;
        CUDA_CHECK(cudaEventElapsedTime(&a, e0, e1));  // ms
        CUDA_CHECK(cudaEventElapsedTime(&b, e1, e2));
        CUDA_CHECK(cudaEventElapsedTime(&c, e2, e3));
        CUDA_CHECK(cudaEventElapsedTime(&d, e0, e3));
        h2d.add(a * 1e3f);  // -> us
        krn.add(b * 1e3f);
        d2h.add(c * 1e3f);
        e2e.add(d * 1e3f);
    }

    const char* prec = args.int8 ? "INT8" : args.fp16 ? "FP16" : "FP32";
    std::cout << "\nprecision " << prec << ", " << args.iters << " iters\n";
    std::cout << "stage      mean(us)   p99(us)\n";
    auto row = [](const char* n, Stat& s) {
        std::printf("%-9s %9.2f %9.2f\n", n, double(s.mean()), double(s.p99()));
    };
    row("H2D", h2d);
    row("enqueue", krn);
    row("D2H", d2h);
    row("e2e", e2e);

    npy_save(args.out, dout.host(), {1, 2, 14, 64});
    std::cout << "wrote " << args.out << " (compare with scripts/compare_trt_onnx.py)\n";

    for (auto* e : {e0, e1, e2, e3}) cudaEventDestroy(e);
    cudaStreamDestroy(stream);
    return 0;
}
