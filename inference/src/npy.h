// Minimal .npy reader/writer: float32, little-endian, C-order only.
// Enough for calibration sets and sample inputs — not a general parser.
#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct NpyArray {
    std::vector<size_t> shape;
    std::vector<float> data;
    size_t size() const {
        size_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

inline NpyArray npy_load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy_load: cannot open " + path);
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("npy_load: bad magic in " + path);
    uint8_t ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    uint32_t hlen = 0;
    if (ver[0] == 1) {
        uint16_t h16;
        f.read(reinterpret_cast<char*>(&h16), 2);
        hlen = h16;
    } else {
        f.read(reinterpret_cast<char*>(&hlen), 4);
    }
    std::string header(hlen, '\0');
    f.read(header.data(), hlen);
    if (header.find("'<f4'") == std::string::npos)
        throw std::runtime_error("npy_load: dtype must be float32 LE");
    if (header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error("npy_load: must be C-order");

    NpyArray a;
    auto lp = header.find('(');
    auto rp = header.find(')', lp);
    std::stringstream ss(header.substr(lp + 1, rp - lp - 1));
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.find_first_of("0123456789") != std::string::npos)
            a.shape.push_back(std::stoul(tok));
    }
    a.data.resize(a.size());
    f.read(reinterpret_cast<char*>(a.data.data()),
           static_cast<std::streamsize>(a.size() * sizeof(float)));
    if (!f) throw std::runtime_error("npy_load: truncated data in " + path);
    return a;
}

inline void npy_save(const std::string& path, const float* data,
                     const std::vector<size_t>& shape) {
    std::ostringstream sh;
    sh << "(";
    for (size_t i = 0; i < shape.size(); ++i)
        sh << shape[i] << (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
    sh << ")";
    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': " +
                       sh.str() + ", }";
    size_t base = 6 + 2 + 2;
    size_t pad = 64 - ((base + dict.size() + 1) % 64);
    dict += std::string(pad, ' ') + "\n";
    std::ofstream f(path, std::ios::binary);
    f.write("\x93NUMPY\x01\x00", 8);
    auto h16 = static_cast<uint16_t>(dict.size());
    f.write(reinterpret_cast<char*>(&h16), 2);
    f.write(dict.data(), static_cast<std::streamsize>(dict.size()));
    size_t n = 1;
    for (auto s : shape) n *= s;
    f.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(n * sizeof(float)));
}
