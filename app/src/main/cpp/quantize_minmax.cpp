#include <vector>
#include <cmath>
#include <algorithm>

namespace quantize {

// INT8 Min-Max linear quantization
struct QuantizedTensor {
    std::vector<int8_t> data;
    float scale;
    int zero_point;
};

QuantizedTensor quantize_minmax(const float* weights, size_t count) {
    // Find min/max
    float min_val = weights[0], max_val = weights[0];
    for (size_t i = 1; i < count; i++) {
        if (weights[i] < min_val) min_val = weights[i];
        if (weights[i] > max_val) max_val = weights[i];
    }
    
    float scale = (max_val > min_val) ? (max_val - min_val) / 255.0f : 1.0f;
    int zero_point = 0;
    
    QuantizedTensor result;
    result.data.resize(count);
    result.scale = scale;
    result.zero_point = zero_point;
    
    for (size_t i = 0; i < count; i++) {
        int q = (int)roundf((weights[i] - min_val) / scale);
        q = std::max(-128, std::min(127, q));
        result.data[i] = (int8_t)q;
    }
    
    return result;
}

void dequantize_minmax(const QuantizedTensor& q, float* output, size_t count) {
    for (size_t i = 0; i < count; i++) {
        output[i] = (float)q.data[i] * q.scale + q.zero_point;
    }
}

} // namespace quantize
