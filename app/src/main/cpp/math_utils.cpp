#include "math_utils.h"
#include <algorithm>

namespace math {

void softmax(float* data, size_t count) {
    float max_val = data[0];
    for (size_t i = 1; i < count; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    if (sum > 0) {
        for (size_t i = 0; i < count; i++) {
            data[i] /= sum;
        }
    }
}

void layer_norm(float* data, size_t count, float eps) {
    float mean = 0.0f, var = 0.0f;
    for (size_t i = 0; i < count; i++) mean += data[i];
    mean /= count;
    for (size_t i = 0; i < count; i++) {
        float d = data[i] - mean;
        var += d * d;
    }
    var /= count;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (size_t i = 0; i < count; i++) {
        data[i] = (data[i] - mean) * inv_std;
    }
}

void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

void matmul_int8(const int8_t* A, const int8_t* B, float* C, int M, int N, int K, float scale_a, float scale_b) {
    float scale = scale_a * scale_b;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[m * K + k] * (int32_t)B[k * N + n];
            }
            C[m * N + n] = (float)sum * scale;
        }
    }
}

void gelu(float* data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        float x = data[i];
        data[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

void silu(float* data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        data[i] = data[i] / (1.0f + expf(-data[i]));
    }
}

void rms_norm(float* data, size_t count, float eps) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < count; i++) sum_sq += data[i] * data[i];
    float rms = sqrtf(sum_sq / count + eps);
    for (size_t i = 0; i < count; i++) {
        data[i] /= rms;
    }
}

void attention_qkv(float* q, float* k, float* v, float* output, int seq_len, int head_dim, int num_heads) {
    int total = seq_len * head_dim * num_heads;
    // Simplified: just copy V (skip actual attention computation)
    for (int i = 0; i < total; i++) output[i] = v[i % (seq_len * head_dim)];
}

} // namespace math
