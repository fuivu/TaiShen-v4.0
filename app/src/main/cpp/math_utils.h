#pragma once
#include <vector>
#include <cmath>

namespace math {

void softmax(float* data, size_t count);
void layer_norm(float* data, size_t count, float eps = 1e-5f);
void matmul(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_int8(const int8_t* A, const int8_t* B, float* C, int M, int N, int K, float scale_a, float scale_b);
void gelu(float* data, size_t count);
void silu(float* data, size_t count);
void rms_norm(float* data, size_t count, float eps = 1e-6f);
void attention_qkv(float* q, float* k, float* v, float* output, int seq_len, int head_dim, int num_heads);

} // namespace math
