/**
 * INT2 / FP8 量化器实现 (v4.0 TaiShen)
 */
#include "v35/quantization/int2_fp8_quantizer.h"
#include <cstring>
#include <cmath>
#include <numeric>
#include <queue>

namespace locai::v35::quant {

// ══════════════════════════════════════════════════════════════
//  INT2 Quantization
// ══════════════════════════════════════════════════════════════

// SEQ 量化函数 (ParetoQ): levels = [-2,-1,0,1,2]
// 包含 zero 比对称 [-2,-1,1,2] 更优 (ParetoQ NeurIPS 2025)
static int quantize_scalar_seq(float v, float scale) {
    float q = v / scale;
    if (q >= 2.f) return 2;
    if (q >= 1.f) return 1;
    if (q >= 0.f) return 0; // zero level
    if (q >= -1.f) return -1;
    return -2;
}

static int quantize_scalar_symmetric(float v, float scale) {
    float q = v / scale;
    if (q >= 1.5f) return 2;
    if (q >= 0.5f) return 1;
    if (q >= -0.5f) return 0;
    if (q >= -1.5f) return -1;
    return -2;
}

void quantize_int2(const float* src, uint8_t* dst_packed, int n, const Int2Config& cfg) {
    // 4 个 INT2 值打包进 1 byte: [v0,v1,v2,v3] → bits [1:0][3:2][5:4][7:6]
    int packed_n = (n + 3) / 4;
    std::memset(dst_packed, 0, packed_n);
    for (int i = 0; i < n; i++) {
        int q = (cfg.scheme == Int2Scheme::SYMMETRIC)
            ? quantize_scalar_symmetric(src[i], cfg.scale)
            : quantize_scalar_seq(src[i], cfg.scale);
        // map [-2,-1,0,1,2] → [0,1,2,3,4] (2-bit values)
        uint8_t v2 = (q + 2) & 0x03;
        dst_packed[i >> 2] |= (v2 << ((i & 3) * 2));
    }
}

void dequantize_int2(const uint8_t* src_packed, float* dst, int n, const Int2Config& cfg) {
    for (int i = 0; i < n; i++) {
        uint8_t v2 = (src_packed[i >> 2] >> ((i & 3) * 2)) & 0x03;
        int q = (int)v2 - 2; // [0,1,2,3,4] → [-2,-1,0,1,2]
        dst[i] = (float)q * cfg.scale;
    }
}

float calibrate_int2_scale(const float* weights, int n, Int2Scheme scheme) {
    // 使用绝对最大值法 (简单有效)
    float max_abs = 0.f;
    for (int i = 0; i < n; i++) {
        float a = std::fabs(weights[i]);
        if (a > max_abs) max_abs = a;
    }
    // 映射到 [-2,2] 范围 → scale = max_abs / 2
    float scale = max_abs / 2.f;
    if (scale < 1e-8f) scale = 1e-8f;
    return scale;
}

// ══════════════════════════════════════════════════════════════
//  FP8 Quantization
// ══════════════════════════════════════════════════════════════

void quantize_fp8(const float* src, uint8_t* dst, int n, const FP8Config& cfg) {
    if (cfg.use_e4m3) {
        for (int i = 0; i < n; i++) dst[i] = FP8_E4M3::from_float(src[i] * cfg.scale);
    } else {
        for (int i = 0; i < n; i++) dst[i] = FP8_E5M2::from_float(src[i] * cfg.scale);
    }
}

void dequantize_fp8(const uint8_t* src, float* dst, int n, const FP8Config& cfg) {
    if (cfg.use_e4m3) {
        for (int i = 0; i < n; i++) dst[i] = FP8_E4M3::to_float(src[i]) / cfg.scale;
    } else {
        for (int i = 0; i < n; i++) dst[i] = FP8_E5M2::to_float(src[i]) / cfg.scale;
    }
}

float calibrate_fp8_scale(const float* weights, int n, bool e4m3) {
    float max_abs = 0.f;
    for (int i = 0; i < n; i++) {
        float a = std::fabs(weights[i]);
        if (a > max_abs) max_abs = a;
    }
    // E4M3 max normal ≈ 448, E5M2 max ≈ 57344
    float max_val = e4m3 ? 240.f : 50000.f; // 留一些余量
    float scale = max_abs / max_val;
    if (scale < 1e-8f) scale = 1e-8f;
    return scale;
}

// ══════════════════════════════════════════════════════════════
//  INT2 × FP16 MatMul (模拟: INT2 权重解包 → FP32 累加)
// ══════════════════════════════════════════════════════════════

void matmul_int2x_fp16(const uint8_t* w_int2_packed, const float* act,
                        float* output, int M, int N, int K, const Int2Config& cfg) {
    // output[M][N] = act[M][K] × dequant(w[N][K])
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++) {
                uint8_t v2 = (w_int2_packed[(n * K + k) >> 2] >> (((n * K + k) & 3) * 2)) & 0x03;
                int q = (int)v2 - 2;
                acc += act[m * K + k] * ((float)q * cfg.scale);
            }
            output[m * N + n] = acc;
        }
    }
}

void matmul_fp8(const uint8_t* w_fp8, const uint8_t* act_fp8,
                float* output, int M, int N, int K, const FP8Config& cfg) {
    // 参考实现: 解包 FP8 → FP32 → 乘加
    // 天玑/骁龙硬件上这一步由 NPU/张量核心原生完成
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++) {
                float w = cfg.use_e4m3
                    ? FP8_E4M3::to_float(w_fp8[n * K + k])
                    : FP8_E5M2::to_float(w_fp8[n * K + k]);
                float a = cfg.use_e4m3
                    ? FP8_E4M3::to_float(act_fp8[m * K + k])
                    : FP8_E5M2::to_float(act_fp8[m * K + k]);
                acc += w * a;
            }
            output[m * N + n] = acc / (cfg.scale * cfg.scale);
        }
    }
}

// ══════════════════════════════════════════════════════════════
//  Weight Streamer (LRU + 热点预取)
// ══════════════════════════════════════════════════════════════

WeightStreamer::WeightStreamer(size_t resident_mb)
    : resident_limit_(resident_mb * 1024 * 1024),
      current_bytes_(0), clock_(0), hits_(0), misses_(0) {}

WeightStreamer::~WeightStreamer() = default;

const float* WeightStreamer::request(const std::string& name, const std::string& path) {
    clock_++;
    auto it = cache_.find(name);
    if (it != cache_.end()) {
        it->second.last_access = clock_;
        hits_++;
        return it->second.data.data();
    }
    misses_++;
    // 未命中: 从磁盘加载 (模拟: 实际应 mmap / fread)
    // 这里创建占位数据
    size_t sz = 1024 * 1024; // 默认 1MB 占位
    while (current_bytes_ + sz > resident_limit_) evict_lru();
    CacheEntry entry;
    entry.data.resize(sz / sizeof(float));
    entry.size_bytes = sz;
    entry.last_access = clock_;
    cache_[name] = std::move(entry);
    current_bytes_ += sz;
    return cache_[name].data.data();
}

void WeightStreamer::prefetch(const std::string& name) {
    // 异步预取 (简化: 同步加载)
    // 实际实现: std::async + 后台线程
    clock_++;
    auto it = cache_.find(name);
    if (it == cache_.end()) {
        size_t sz = 1024 * 1024;
        while (current_bytes_ + sz > resident_limit_) evict_lru();
        CacheEntry entry;
        entry.data.resize(sz / sizeof(float));
        entry.size_bytes = sz;
        entry.last_access = clock_;
        cache_[name] = std::move(entry);
        current_bytes_ += sz;
    }
}

void WeightStreamer::evict(const std::string& name) {
    auto it = cache_.find(name);
    if (it != cache_.end()) {
        current_bytes_ -= it->second.size_bytes;
        cache_.erase(it);
    }
}

void WeightStreamer::evict_lru() {
    if (cache_.empty()) return;
    uint64_t oldest = UINT64_MAX;
    std::string oldest_key;
    for (const auto& kv : cache_) {
        if (kv.second.last_access < oldest) {
            oldest = kv.second.last_access;
            oldest_key = kv.first;
        }
    }
    evict(oldest_key);
}

} // namespace locai::v35::quant
