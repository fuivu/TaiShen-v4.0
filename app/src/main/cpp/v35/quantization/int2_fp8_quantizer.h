#pragma once
/**
 * INT2 / FP8 权重量化器 (v4.0 TaiShen)
 * 支持: INT2 (ParetoQ), FP8 E4M3, FP8 E5M2
 * 天玑 NPU 990 原生 INT2 加速, 骁龙 8 至尊原生 FP8
 */
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <string>

namespace locai::v35::quant {

// ── FP8 类型模拟 (IEEE 754 风格) ──────────────────────────────────
struct FP8_E4M3 {
    uint8_t bits; // 1 sign + 4 exp + 3 mantissa
    static constexpr int  EXP_BIAS = 7;
    static constexpr int  MAX_EXP  = 15;
    static constexpr int  MIN_EXP  = -6; // 非规格化下限
    static float to_float(uint8_t b) {
        int s = (b >> 7) & 1;
        int e = (b >> 3) & 0x0F;
        int m = b & 0x07;
        if (e == 0) { // 非规格化
            return (s ? -1.f : 1.f) * (float)m / 8.f * 0.125f;
        }
        if (e == 0x0F) return (s ? -1.f : 1.f) * INFINITY;
        return (s ? -1.f : 1.f) * (1.f + (float)m / 8.f) * std::pow(2.f, e - EXP_BIAS);
    }
    static uint8_t from_float(float v) {
        if (v == 0.f) return 0;
        int s = v < 0 ? 1 : 0; if (v < 0) v = -v;
        int e; float m = std::frexp(v, &e); e--; // 1 <= 2m < 2
        int exp = e + EXP_BIAS;
        if (exp >= MAX_EXP) return (s << 7) | 0x78; // saturate to max
        if (exp <= 0) { // 非规格化
            float frac = m * 8.f;
            return (s << 7) | (uint8_t)(frac + 0.5f);
        }
        int mantissa = (int)((m - 0.5f) * 16.f + 0.5f) & 0x07;
        return (s << 7) | ((exp & 0x0F) << 3) | mantissa;
    }
};

struct FP8_E5M2 {
    uint8_t bits;
    static constexpr int EXP_BIAS = 15;
    static float to_float(uint8_t b) {
        int s = (b >> 7) & 1, e = (b >> 2) & 0x1F, m = b & 0x03;
        if (e == 0) return (s?-1.f:1.f)*(float)m/4.f*0.25f;
        if (e == 0x1F) return (s?-1.f:1.f)*INFINITY;
        return (s?-1.f:1.f)*(1.f+(float)m/4.f)*std::pow(2.f,e-EXP_BIAS);
    }
    static uint8_t from_float(float v) {
        if (v==0.f) return 0;
        int s=v<0?1:0; if(v<0) v=-v;
        int e; float m=std::frexp(v,&e); e--;
        int exp=e+EXP_BIAS;
        if (exp>=31) return (s<<7)|0xF0;
        if (exp<=0) {float frac=m*4.f; return (s<<7)|(uint8_t)(frac+0.5f);}
        int mantissa=((int)((m-0.5f)*8.f+0.5f))&0x03;
        return (s<<7)|((exp&0x1F)<<2)|mantissa;
    }
};

// ── INT2 量化 (ParetoQ 风格, 含 zero level) ──────────────────────
// 量化级别: -2, -1, 0, +1, +2 (5 levels → 3 bits 存储, 或 packed 2-bit)
// ParetoQ 核心发现: 包含 zero 的 SEQ 量化函数优于对称量化
enum class Int2Scheme { SYMMETRIC, ASYMMETRIC_SEQ, PARETOQ };

struct Int2Config {
    Int2Scheme scheme = Int2Scheme::PARETOQ;
    float      scale  = 1.0f;  // per-tensor scale
    float      zero_point = 0.f;
    int        num_levels = 5;  // -2,-1,0,1,2
};

// 量化: FP32 → INT2 (packed: 4 个 INT2 值装进 1 个 uint8)
void quantize_int2(const float* src, uint8_t* dst_packed, int n, const Int2Config& cfg);
// 反量化: INT2 → FP32
void dequantize_int2(const uint8_t* src_packed, float* dst, int n, const Int2Config& cfg);
// 计算最优 scale (MSE 最小化)
float calibrate_int2_scale(const float* weights, int n, Int2Scheme scheme);

// ── FP8 量化 ──────────────────────────────────────────────────────
struct FP8Config {
    bool use_e4m3 = true; // false → E5M2
    float scale = 1.0f;   // per-tensor scaling
};

void quantize_fp8(const float* src, uint8_t* dst, int n, const FP8Config& cfg);
void dequantize_fp8(const uint8_t* src, float* dst, int n, const FP8Config& cfg);
float calibrate_fp8_scale(const float* weights, int n, bool e4m3);

// ── 混合精度矩阵乘法 ──────────────────────────────────────────────
// INT2 权重 × FP16 激活 → FP16 输出 (天玑 NPU 990 原生指令)
void matmul_int2x_fp16(const uint8_t* w_int2_packed, const float* act_fp16,
                       float* output, int M, int N, int K, const Int2Config& cfg);
// FP8 权重 × FP8 激活 → FP16 输出 (骁龙 8 至尊原生 FP8 张量核心)
void matmul_fp8(const uint8_t* w_fp8, const uint8_t* act_fp8,
                float* output, int M, int N, int K, const FP8Config& cfg);

// ── 权重流式加载器 ────────────────────────────────────────────────
// LRU 缓存 + 热点预取, 常驻内存可配置 (默认 512MB)
class WeightStreamer {
public:
    WeightStreamer(size_t resident_mb = 512);
    ~WeightStreamer();
    // 请求权重块, 命中缓存直接返回, 未命中从磁盘加载
    const float* request(const std::string& layer_name, const std::string& path);
    void prefetch(const std::string& layer_name); // 异步预取下一层
    void evict(const std::string& layer_name);
    size_t cache_hits() const { return hits_; }
    size_t cache_misses() const { return misses_; }
    float hit_rate() const { return (hits_+misses_)==0?0.f:(float)hits_/(hits_+misses_); }
    void set_resident_limit(size_t mb) { resident_limit_ = mb * 1024 * 1024; }
private:
    struct CacheEntry {
        std::vector<float> data;
        size_t size_bytes;
        uint64_t last_access;
    };
    std::unordered_map<std::string, CacheEntry> cache_;
    size_t resident_limit_;
    size_t current_bytes_;
    uint64_t clock_;
    size_t hits_, misses_;
    void evict_lru();
};

} // namespace locai::v35::quant
