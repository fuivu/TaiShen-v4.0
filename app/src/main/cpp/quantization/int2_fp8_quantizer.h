#pragma once
/**
 * INT2 / FP8 Quantizer — ParetoQ + E4M3/E5M2
 * 权重流式加载 + LRU 缓存
 */
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <mutex>

namespace locai::quant {

// ─── INT2 Quantization (ParetoQ, Meta 2025) ───────────────
struct Int2Weight {
    std::vector<uint8_t> packed;  // 4×INT2 per byte
    std::vector<float>    scales;  // per-group scale
    std::vector<float>    zeros;   // per-group zero-point
    int group_size = 16;
    int numel = 0;
};

Int2Weight quantize_int2(const float* weights, int numel, int group_size = 16);
void       dequantize_int2(const Int2Weight& w, float* out);
float      compute_kl_divergence(const float* orig, const float* recon, int n);

// ─── FP8 Quantization (E4M3 / E5M2) ───────────────────────
enum class Fp8Format { E4M3, E5M2 };

std::vector<uint8_t> quantize_fp8_e4m3(const float* x, int n);
std::vector<uint8_t> quantize_fp8_e5m2(const float* x, int n);
std::vector<float>   dequantize_fp8(const uint8_t* x, int n, Fp8Format fmt);

// ─── INT4 Groupwise (天玑 8400 NPU 880 原生加速) ──────────
struct Int4Weight {
    std::vector<uint8_t> packed;  // 2×INT4 per byte
    std::vector<float>    scales;
    int group_size = 32;
    int numel = 0;
};

Int4Weight quantize_int4_groupwise(const float* w, int n, int gs = 32);

// ─── 权重流式加载器 ────────────────────────────────────────
class WeightStreamer {
public:
    explicit WeightStreamer(size_t max_cache_bytes = 512 * 1024 * 1024);
    ~WeightStreamer();

    void   load_index(const std::string& safetensors_path);
    void*  pin_weight(const std::string& key);   // 钉住（LRU）
    void   unpin_weight(const std::string& key);
    void*  stream_block(const std::string& key, int block_idx); // 异步预取
    float  cache_hit_rate() const;

private:
    struct CacheEntry {
        void* ptr;
        size_t size;
        int    last_used;
        bool   pinned;
    };
    std::unordered_map<std::string, CacheEntry> cache_;
    size_t max_cache_;
    size_t used_cache_;
    mutable std::mutex mu_;
    int    tick_ = 0;

    void evict_lru(size_t need);
};

} // namespace locai::quant
