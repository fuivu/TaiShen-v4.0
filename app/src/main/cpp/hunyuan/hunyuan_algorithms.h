/*
 * ════════════════════════════════════════════════════════════
 *  混元 (HunYuan) — 推理算法混合调度  v4.0 TaiShen
 *  10 种算法 · 6 种策略 · 统一 C API
 * ════════════════════════════════════════════════════════════
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hunyuan {

// ─── 算法 ID ───────────────────────────────────────────────
enum AlgoID : int {
    ALGO_FP32_FULL = 0,
    ALGO_FP16_HALF,
    ALGO_INT8_QUANT,
    ALGO_SPARSE_ATTN,
    ALGO_SPECULATIVE,
    ALGO_FUSED_OPS,
    ALGO_MIXED_PRECISION,
    ALGO_INT4_EXTREME,
    ALGO_NPU_HW,
    ALGO_CASCADE_SR,
    ALGO_COUNT
};

const char* algo_name(AlgoID id);
int         algo_speed_tier(AlgoID id);   // 1=slow … 5=fast
float       algo_quality_score(AlgoID id); // 0…1
float       algo_power_eff(AlgoID id);    // 0…1

// ─── 策略 ───────────────────────────────────────────────────
enum Strategy : int {
    STRAT_MAX_SPEED = 0,
    STRAT_MAX_QUALITY,
    STRAT_POWER_SAVE,
    STRAT_BALANCED,
    STRAT_ADAPTIVE,
    STRAT_CUSTOM,
    STRAT_COUNT
};

const char* strategy_name(Strategy s);

// ─── 设备能力 ───────────────────────────────────────────────
struct DeviceCap {
    int   cpu_cores;
    float cpu_freq_ghz;
    int   ram_gb;
    bool  has_npu;
    bool  has_gpu;
    bool  has_vulkan;
    int   soc_tier; // 1=low 2=mid 3=high 4=flagship
};

// ─── 性能快照 ───────────────────────────────────────────────
struct PerfSnapshot {
    float cpu_usage;     // 0…1
    float cpu_temp_c;    // Celsius
    float mem_used_ratio; // 0…1
    float battery_pct;   // 0…100
    float battery_temp;   // C
    bool  charging;
    float gpu_usage;     // 0…1
    long  avail_mem_kb;
};

// ─── 算法排序 ───────────────────────────────────────────────
void sort_algos_by_strategy(AlgoID* out, Strategy s, const DeviceCap& dev, const PerfSnapshot& snap);

// ─── 算法执行 ──────────────────────────────────────────────
bool run_algorithm(AlgoID algo, const float* input, size_t n, float* output);

// ─── 引擎单例 ──────────────────────────────────────────────
class HybridEngine {
public:
    static HybridEngine& instance();
    void set_strategy(Strategy s);
    Strategy get_strategy() const;
    void set_custom_order(const AlgoID* order, int n);
    void feed_perf(const PerfSnapshot& snap);
    AlgoID pick_algorithm();
    bool execute(const float* input, size_t n, float* output);
    void set_device_cap(const DeviceCap& cap);
    std::string status_json() const;
    void reset_stats();
private:
    HybridEngine();
    ~HybridEngine();
    struct Impl; Impl* p;
};

} // namespace hunyuan
