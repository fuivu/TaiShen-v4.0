#pragma once
/**
 * MediaTek Dimensity 8400 NPU Adapter
 * NPU 880 / NeuroPilot SDK 适配层
 * STUB 模式：缺 .so 时自动降级到 Vulkan → CPU
 */
#include <string>
#include <vector>
#include <cstdint>
#include <functional>

namespace locai::mediatek {

// ─── 芯片型号枚举 ──────────────────────────────────
enum class ChipModel {
    UNKNOWN,
    DIMENSITY_8400,    // NPU 880, INT4, L3 6MB, SLC 5MB
    DIMENSITY_9300,
    DIMENSITY_9400,
    DIMENSITY_9500,    // NPU 990, INT2, CIM
};

// ─── 能力检测 ──────────────────────────────────────
struct NpuCapabilities {
    bool   available        = false;
    ChipModel model          = ChipModel::UNKNOWN;
    std::string model_name  = "Unknown";
    int    npu_version      = 0;       // e.g. 880, 890, 990
    bool   supports_int8    = true;
    bool   supports_int4    = false;   // 8400 ✅
    bool   supports_int2    = false;   // 8400 ❌ (9500 ✅)
    bool   supports_fp8     = false;   // 8400 ❌ (9400+ ✅)
    bool   supports_cim     = false;   // 存算一体 (9500 ✅)
    bool   supports_dit     = true;    // DiT 架构 (8400+ ✅)
    bool   supports_agentic  = false;   // DAE 智能体 (9400+ ✅)
    int    l3_cache_mb      = 0;       // 8400: 6MB
    int    slc_cache_mb     = 0;       // 8400: 5MB
    int    num_npu_cores    = 0;
    bool   stub_mode        = true;     // 缺 .so 时为 true
    std::string sdk_version = "unknown";
};

// ─── 权重缓存策略 ──────────────────────────────────
struct WeightCacheStrategy {
    bool   pin_l3       = true;   // 钉住 L3 缓存
    bool   pin_slc      = true;   // 钉住 SLC
    int    l3_budget_mb = 6;     // 8400: 6MB L3
    int    slc_budget_mb= 5;     // 8400: 5MB SLC
    bool   double_buffer = true;  // 双缓冲预取
    int    prefetch_distance = 2;
    bool   enable_kv_cache  = true;
    int    kv_cache_tokens  = 512;
};

// ─── 推理上下文 ────────────────────────────────────
struct NpuContext {
    void*  handle       = nullptr;  // dlopen 句柄
    void*  npu_session  = nullptr;  // NeuroPilot session
    bool   initialized  = false;
    int    device_id    = 0;
    // 性能统计
    float  last_inference_ms = 0;
    float  avg_power_mw     = 0;
    int    total_runs        = 0;
};

// ─── 主适配器类 ────────────────────────────────────
class Dimensity8400Adapter {
public:
    Dimensity8400Adapter();
    ~Dimensity8400Adapter();

    // ── 初始化 / 销毁 ──
    bool  init();
    void  shutdown();

    // ── 能力检测 ──
    NpuCapabilities detect_capabilities() const;
    bool  is_available() const { return caps_.available; }
    bool  is_stub_mode() const { return caps_.stub_mode; }
    ChipModel get_model() const { return caps_.model; }

    // ── 模型加载 ──
    bool  load_model(const std::string& model_path,
                     const std::string& model_type = "unet");
    void  unload_model();

    // ── 推理 ──
    bool  run_inference(const void* input, void* output,
                        const std::vector<int>& input_shape,
                        std::vector<int>& output_shape);

    // ── 量化工具 ──
    std::vector<uint8_t> quantize_to_int4(const float* weights, int numel,
                                           int group_size = 32);
    std::vector<uint8_t> quantize_to_int8(const float* weights, int numel);

    // ── 缓存策略 ──
    void  set_cache_strategy(const WeightCacheStrategy& s);
    WeightCacheStrategy get_cache_strategy() const { return cache_strategy_; }

    // ── 性能统计 ──
    float get_avg_latency_ms() const;
    float get_avg_power_mw() const;
    void  reset_stats();

    // ── 文本编码（T5/CLIP）──
    bool  encode_text(const std::string& prompt,
                      std::vector<float>& embedding,
                      int embedding_dim = 768);

    // ── VAE 解码 ──
    bool  decode_latent(const float* latent, int latent_h, int latent_w,
                         uint8_t* rgb_output, int out_w, int out_h);

    // ── 降级通知 ──
    std::string get_status_message() const;

private:
    NpuCapabilities     caps_;
    NpuContext          ctx_;
    WeightCacheStrategy cache_strategy_;
    std::string         so_path_;
    bool                 so_loaded_ = false;

    // NeuroPilot SDK 函数指针（dlopen 动态加载）
    void*  dl_handle_   = nullptr;
    bool   load_neuropilot_so();
    void   close_neuropilot_so();

    // 降级路径
    bool  run_vulkan_fallback(const void* input, void* output,
                               const std::vector<int>& shape);
    bool  run_cpu_fallback(const void* input, void* output,
                            const std::vector<int>& shape);

    // 统计
    mutable std::mutex  stats_mu_;
    std::vector<float>  latency_history_;
};

// ─── 全局工厂函数 ──────────────────────────────────
Dimensity8400Adapter* create_dimensity_8400_adapter();
void                   destroy_dimensity_8400_adapter(Dimensity8400Adapter* p);

} // namespace locai::mediatek
