#pragma once
/**
 * 天玑 NPU 深度适配层 (v4.0 TaiShen)
 * 支持: 天玑 9500 (NPU 990) / 9400+ (NPU 890+) / 9400
 *
 * 功能矩阵:
 *   ✅ INT2 原生推理 (NPU 990)
 *   ✅ FP8 推理 (NPU 990 / 890+)
 *   ✅ 存算一体 CIM (NPU 990, 功耗 -33%)
 *   ✅ 双 NPU 调度 (NPU 990)
 *   ✅ 时域张量 5D (NPU 990 / 890+)
 *   ✅ MoE 推理 (NPU 990 / 890+)
 *   ✅ SpD+ 投机解码 (NPU 990 / 890+, +20%)
 *   ✅ 端侧 LoRA 训练 (全系, 50×CPU)
 *   ✅ 4K 图像生成 (NPU 990 业界首发)
 *   ✅ 128K 长文本 (NPU 990)
 */
#include <string>
#include <vector>
#include <cstdint>
#include <functional>

namespace locai::v35::mediatek {

// ═════════════════════════════════════════════════════════
//  芯片枚举
// ═════════════════════════════════════════════════════════

enum class DimensityChip {
    UNKNOWN = 0,
    D9400,         // 天玑 9400 — 首发端侧 LoRA 训练
    D9400_PLUS,    // 天玑 9400+ — NPU 890+, 时域张量 5D
    D9500,         // 天玑 9500 — NPU 990, 旗舰
};

enum class NpuGeneration {
    UNKNOWN = 0,
    NPU_790,   // D9300
    NPU_890,   // D9400
    NPU_890_PLUS, // D9400+
    NPU_990,   // D9500
};

// ═════════════════════════════════════════════════════════
//  能力检测
// ═════════════════════════════════════════════════════════

struct NPUCapabilities {
    DimensityChip   chip = DimensityChip::UNKNOWN;
    NpuGeneration   npu  = NpuGeneration::UNKNOWN;
    std::string     chip_name;
    int             npu_cores = 0;
    int             npu_freq_mhz = 0;
    int             total_npu_tops = 0; // INT8 TOPS
    size_t          npu_sram_kb = 0;

    // 特性开关
    bool int2_native    = false;
    bool fp8_inference  = false;
    bool cim_enabled    = false; // 存算一体
    bool dual_npu       = false;
    bool spatiotemporal_5d = false;
    bool moe_support    = false;
    bool spd_plus       = false; // 投机解码+
    bool lora_training  = false;
    bool gen_4k_image   = false;
    bool long_context_128k = false;

    int  max_context_len = 32000;
    int  max_image_res   = 2048;

    std::string to_string() const;
};

// ═════════════════════════════════════════════════════════
//  天玑 NPU 适配器 (核心类)
// ═════════════════════════════════════════════════════════

class MediatekNPUAdapter {
public:
    MediatekNPUAdapter();
    ~MediatekNPUAdapter();

    // ── 初始化 / 检测 ─────────────────────────────────
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }

    // 自动检测芯片型号
    NPUCapabilities detect_capabilities() const;

    // 获取当前能力 (初始化后有效)
    const NPUCapabilities& caps() const { return caps_; }

    // ── INT2 推理 (NPU 990) ──────────────────────────
    // 2-bit 权重 + INT8 激活 → INT16 输出
    bool supports_int2() const { return caps_.int2_native; }
    // 执行 INT2 推理
    int run_int2_inference(
        const uint8_t* weights_int2_packed, // 4 values per byte
        const int8_t*  activations_int8,
        int16_t*       output,
        int M, int N, int K,
        float weight_scale, float act_scale);

    // ── FP8 推理 (NPU 990 / 890+) ─────────────────────
    bool supports_fp8() const { return caps_.fp8_inference; }
    int run_fp8_inference(
        const uint8_t* weights_fp8,
        const uint8_t* activations_fp8,
        float*         output,
        int M, int N, int K,
        float scale_w, float scale_a);

    // ── CIM 存算一体 (NPU 990) ────────────────────────
    // 权重驻留 SRAM, 激活流入, 在存储单元内完成 MAC
    bool supports_cim() const { return caps_.cim_enabled; }
    void enable_cim(bool on) { cim_enabled_ = on && caps_.cim_enabled; }
    bool is_cim_enabled() const { return cim_enabled_; }
    // CIM 矩阵乘
    int run_cim_matmul(
        const float* weights_sram, // 已加载到 CIM SRAM
        const float* activation,
        float* output, int M, int N, int K);
    // 预加载权重到 CIM SRAM
    int cim_load_weights(const float* weights, size_t bytes);

    // ── 双 NPU 调度 (NPU 990) ─────────────────────────
    bool supports_dual_npu() const { return caps_.dual_npu; }
    void enable_dual_npu(bool on) { dual_npu_ = on && caps_.dual_npu; }
    // 拆分矩阵到两个 NPU
    int run_dual_npu_inference(
        const float* weights, const float* input,
        float* output, int M, int N, int K);

    // ── SpD+ 投机解码 (NPU 990 / 890+) ────────────────
    bool supports_spd_plus() const { return caps_.spd_plus; }
    struct SPDConfig {
        int draft_tokens = 4;
        float acceptance_threshold = 0.85f;
        bool use_int2_draft = true; // 草稿模型用 INT2
    };
    // 投机解码一步
    // 返回接受的 token 数量
    int speculative_decode_step(
        const float* prompt_embedding,
        float* output_tokens,
        int seq_len,
        const SPDConfig& cfg);

    // ── 端侧 LoRA 训练 (全系, 50×CPU) ────────────────
    bool supports_lora_training() const { return caps_.lora_training; }
    struct LoRATrainConfig {
        float lr = 1e-4f;
        int  rank = 8;
        int  steps = 100;
        bool use_fp8 = true;
    };
    // 在 NPU 上微调 LoRA 权重
    int train_lora_on_device(
        const float* base_weights,
        const float* training_data,
        float* lora_delta_weights, // output: ΔW
        int rows, int cols,
        const LoRATrainConfig& cfg);

    // ── 4K 图像生成 (NPU 990 业界首发) ───────────────
    bool supports_4k_generation() const { return caps_.gen_4k_image; }
    int generate_4k_image(
        const float* latent_input,  // [1,4,128,128] for 1024² → upscale to 4K
        float* output_image,         // [3, 3840, 2160] or [3, 4096, 4096]
        int target_width, int target_height);

    // ── 长上下文 (NPU 990: 128K) ──────────────────────
    bool supports_long_context(int tokens) const {
        return caps_.long_context_128k && tokens <= 131072;
    }

    // ── 时域张量 5D (NPU 990 / 890+) ──────────────────
    bool supports_spatiotemporal_5d() const { return caps_.spatiotemporal_5d; }
    // 5D 张量: [batch, time, channel, height, width]
    int run_5d_conv(
        const float* input_5d,
        float* output_5d,
        int batch, int time, int ch, int h, int w,
        const float* kernel, int kh, int kw);

    // ── MoE 推理 (NPU 990 / 890+) ──────────────────────
    bool supports_moe() const { return caps_.moe_support; }
    struct MoEConfig {
        int num_experts = 8;
        int top_k = 2;
        int hidden_dim = 4096;
    };
    int run_moe_inference(
        const float* input,
        float* output,
        int batch, int seq_len,
        const MoEConfig& cfg);

    // ── 性能监控 ──────────────────────────────────────
    struct PerfCounters {
        double last_inference_ms = 0;
        double avg_inference_ms = 0;
        float  power_watts = 0;
        float  npu_utilization = 0;  // 0-100%
        float  sram_hit_rate = 0;
        size_t total_inferences = 0;
    };
    const PerfCounters& perf() const { return perf_; }

    // 功耗估算
    float estimate_power_watts(int macc_count) const;

    // ── 生成 NeuroPilot 兼容图描述 ──────────────────────
    std::string emit_neuropilot_graph() const;

    // ── 芯片名称 → 枚举 ───────────────────────────────
    static DimensityChip parse_chip_name(const std::string& name);
    static std::string chip_to_string(DimensityChip c);

private:
    bool initialized_ = false;
    NPUCapabilities caps_;
    PerfCounters perf_;

    // 运行时状态
    bool cim_enabled_ = false;
    bool dual_npu_ = false;

    // CIM SRAM 模拟
    std::vector<float> cim_sram_;

    // 辅助
    void detect_from_sysfs(NPUCapabilities& c) const;
    void detect_from_build_prop(NPUCapabilities& c) const;
    void apply_chip_defaults(NPUCapabilities& c) const;
};

// ═════════════════════════════════════════════════════════
//  全局天玑优化器
// ═════════════════════════════════════════════════════════

class DimensityOptimizer {
public:
    DimensityOptimizer();
    ~DimensityOptimizer();

    // 获取适配器单例
    MediatekNPUAdapter* npu() { return &npu_adapter_; }

    // 一键最优配置 (根据检测到的芯片自动选择)
    void apply_optimal_settings();

    // 推荐配置报告
    struct Recommendation {
        std::string quantization = "INT8";   // INT2/INT4/INT8/FP8/FP16
        std::string attention = "flash_attn"; // flash_attn / xformers
        int  num_threads = 4;
        bool use_cim = false;
        bool use_dual_npu = false;
        bool use_spd_plus = false;
        int  preferred_resolution = 1024;
        std::string memory_strategy = "balanced";
        float estimated_speedup = 1.0f;
        float estimated_power_save = 0.f;
    };

    Recommendation get_recommendation() const;
    std::string recommendation_report() const;

private:
    MediatekNPUAdapter npu_adapter_;
    Recommendation rec_;
};

} // namespace locai::v35::mediatek
