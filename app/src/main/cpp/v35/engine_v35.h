#pragma once
/**
 * Local AI Painter v4.0 TaiShen — 引擎总入口
 * 集成: INT2/FP8 量化 + 图融合 + 双缓冲DMA + 天玑NPU + Vulkan零拷贝
 * 自动检测芯片并选择最优策略组合
 */
#include <string>
#include <vector>
#include <cstdint>
#include "v35/quantization/int2_fp8_quantizer.h"
#include "v35/optimization/graph_fusion.h"
#include "v35/optimization/double_buffered_pipeline.h"
#include "v35/engine_mediatek/mediatek_npu_adapter.h"
#include "v35/rendering/vulkan_zero_copy.h"

namespace locai::v35 {

// ═════════════════════════════════════════════════
//  平台类型
// ═════════════════════════════════════════════════

enum class Platform {
    UNKNOWN = 0,
    QUALCOMM_SNAPDRAGON,  // 骁龙 (QNN / Adreno Vulkan)
    MEDIATEK_DIMENSITY,   // 天玑 (NeuroPilot / Mali Vulkan)
    HISILICON_KIRIN,      // 麒麟 (华为, 受限)
    APPLE_SILICON,         // Apple (非 Android, 预留)
    SAMSUNG_EXYNOS,        // 三星 Exynos
};

enum class ChipTier {
    FLAGSHIP,    // 天玑 9500 / 骁龙 8 至尊 / 8 Gen3
    HIGH_END,    // 天玑 9400+ / 骁龙 8s Gen3
    MID_RANGE,   // 天玑 9400 / 骁龙 7+ Gen3
    ENTRY,       // 天玑 8000 系列 / 骁龙 6 系
    LEGACY,      // 更老的设备
};

// ═════════════════════════════════════════════════
//  引擎能力总览
// ═════════════════════════════════════════════════

struct EngineCapabilities {
    Platform    platform = Platform::UNKNOWN;
    ChipTier    tier = ChipTier::ENTRY;
    std::string chip_name;
    std::string gpu_name;
    std::string npu_name;

    // 量化支持
    bool int2_support = false;
    bool fp8_support  = false;
    bool int4_support = true;
    bool int8_support = true;

    // 后端支持
    bool vulkan_available = false;
    bool opengl_available  = false;
    bool opencl_available  = false;
    bool npu_available    = false;
    bool cim_available    = false; // 天玑 CIM

    // 精度
    bool fp16_support = true;
    bool fp32_support = true;

    // 内存
    size_t total_ram_mb = 0;
    size_t gpu_memory_mb = 0;
    size_t npu_sram_kb  = 0;

    // 推荐
    std::string recommended_quant = "INT8";
    std::string recommended_backend = "vulkan";
    int        recommended_threads = 4;
    int        recommended_resolution = 512;
    float      estimated_speedup = 1.0f;

    std::string to_string() const;
};

// ═════════════════════════════════════════════════
//  v4.0 引擎 (核心类)
// ═════════════════════════════════════════════════

class EngineV35 {
public:
    EngineV35();
    ~EngineV35();

    // ── 初始化 (自动检测一切) ────────────────
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }

    // 获取能力
    const EngineCapabilities& caps() const { return caps_; }

    // ── 一键最优配置 ──────────────────────────
    void apply_recommended_optimizations();

    // ── 子模块访问 ────────────────────────────
    quant::WeightStreamer*      weight_streamer() { return &weight_streamer_; }
    opt::GraphFusion*           graph_fusion()    { return &graph_fusion_; }
    opt::PipelineManager*       pipeline_mgr()   { return &pipeline_mgr_; }
    mediatek::MediatekNPUAdapter* mediatek_npu() { return &mediatek_npu_; }
    render::VulkanZeroCopyPipeline* vulkan_pipeline() { return &vulkan_pipeline_; }
    render::RenderManager*      render_mgr()     { return &render_mgr_; }

    // ── 完整推理流程 ──────────────────────────
    struct InferenceRequest {
        std::string model_path;
        std::string prompt;
        std::string negative_prompt;
        int    width = 512;
        int    height = 512;
        int    steps = 20;
        float  cfg_scale = 7.5f;
        int    seed = -1;
        std::string sampler = "Euler";
        std::string quantization = "auto"; // auto/INT2/INT4/INT8/FP8/FP16
        std::string backend = "auto";    // auto/vulkan/opengl/npu/cpu/hybrid
        bool   use_cim = true;            // 天玑 CIM
        bool   use_spd_plus = true;       // 投机解码
        int    batch_size = 1;
    };

    struct InferenceResult {
        bool   success = false;
        float* image_data = nullptr; // RGBA float [w*h*4]
        int    width = 0, height = 0;
        double total_time_ms = 0;
        double compute_time_ms = 0;
        double dma_time_ms = 0;
        double render_time_ms = 0;
        float  memory_peak_mb = 0;
        float  power_watts = 0;
        std::string quantization_used;
        std::string backend_used;
        int    steps_completed = 0;
        float  cache_hit_rate = 0;
    };

    InferenceResult run_inference(const InferenceRequest& req);

    // ── 进度回调 ──────────────────────────────
    using ProgressCallback = std::function<void(int step, int total, float progress, double eta_ms)>;
    void set_progress_callback(ProgressCallback cb) { progress_cb_ = cb; }

    // ── 取消 ──────────────────────────────────
    void cancel() { cancelled_ = true; }
    bool is_cancelled() const { return cancelled_; }

    // ── JNI 接口 (供 Android 层调用) ──────────
    // 以下方法对应 Java 端 native 调用
    static EngineV35* jni_get_instance();
    static void       jni_release_instance();

    // ── 性能报告 ──────────────────────────────
    struct FullReport {
        EngineCapabilities        caps;
        opt::PipelineManager::PipelineStats pipeline;
        mediatek::MediatekNPUAdapter::PerfCounters npu;
        render::VulkanZeroCopyPipeline::PerfStats render;
        float total_speedup_vs_v32;
        float total_power_save_pct;
    };
    FullReport generate_full_report() const;

    // ── 平台检测 (静态) ──────────────────────
    static Platform detect_platform();
    static ChipTier detect_tier(Platform p, const std::string& chip_name);
    static EngineCapabilities detect_all();

private:
    bool initialized_ = false;
    EngineCapabilities caps_;
    std::atomic<bool> cancelled_{false};
    ProgressCallback progress_cb_;

    // 子模块
    quant::WeightStreamer         weight_streamer_;
    opt::GraphFusion              graph_fusion_;
    opt::PipelineManager          pipeline_mgr_;
    mediatek::MediatekNPUAdapter  mediatek_npu_;
    render::VulkanZeroCopyPipeline vulkan_pipeline_;
    render::RenderManager         render_mgr_;

    // 内部方法
    void detect_platform_caps();
    void configure_for_mediatek();
    void configure_for_qualcomm();
    void configure_for_generic();
    std::string select_quantization(const InferenceRequest& req);
    std::string select_backend(const InferenceRequest& req);
    bool execute_inference_pipeline(const InferenceRequest& req, InferenceResult& result);
};

// ═════════════════════════════════════════════════
//  全局引擎实例
// ═════════════════════════════════════════════════

EngineV35* get_engine();
void        release_engine();

} // namespace locai::v35
