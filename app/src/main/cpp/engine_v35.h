#pragma once
/**
 * Engine V3.5 — TaiShen
 * 总入口：平台自动检测 + 一键最优配置
 * 支持：CPU / Vulkan / OpenCL / NPU(QNN) / MNN / NCNN / ORT / 天玑 NPU
 */
#include <string>
#include <vector>
#include <memory>
#include <functional>

// 子模块头文件
#include "quantization/int2_fp8_quantizer.h"
#include "optimization/graph_fusion.h"
#include "optimization/double_buffered_pipeline.h"
#include "mediatek/dimensity8400_adapter.h"
#include "rendering/vulkan_zero_copy.h"

namespace locai::v35 {

// ─── 平台类型 ──────────────────────────────────
enum class Platform {
    UNKNOWN,
    QUALCOMM_SNAPDRAGON,   // 骁龙
    MEDIATEK_DIMENSITY,     // 天玑
    HUAWEI_KIRIN,          // 麒麟
    SAMSUNG_EXYNOS,        // 猎户座
    APPLE_SILICON,         // A 系列
    GENERIC_ARM,           // 通用 ARM
    X86_64,                // PC 模拟
};

// ─── 推理后端 ──────────────────────────────────
enum class Backend {
    CPU,
    OPENCL,
    VULKAN,
    OPENGL_ES,
    NPU_QNN,        // 高通
    NPU_MEDIATEK,   // 天玑
    MNN,
    NCNN,
    ORT,
    HYBRID,         // 混动模式
};

// ─── 混动策略 ──────────────────────────────────
enum class HybridStrategy {
    SEQUENTIAL,        // 串行
    PIPELINED,         // 流水线
    DATA_PARALLEL,     // 数据并行
    ADAPTIVE,          // 自适应（推荐）
};

// ─── 平台信息 ──────────────────────────────────
struct PlatformInfo {
    Platform    platform       = Platform::UNKNOWN;
    std::string soc_model     = "Unknown";
    std::string cpu_info      = "";
    int         cpu_cores      = 0;
    int         big_cores      = 0;
    int         little_cores   = 0;
    bool        has_vulkan     = false;
    bool        has_opencl     = false;
    bool        has_npu        = false;
    std::string npu_name      = "";
    int         npu_version    = 0;
    size_t      total_ram_mb   = 0;
    size_t      available_ram_mb = 0;
    std::string android_version = "";
    int         api_level      = 0;
};

// ─── 推荐配置 ──────────────────────────────────
struct RecommendedConfig {
    Backend         primary_backend   = Backend::CPU;
    Backend         secondary_backend = Backend::CPU;
    HybridStrategy  hybrid_strategy  = HybridStrategy::ADAPTIVE;
    bool            use_int8         = true;
    bool            use_int4         = false;
    bool            use_int2         = false;
    bool            use_fp8          = false;
    bool            use_graph_fusion = true;
    bool            use_double_buffer= true;
    bool            use_kv_cache    = true;
    bool            use_zero_copy   = true;
    int             thread_count    = 4;
    int             max_memory_mb   = 2048;
    std::string     summary() const;
};

// ─── 性能报告 ──────────────────────────────────
struct PerfReport {
    std::string backend_name;
    float       avg_inference_ms   = 0;
    float       peak_memory_mb     = 0;
    float       avg_power_mw       = 0;
    float       thermal_throttle   = 0;  // 0~1
    int         successful_runs    = 0;
    int         failed_runs        = 0;
    std::string bottleneck;  // "memory", "compute", "io"
    std::string suggestions; // 优化建议
};

// ─── 引擎总入口 ────────────────────────────────
class EngineV35 {
public:
    EngineV35();
    ~EngineV35();

    // ── 初始化 ──
    bool  init();
    void  shutdown();

    // ── 平台检测 ──
    PlatformInfo   detect_platform() const;
    static PlatformInfo get_platform_info();  // 静态快捷方式

    // ── 推荐配置 ──
    RecommendedConfig get_recommended_config() const;
    void  apply_recommended_config();

    // ── 手动配置 ──
    void  set_backend(Backend primary, Backend secondary = Backend::CPU);
    void  set_hybrid_strategy(HybridStrategy s);
    void  enable_quantization(bool int8, bool int4 = false,
                               bool int2 = false, bool fp8 = false);
    void  enable_graph_fusion(bool on);
    void  enable_double_buffering(bool on);
    void  enable_kv_cache(bool on, int tokens = 512);
    void  enable_zero_copy_render(bool on);
    void  set_thread_count(int n);

    // ── 推理 ──
    bool  run_text_encoding(const std::string& prompt,
                             std::vector<float>& embedding);
    bool  run_unet_inference(const std::vector<float>& latent,
                               const std::vector<float>& conditioning,
                               std::vector<float>& output);
    bool  run_vae_decoding(const std::vector<float>& latent,
                             std::vector<uint8_t>& rgb_image,
                             int out_w, int out_h);

    // ── 完整管线（一步到位）──
    struct GenerateParams {
        std::string prompt;
        std::string negative_prompt;
        int         width          = 512;
        int         height         = 512;
        int         steps           = 20;
        float       cfg_scale       = 7.5f;
        int         seed            = -1;  // -1 = random
        std::string sampler         = "euler_a";
        std::string model_path;
        std::vector<std::string> lora_paths;
        std::string controlnet_type;  // "canny", "pose", "" 
    };
    struct GenerateResult {
        bool        success     = false;
        std::string error_msg;
        std::vector<uint8_t> image_rgb;
        int         image_w     = 0;
        int         image_h     = 0;
        float       total_time_ms = 0;
        float       encode_ms    = 0;
        float       denoise_ms   = 0;
        float       decode_ms    = 0;
        float       peak_memory_mb = 0;
        float       avg_power_mw  = 0;
    };
    GenerateResult generate(const GenerateParams& params);

    // ── 性能报告 ──
    PerfReport  get_perf_report() const;
    void        reset_stats();

    // ── 能力查询 ──
    bool        is_backend_available(Backend b) const;
    std::vector<Backend> list_available_backends() const;
    std::string get_engine_version() const { return "4.0.0 TaiShen"; }
    std::string get_build_info() const;

    // ── 回调 ──
    using ProgressCallback = std::function<void(int step, int total, float progress)>;
    void set_progress_callback(ProgressCallback cb);

    // ── 取消 ──
    void cancel();
    bool is_cancelled() const;

private:
    PlatformInfo    platform_;
    RecommendedConfig cfg_;
    bool            initialized_ = false;
    ProgressCallback progress_cb_;

    // 子模块
    std::unique_ptr<quant::WeightStreamer>        weight_streamer_;
    std::unique_ptr<fusion::GraphFusionEngine>   fusion_engine_;
    std::unique_ptr<pipeline::DoubleBufferedPipeline> pipeline_;
    std::unique_ptr<mediatek::Dimensity8400Adapter> dimensity_adapter_;
    std::unique_ptr<render::VulkanZeroCopyPipeline> vulkan_pipeline_;

    // 状态
    mutable std::mutex  mu_;
    std::atomic<bool>   cancelled_{false};
    PerfReport          perf_report_;

    // 内部
    void  detect_and_init_backends();
    void  log_capabilities() const;
};

// ─── C API（供 JNI 调用）──────────────────────
extern "C" {
    void* locai_engine_v35_create();
    void  locai_engine_v35_destroy(void* engine);
    bool  locai_engine_v35_init(void* engine);
    void  locai_engine_v35_apply_recommended(void* engine);
    bool  locai_engine_v35_generate(void* engine,
                                      const char* prompt,
                                      const char* negative_prompt,
                                      int width, int height,
                                      int steps, float cfg_scale,
                                      int seed,
                                      const char* sampler,
                                      const char* model_path,
                                      uint8_t** out_rgb, int* out_w, int* out_h,
                                      float* out_total_ms);
    void  locai_engine_v35_free_rgb(uint8_t* ptr);
    char* locai_engine_v35_get_platform_info(void* engine);
    char* locai_engine_v35_get_perf_report(void* engine);
    void  locai_engine_v35_cancel(void* engine);
    void  locai_engine_v35_free_string(char* s);
}

} // namespace locai::v35
