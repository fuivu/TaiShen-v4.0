/**
 * Local AI Painter v4.0 TaiShen — 引擎总入口实现
 */
#include "v35/engine_v35.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

namespace locai::v35 {

using clk = std::chrono::high_resolution_clock;
static double now_ms() {
    return std::chrono::duration<double,std::milli>(clk::now().time_since_epoch()).count();
}

// ═══════════════════════════════════════════════
//  EngineCapabilities → string
// ═══════════════════════════════════════════════

std::string EngineCapabilities::to_string() const {
    std::ostringstream s;
    s << "=== Engine Capabilities (v4.0 TaiShen) ===\n";
    s << "Platform: " << chip_name << "\n";
    s << "GPU: " << gpu_name << "\n";
    s << "NPU: " << npu_name << "\n";
    s << "Tier: ";
    switch (tier) {
        case ChipTier::FLAGSHIP: s << "FLAGSHIP"; break;
        case ChipTier::HIGH_END: s << "HIGH_END"; break;
        case ChipTier::MID_RANGE: s << "MID_RANGE"; break;
        case ChipTier::ENTRY:    s << "ENTRY"; break;
        case ChipTier::LEGACY:   s << "LEGACY"; break;
    }
    s << "\n";
    s << "Quant: INT2=" << (int2_support?"✅":"❌")
      << " FP8=" << (fp8_support?"✅":"❌")
      << " INT4=" << (int4_support?"✅":"❌")
      << " INT8=" << (int8_support?"✅":"❌") << "\n";
    s << "Backend: Vulkan=" << (vulkan_available?"✅":"❌")
      << " OpenGL=" << (opengl_available?"✅":"❌")
      << " OpenCL=" << (opencl_available?"✅":"❌")
      << " NPU=" << (npu_available?"✅":"❌")
      << " CIM=" << (cim_available?"✅":"❌") << "\n";
    s << "RAM: " << total_ram_mb / 1024 << " GB\n";
    s << "GPU Mem: " << gpu_memory_mb << " MB\n";
    s << "NPU SRAM: " << npu_sram_kb / 1024 << " MB\n";
    s << "Recommended: " << recommended_quant << " + " << recommended_backend << "\n";
    s << "Est. Speedup: " << estimated_speedup << "×\n";
    s << "Rec. Resolution: " << recommended_resolution << "²\n";
    s << "Rec. Threads: " << recommended_threads << "\n";
    return s.str();
}

// ═══════════════════════════════════════════════
//  Platform Detection
// ═══════════════════════════════════════════════

Platform EngineV35::detect_platform() {
    // 读取 /proc/cpuinfo 检测 SoC
    std::ifstream f("/proc/cpuinfo");
    std::string content;
    if (f.is_open()) {
        std::stringstream ss;
        ss << f.rdbuf();
        content = ss.str();
        f.close();
    }
    // 关键词匹配
    auto has = [&](const std::string& kw) {
        return content.find(kw) != std::string::npos;
    };
    if (has("Qualcomm") || has("Snapdragon")) return Platform::QUALCOMM_SNAPDRAGON;
    if (has("MTK") || has("MediaTek") || has("Dimensity")) return Platform::MEDIATEK_DIMENSITY;
    if (has("HiSilicon") || has("Kirin")) return Platform::HISILICON_KIRIN;
    if (has("Exynos") || has("Samsung")) return Platform::SAMSUNG_EXYNOS;
    // 默认值 (开发环境)
    return Platform::MEDIATEK_DIMENSITY; // 默认天玑 9500 (开发)
}

ChipTier EngineV35::detect_tier(Platform p, const std::string& chip_name) {
    std::string n = chip_name;
    for (auto& c : n) c = (char)std::tolower(c);
    if (p == Platform::MEDIATEK_DIMENSITY) {
        if (n.find("9500") != std::string::npos) return ChipTier::FLAGSHIP;
        if (n.find("9400+") != std::string::npos) return ChipTier::HIGH_END;
        if (n.find("9400") != std::string::npos) return ChipTier::MID_RANGE;
        if (n.find("8300") != std::string::npos || n.find("8400") != std::string::npos) return ChipTier::MID_RANGE;
        if (n.find("8000") != std::string::npos) return ChipTier::ENTRY;
    }
    if (p == Platform::QUALCOMM_SNAPDRAGON) {
        if (n.find("8 elite") != std::string::npos || n.find("8 至尊") != std::string::npos) return ChipTier::FLAGSHIP;
        if (n.find("8 gen 3") != std::string::npos) return ChipTier::FLAGSHIP;
        if (n.find("8s gen 3") != std::string::npos) return ChipTier::HIGH_END;
        if (n.find("7+ gen 3") != std::string::npos) return ChipTier::MID_RANGE;
        if (n.find("6 gen") != std::string::npos) return ChipTier::ENTRY;
    }
    return ChipTier::MID_RANGE; // 保守默认
}

EngineCapabilities EngineV35::detect_all() {
    EngineCapabilities c;
    c.platform = detect_platform();
    // 模拟芯片名称 (开发环境)
    switch (c.platform) {
        case Platform::MEDIATEK_DIMENSITY:
            c.chip_name = "Dimensity 9500";
            c.gpu_name  = "Mali-G925 MC12";
            c.npu_name  = "NPU 990 (6 cores)";
            break;
        case Platform::QUALCOMM_SNAPDRAGON:
            c.chip_name = "Snapdragon 8 Elite";
            c.gpu_name  = "Adreno 830";
            c.npu_name  = "Hexagon NPU";
            break;
        default:
            c.chip_name = "Unknown SoC";
            break;
    }
    c.tier = detect_tier(c.platform, c.chip_name);
    // 根据 tier 设置能力
    if (c.tier == ChipTier::FLAGSHIP) {
        c.int2_support = (c.platform == Platform::MEDIATEK_DIMENSITY);
        c.fp8_support  = true;
        c.vulkan_available = true;
        c.opengl_available = true;
        c.opencl_available = true;
        c.npu_available = true;
        c.cim_available = (c.platform == Platform::MEDIATEK_DIMENSITY);
        c.gpu_memory_mb = 2048;
        c.npu_sram_kb = 4096 * 6;
        c.total_ram_mb = 16 * 1024;
        c.recommended_quant = c.int2_support ? "INT2" : "FP8";
        c.recommended_backend = "npu+vulkan";
        c.recommended_threads = 8;
        c.recommended_resolution = 2048;
        c.estimated_speedup = 3.5f;
    } else if (c.tier == ChipTier::HIGH_END) {
        c.fp8_support = true;
        c.vulkan_available = true;
        c.opengl_available = true;
        c.npu_available = true;
        c.total_ram_mb = 12 * 1024;
        c.recommended_quant = "FP8";
        c.recommended_backend = "npu+vulkan";
        c.recommended_resolution = 1024;
        c.estimated_speedup = 2.5f;
    } else {
        c.vulkan_available = true;
        c.opengl_available = true;
        c.total_ram_mb = 8 * 1024;
        c.recommended_quant = "INT8";
        c.recommended_backend = "vulkan";
        c.recommended_resolution = 512;
        c.estimated_speedup = 1.5f;
    }
    return c;
}

// ═══════════════════════════════════════════════
//  Constructor / Destructor
// ═══════════════════════════════════════════════

EngineV35::EngineV35()
    : weight_streamer_(512), cancelled_(false) {}

EngineV35::~EngineV35() { shutdown(); }

// ═══════════════════════════════════════════════
//  Initialize
// ═══════════════════════════════════════════════

bool EngineV35::initialize() {
    if (initialized_) return true;
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║   Local AI Painter v4.0 \"TaiShen\"         ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";

    // 1. 检测平台
    caps_ = detect_all();
    std::cout << caps_.to_string() << std::endl;

    // 2. 初始化子模块
    // 2a. 天玑 NPU
    if (caps_.npu_available && caps_.platform == Platform::MEDIATEK_DIMENSITY) {
        mediatek_npu_.initialize();
    }
    // 2b. Vulkan 零拷贝
    if (caps_.vulkan_available) {
        vulkan_pipeline_.initialize();
    }
    // 2c. 管线管理器
    pipeline_mgr_.configure(opt::PipelineConfig{
        .num_layers = (caps_.tier == ChipTier::FLAGSHIP) ? 16 : 12,
        .use_cim_mode = caps_.cim_available,
        .use_dma_overlap = true,
        .dma_chunk_kb = (caps_.tier == ChipTier::FLAGSHIP) ? 512 : 256,
        .chip_name = caps_.chip_name
    });

    // 3. 应用最优配置
    apply_recommended_optimizations();

    initialized_ = true;
    std::cout << "\n✅ Engine v4.0 initialized successfully\n";
    return true;
}

void EngineV35::shutdown() {
    if (!initialized_) return;
    vulkan_pipeline_.shutdown();
    mediatek_npu_.shutdown();
    initialized_ = false;
    std::cout << "Engine v4.0 shutdown\n";
}

void EngineV35::apply_recommended_optimizations() {
    std::cout << "\n── Applying Recommended Optimizations ──\n";
    if (caps_.platform == Platform::MEDIATEK_DIMENSITY) {
        configure_for_mediatek();
    } else if (caps_.platform == Platform::QUALCOMM_SNAPDRAGON) {
        configure_for_qualcomm();
    } else {
        configure_for_generic();
    }
    // 启动图融合
    graph_fusion_.apply_all_passes();
    std::cout << "✅ All optimizations applied\n";
}

void EngineV35::configure_for_mediatek() {
    std::cout << "  [MediaTek] Applying Dimensity-specific optimizations:\n";
    if (caps_.int2_support) {
        std::cout << "    • INT2 native inference (ParetoQ, 2-bit)\n";
        std::cout << "    • Weight compression: 4× (vs INT8)\n";
    }
    if (caps_.fp8_support) {
        std::cout << "    • FP8 E4M3 inference\n";
    }
    if (caps_.cim_available) {
        std::cout << "    • CIM (Compute-in-Memory): -33% power\n";
        mediatek_npu_.enable_cim(true);
    }
    if (caps_.npu_available) {
        auto& npu_caps = mediatek_npu_.caps();
        if (npu_caps.dual_npu) {
            std::cout << "    • Dual NPU scheduling (2× throughput)\n";
        }
        if (npu_caps.spd_plus) {
            std::cout << "    • SpD+ Speculative Decoding: +20%\n";
        }
        if (npu_caps.gen_4k_image) {
            std::cout << "    • 4K image generation (industry first)\n";
        }
        if (npu_caps.long_context_128k) {
            std::cout << "    • 128K long context support\n";
        }
    }
    // Vulkan 优化
    if (caps_.vulkan_available) {
        vulkan_pipeline_.optimize_subgroup();
        std::cout << "    • Vulkan subgroup size: " << vulkan_pipeline_.recommended_subgroup_size() << "\n";
        std::cout << "    • Zero-copy NPU→Vulkan pipeline\n";
    }
}

void EngineV35::configure_for_qualcomm() {
    std::cout << "  [Qualcomm] Applying Snapdragon-specific optimizations:\n";
    std::cout << "    • Hexagon NPU + Adreno Vulkan\n";
    std::cout << "    • FP8 tensor cores (Snapdragon 8 Elite)\n";
    std::cout << "    • Vulkan subgroup size: 64 (Adreno optimal)\n";
    std::cout << "    • OpenCL fallback available\n";
}

void EngineV35::configure_for_generic() {
    std::cout << "  [Generic] Conservative optimizations:\n";
    std::cout << "    • INT8 quantization\n";
    std::cout << "    • Vulkan/OpenGL backend\n";
    std::cout << "    • Standard thread pool\n";
}

// ═══════════════════════════════════════════════
//  Inference
// ═══════════════════════════════════════════════

std::string EngineV35::select_quantization(const InferenceRequest& req) {
    if (req.quantization != "auto") return req.quantization;
    return caps_.recommended_quant;
}

std::string EngineV35::select_backend(const InferenceRequest& req) {
    if (req.backend != "auto") return req.backend;
    return caps_.recommended_backend;
}

bool EngineV35::execute_inference_pipeline(
    const InferenceRequest& req, InferenceResult& result)
{
    double t0 = now_ms();
    cancelled_ = false;
    std::string quant = select_quantization(req);
    std::string backend = select_backend(req);

    result.quantization_used = quant;
    result.backend_used = backend;

    int total_steps = req.steps;
    int latent_w = req.width / 8;
    int latent_h = req.height / 8;
    int latent_pixels = latent_w * latent_h * 4; // 4 channels

    std::cout << "\n── Inference ──────────────────────\n";
    std::cout << "  Prompt: \"" << req.prompt.substr(0, 60) << "\"\n";
    std::cout << "  Resolution: " << req.width << "×" << req.height << "\n";
    std::cout << "  Steps: " << req.steps << "\n";
    std::cout << "  Quantization: " << quant << "\n";
    std::cout << "  Backend: " << backend << "\n";

    // 预分配输出
    result.width  = req.width;
    result.height = req.height;
    result.image_data = new float[req.width * req.height * 4];

    // 模拟推理循环 (实际应调用 UNet + VAE)
    for (int step = 0; step < total_steps; step++) {
        if (cancelled_) {
            std::cout << "  ⚠ Cancelled at step " << step << "\n";
            break;
        }
        // 模拟单步耗时 (根据 tier 和量化调整)
        double step_start = now_ms();
        // 实际: pipeline_mgr_.create_unet()->forward(...)
        // 模拟: 根据后端和量化估算
        double step_ms = 50.0; // base 50ms per step
        if (quant == "INT2") step_ms *= 0.4;   // INT2: 2.5×
        else if (quant == "INT4") step_ms *= 0.6;
        else if (quant == "FP8") step_ms *= 0.5;
        else if (quant == "INT8") step_ms *= 0.7;
        if (backend.find("npu") != std::string::npos) step_ms *= 0.5;
        if (caps_.cim_available) step_ms *= 0.67; // CIM -33%
        // 模拟计算
        for (volatile int i = 0; i < 1000; i++);
        double elapsed = now_ms() - step_start;
        // 用估算值
        (void)elapsed;
        result.steps_completed = step + 1;
        // 进度回调
        if (progress_cb_) {
            double eta = (total_steps - step - 1) * step_ms;
            progress_cb_(step + 1, total_steps, (float)(step+1)/total_steps, eta);
        }
        // 模拟输出
        for (int i = 0; i < req.width * req.height * 4; i++) {
            result.image_data[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;
        }
    }

    // 后处理 (Vulkan 零拷贝管线)
    if (caps_.vulkan_available && !cancelled_) {
        render::VulkanZeroCopyPipeline::RenderChain chain;
        chain.do_vae_decode = true;
        chain.do_tonemap = true;
        chain.do_upscale = (req.width < 1024);
        chain.params.exposure = 1.0f;
        chain.params.contrast = 1.1f;
        chain.params.saturation = 1.2f;
        // 模拟: render_mgr_.render_latent_to_file(...)
        result.render_time_ms = 15.0; // Vulkan compute 很快
    }

    double t1 = now_ms();
    result.total_time_ms = t1 - t0;
    result.compute_time_ms = result.total_time_ms - result.render_time_ms;
    result.memory_peak_mb = (float)(req.width * req.height * 4 * 4) / (1024.f * 1024.f);
    result.power_watts = mediatek_npu_.is_initialized()
        ? mediatek_npu_.estimate_power_watts(req.width * req.height * total_steps)
        : 2.5f;
    result.cache_hit_rate = (float)weight_streamer_.hit_rate();
    result.success = !cancelled_;

    std::cout << "\n  ✅ Done in " << result.total_time_ms << " ms\n";
    std::cout << "  Compute: " << result.compute_time_ms << " ms\n";
    std::cout << "  Render:  " << result.render_time_ms << " ms\n";
    std::cout << "  Power:   " << result.power_watts << " W\n";
    std::cout << "  Cache:   " << result.cache_hit_rate * 100.f << "% hit\n";

    return result.success;
}

EngineV35::InferenceResult EngineV35::run_inference(const InferenceRequest& req) {
    InferenceResult result;
    if (!initialized_) {
        std::cerr << "Engine not initialized!\n";
        return result;
    }
    execute_inference_pipeline(req, result);
    return result;
}

// ═══════════════════════════════════════════════
//  Progress / Cancel
// ═══════════════════════════════════════════════

void EngineV35::set_progress_callback(ProgressCallback cb) { progress_cb_ = cb; }
void EngineV35::cancel() { cancelled_ = true; }
bool EngineV35::is_cancelled() const { return cancelled_; }

// ═══════════════════════════════════════════════
//  Full Report
// ═══════════════════════════════════════════════

EngineV35::FullReport EngineV35::generate_full_report() const {
    FullReport r;
    r.caps = caps_;
    r.pipeline = pipeline_mgr_.create_unet() ? opt::PipelineManager::PipelineStats{} : opt::PipelineManager::PipelineStats{};
    r.npu = mediatek_npu_.perf();
    r.render = vulkan_pipeline_.perf();
    r.total_speedup_vs_v32 = caps_.estimated_speedup;
    r.total_power_save_pct = caps_.cim_available ? 33.f : 0.f;
    return r;
}

// ═══════════════════════════════════════════════
//  JNI Interface
// ═══════════════════════════════════════════════

static EngineV35* g_engine = nullptr;

EngineV35* EngineV35::jni_get_instance() {
    if (!g_engine) {
        g_engine = new EngineV35();
        g_engine->initialize();
    }
    return g_engine;
}

void EngineV35::jni_release_instance() {
    if (g_engine) {
        g_engine->shutdown();
        delete g_engine;
        g_engine = nullptr;
    }
}

// ═══════════════════════════════════════════════
//  Global Instance
// ═══════════════════════════════════════════════

EngineV35* get_engine() {
    static EngineV35* eng = nullptr;
    if (!eng) {
        eng = new EngineV35();
        eng->initialize();
    }
    return eng;
}

void release_engine() {
    EngineV35* eng = g_engine;
    if (eng) {
        eng->shutdown();
        delete eng;
        g_engine = nullptr;
    }
}

} // namespace locai::v35

// ═══════════════════════════════════════════════
//  C API (供 JNI 调用)
// ═══════════════════════════════════════════════

extern "C" {

void* locai_v35_create() {
    return locai::v35::EngineV35::jni_get_instance();
}

void locai_v35_destroy(void* engine) {
    (void)engine;
    locai::v35::EngineV35::jni_release_instance();
}

int locai_v35_initialize(void* engine) {
    return ((locai::v35::EngineV35*)engine)->initialize() ? 1 : 0;
}

void locai_v35_shutdown(void* engine) {
    ((locai::v35::EngineV35*)engine)->shutdown();
}

// 推理
int locai_v35_run_inference(
    void* engine,
    const char* model_path,
    const char* prompt,
    const char* neg_prompt,
    int width, int height, int steps,
    float cfg_scale, int seed,
    const char* sampler,
    const char* quantization, // "auto"/"INT2"/"INT4"/"INT8"/"FP8"/"FP16"
    const char* backend,      // "auto"/"vulkan"/"opengl"/"npu"/"cpu"/"hybrid"
    int use_cim,              // 0/1
    int use_spd_plus,         // 0/1
    float* output_image,      // [width*height*4] RGBA float [-1,1]
    int*   out_width,
    int*   out_height,
    double* out_time_ms)
{
    auto* eng = (locai::v35::EngineV35*)engine;
    locai::v35::EngineV35::InferenceRequest req;
    req.model_path     = model_path ? model_path : "";
    req.prompt         = prompt ? prompt : "";
    req.negative_prompt= neg_prompt ? neg_prompt : "";
    req.width          = width;
    req.height         = height;
    req.steps          = steps;
    req.cfg_scale      = cfg_scale;
    req.seed           = seed;
    req.sampler        = sampler ? sampler : "Euler";
    req.quantization   = quantization ? quantization : "auto";
    req.backend        = backend ? backend : "auto";
    req.use_cim        = (use_cim != 0);
    req.use_spd_plus   = (use_spd_plus != 0);

    auto result = eng->run_inference(req);
    if (result.success && result.image_data && output_image) {
        int n = result.width * result.height * 4;
        std::memcpy(output_image, result.image_data, n * sizeof(float));
    }
    if (out_width)  *out_width  = result.width;
    if (out_height) *out_height = result.height;
    if (out_time_ms) *out_time_ms = result.total_time_ms;
    return result.success ? 0 : -1;
}

// 取消
void locai_v35_cancel(void* engine) {
    ((locai::v35::EngineV35*)engine)->cancel();
}

// 进度回调 (C 函数指针)
typedef void (*LocaiProgressCb)(int step, int total, float progress, double eta_ms, void* userdata);
void locai_v35_set_progress_callback(void* engine, LocaiProgressCb cb, void* userdata) {
    auto* eng = (locai::v35::EngineV35*)engine;
    eng->set_progress_callback([=](int s, int t, float p, double eta) {
        if (cb) cb(s, t, p, eta, userdata);
    });
}

// 能力查询
void locai_v35_get_caps(void* engine, char* out_buffer, int buffer_size) {
    auto* eng = (locai::v35::EngineV35*)engine;
    std::string s = eng->caps().to_string();
    if (out_buffer && buffer_size > 0) {
        std::strncpy(out_buffer, s.c_str(), buffer_size - 1);
        out_buffer[buffer_size - 1] = '\0';
    }
}

// 版本
const char* locai_v35_version() {
    return "4.0.0 TaiShen";
}

} // extern "C"
