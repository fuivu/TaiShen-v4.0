#pragma once
/**
 * ============================================================================
 *  sd_pipeline.h  —  Stable Diffusion 完整推理管线（v4.0 TaiShen）
 * ============================================================================
 *
 *  功能：
 *    - 文本编码（CLIP / OpenCLIP / T5）
 *    - 潜变量初始化（随机噪声 + 种子）
 *    - 去噪循环（集成所有 Scheduler）
 *    - VAE 解码
 *    - CFG（Classifier-Free Guidance）
 *    - 天玑 8400 NPU 880 全流程加速
 *    - 进度回调 + 可中断
 *
 *  使用流程：
 *    SDPipeline pipe(pool);
 *    pipe.load("models/sd15");                    // 加载模型
 *    pipe.set_scheduler(SchedulerType::DPM_PP_2M_KARRAS);
 *    pipe.set_resolution(512, 512);
 *    pipe.set_steps(20);
 *    pipe.set_cfg_scale(7.5f);
 *
 *    // 生成
 *    auto image = pipe.generate(
 *        "a cat sitting on a windowsill, sunset, cinematic lighting",
 *        "ugly, blurry, deformed",
 *        42,  // seed
 *        [](int step, int total, float eta) { ... }  // progress
 *    );
 */
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <atomic>
#include "memory_pool.h"
#include "unet.h"
#include "vae_decoder.h"

// ── 调度器类型（与 engine-core 一致）──
enum class SchedulerType : uint32_t {
    EULER       = 0,
    EULER_A     = 1,
    DPM_PP_2M   = 2,
    DPM_PP_2M_KARRAS = 3,
    DPM_SDE     = 4,
    UNIPC       = 5,
    LCM         = 6,
    TURBO       = 7,
    LIGHTNING   = 8,
    HEUN        = 9,
    LMS         = 10,
    DPM_SOLVER  = 11,
    DDPM        = 12,
    RESTART     = 13,
    CUSTOM      = 14,
};

// ── 模型类型 ──
enum class SDModelType : uint32_t {
    SD15   = 0,
    SD21   = 1,
    SDXL   = 2,
    LCM    = 3,
    TURBO  = 4,
    PIXART = 5,
    KOLORS = 6,
};

// ── 文本编码器类型 ──
enum class TextEncoderType : uint32_t {
    CLIP_L14   = 0,  // SD1.5 默认
    OPENCLIP_H  = 1,  // SD2.1
    T5_XL       = 2,  // SD3 / PixArt / Kolors
    DUAL_CLIP    = 3,  // SDXL（两个 CLIP）
};

// ── 生成配置 ──
struct GenerateConfig {
    // 基本
    int    width       = 512;
    int    height      = 512;
    int    steps       = 20;
    float  cfg_scale   = 7.5f;
    long   seed        = 42;
    int    batch_size  = 1;

    // 模型
    SDModelType  model_type   = SDModelType::SD15;
    SchedulerType scheduler  = SchedulerType::DPM_PP_2M_KARRAS;
    TextEncoderType text_enc = TextEncoderType::CLIP_L14;

    // 精度
    UnetPrecision unet_prec   = UnetPrecision::FP16;
    VaePrecision  vae_prec    = VaePrecision::FP16;
    bool          use_npu     = false;  // 天玑 8400 NPU
    bool          use_vulkan  = false;

    // 高级
    int    clip_skip    = 1;     // 1 = 最后一层，2 = 倒数第二层
    float  eta_noise    = 0.0f;  // DDIM/PLMS 用
    bool   do_cfg       = true;  // 关闭 = 无条件生成
    float  init_strength = 1.0f; // img2img 用（1.0 = 全去噪）

    // 性能
    int    num_threads  = 4;
    bool   pin_memory   = true;   // 钉住权重到 L3/SLC
    uint64_t memory_pool_mb = 1536;

    // 天玑 8400 专属
    bool   use_int4_quant  = false;  // NPU 880 INT4 混合精度
    bool   use_kv_cache    = true;   // 文本编码 KV Cache
    bool   use_double_buf  = true;   // 双缓冲 + DMA
    int    prefetch_dist   = 2;     // 预取距离
};

// ── 生成结果 ──
struct GenerateResult {
    bool        success       = false;
    std::string error_msg;
    // 图像数据
    std::vector<float>  image_fp32;   // [B, 3, H, W], 值域 [0, 1]
    std::vector<uint8_t> image_rgb8;  // [B, 3, H, W], 值域 [0, 255]
    // 元数据
    int         width    = 0;
    int         height   = 0;
    int         steps_done = 0;
    float       elapsed_sec = 0.0f;
    float       tokens_per_sec = 0.0f;
    // 性能统计
    uint64_t    npu_cycles    = 0;
    uint64_t    gpu_cycles    = 0;
    float       avg_power_mw  = 0.0f;
    float       peak_temp_c   = 0.0f;
    uint32_t    throttling_lvl = 0;
};

// ── 进度回调 ──
using ProgressCallback = std::function<void(
    int   current_step,
    int   total_steps,
    float progress_0to1,
    float eta_seconds,
    const std::string& stage  // "encoding" / "denoising" / "decoding"
)>;

// ── 中断令牌 ──
struct CancellationToken {
    std::atomic<bool> cancelled{false};
    void cancel() { cancelled.store(true); }
    bool is_cancelled() const { return cancelled.load(); }
};

// ═══════════════════════════════════════════
//  主类
// ═══════════════════════════════════════════
class SDPipeline {
public:
    explicit SDPipeline(MemoryPool* pool);
    ~SDPipeline();

    // ── 禁止拷贝 ──
    SDPipeline(const SDPipeline&) = delete;
    SDPipeline& operator=(const SDPipeline&) = delete;

    // ═══════════════════════════════════
    //  生命周期
    // ═══════════════════════════════════
    /**
     * 加载完整模型包（目录中包含 unet / vae / text_encoder / tokenizer）
     */
    bool load(const std::string& model_dir, SDModelType type);
    bool load_from_paths(
        const std::string& unet_path,
        const std::string& vae_path,
        const std::string& text_enc_path,
        SDModelType type
    );
    void release();
    bool is_loaded() const { return loaded_; }

    // ═══════════════════════════════════
    //  配置
    // ═══════════════════════════════════
    void set_config(const GenerateConfig& cfg) { config_ = cfg; }
    const GenerateConfig& get_config() const { return config_; }
    void set_scheduler(SchedulerType t) { config_.scheduler = t; }
    void set_resolution(int w, int h) { config_.width = w; config_.height = h; }
    void set_steps(int n) { config_.steps = n; }
    void set_cfg_scale(float s) { config_.cfg_scale = s; }
    void set_seed(long s) { config_.seed = s; }

    // ═══════════════════════════════════
    //  文本编码
    // ═══════════════════════════════════
    /**
     * 编码提示词 → 嵌入向量
     * 支持 CLIP / OpenCLIP / T5 三种编码器
     * 自动处理 tokenization + 嵌入 + KV Cache（如启用）
     */
    bool encode_prompt(
        const std::string& prompt,
        std::vector<float>& text_emb,
        std::vector<float>& uncond_emb
    );

    // ═══════════════════════════════════
    //  核心生成
    // ═══════════════════════════════════
    /**
     * 完整生成流程：
     *   prompt → tokenize → encode → init noise → denoise loop → vae decode → RGB
     *
     * @param prompt          正向提示词
     * @param negative_prompt 负向提示词（空 = 使用默认）
     * @param seed            随机种子（0 = 随机）
     * @param progress        进度回调（可为 nullptr）
     * @param cancel          取消令牌（可为 nullptr）
     * @return                GenerateResult
     */
    GenerateResult generate(
        const std::string& prompt,
        const std::string& negative_prompt,
        long seed,
        ProgressCallback progress = nullptr,
        std::shared_ptr<CancellationToken> cancel = nullptr
    );

    // ═══════════════════════════════════
    //  img2img（基于图像的再生成）
    // ═══════════════════════════════════
    GenerateResult img2img(
        const std::vector<uint8_t>& input_image_rgb,  // [3, H, W]
        int in_w, int in_h,
        const std::string& prompt,
        const std::string& negative_prompt,
        float strength,  // 0.0 ~ 1.0
        long seed,
        ProgressCallback progress = nullptr,
        std::shared_ptr<CancellationToken> cancel = nullptr
    );

    // ═══════════════════════════════════
    //  LoRA 管理（委托给 Unet）
    // ═══════════════════════════════════
    bool attach_lora(const LoRAAdapter& lora);
    bool detach_lora(const std::string& name);
    void clear_all_lora();

    // ═══════════════════════════════════
    //  天玑 8400 NPU 全流程
    // ═══════════════════════════════════
    bool enable_npu_pipeline(const std::string& unet_dla,
                            const std::string& vae_dla,
                            const std::string& text_dla);
    bool is_npu_pipeline_active() const { return npu_pipeline_active_; }

    // ═══════════════════════════════════
    //  性能统计
    // ═══════════════════════════════════
    struct PerfStats {
        float  text_encode_ms   = 0;
        float  denoise_total_ms = 0;
        float  denoise_per_step_ms = 0;
        float  vae_decode_ms   = 0;
        float  total_ms        = 0;
        uint64_t peak_memory_bytes = 0;
        uint64_t npu_active_cycles = 0;
        float  avg_power_mw     = 0;
        float  temperature_c    = 0;
    };
    PerfStats get_last_perf() const { return last_perf_; }

    // ═══════════════════════════════════
    //  内存估算
    // ═══════════════════════════════════
    uint64_t estimate_total_memory(int width, int height) const;
    uint64_t estimate_unet_memory(int width, int height) const;
    uint64_t estimate_vae_memory(int width, int height) const;

private:
    // ── 子模块 ──
    MemoryPool*      mem_pool_;
    std::unique_ptr<Unet>        unet_;
    std::unique_ptr<VaeDecoder>  vae_;
    // 文本编码器（内部实现，可切换 CLIP/T5）
    struct TextEncoderImpl;
    std::unique_ptr<TextEncoderImpl> text_enc_impl_;

    // ── 状态 ──
    bool           loaded_ = false;
    bool           npu_pipeline_active_ = false;
    SDModelType    model_type_ = SDModelType::SD15;
    GenerateConfig config_;
    PerfStats      last_perf_;

    // ── 调度器状态（与 engine/scheduler 配合）──
    struct SchedulerState;
    std::unique_ptr<SchedulerState> sched_state_;

    // ── NPU 句柄 ──
    void* npu_pipe_handle_ = nullptr;

    // ── 内部方法 ──
    bool init_scheduler(SchedulerType type, int steps);
    void step_scheduler(float* latents, float* pred_noise,
                       int step, int total_steps, float* output);
    void encode_tokens(const std::vector<int>& tokens,
                       std::vector<float>& emb);
    void encode_tokens_uncond(int seq_len, std::vector<float>& emb);
    void init_latents(std::vector<float>& latents, int w, int h,
                      long seed, float init_strength);
    void latent_to_rgb(const std::vector<float>& latents,
                       std::vector<float>& rgb, int w, int h);
    void rgb_float_to_uint8(const std::vector<float>& rgb_f,
                            std::vector<uint8_t>& rgb_u8);

    // 量化推理路径
    bool run_denoise_quantized(float* latents,
                               const std::vector<float>& cond_emb,
                               const std::vector<float>& uncond_emb,
                               int steps, ProgressCallback progress,
                               std::shared_ptr<CancellationToken> cancel);
    // NPU 推理路径
    bool run_denoise_npu(float* latents,
                         const std::vector<float>& cond_emb,
                         const std::vector<float>& uncond_emb,
                         int steps, ProgressCallback progress,
                         std::shared_ptr<CancellationToken> cancel);

    // 工具
    void report_progress(ProgressCallback cb, int step, int total,
                         float eta, const std::string& stage);
    void update_perf_stats(float encode_ms, float denoise_ms,
                           float vae_ms, uint64_t peak_mem);
};
