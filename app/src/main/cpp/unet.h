#pragma once
/**
 * ============================================================================
 *  unet.h  —  Stable Diffusion UNet 推理引擎（v4.0 TaiShen）
 * ============================================================================
 *
 *  功能：
 *    - 完整的 SD1.5 / SD2.1 / SDXL UNet 前向推理
 *    - 支持 FP32 / FP16 / INT8 / INT4 四种精度
 *    - 内置 Cross-Attention（文本条件注入）
 *    - 支持 ControlNet 分支注入
 *    - 支持 LoRA 权重叠加（最多 5 路）
 *    - 天玑 8400 NPU 880 加速路径（dlopen NeuroPilot）
 *
 *  架构概览：
 *    Input(latents) → TimeEmbed → [DownBlocks ×4] → [MidBlock] → [UpBlocks ×4] → Output(噪声预测)
 *                              ↘ ControlNet 特征注入 ↗
 *                              ↘ Cross-Attention(文本条件) ↗
 */
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <cstdint>

// ── 量化精度 ──
enum class UnetPrecision : uint32_t {
    FP32  = 0,
    FP16  = 1,
    INT8  = 2,
    INT4  = 3,
    MIXED = 4,  // INT4 权重 + FP16 激活
};

// ── 模型变体 ──
enum class UnetVariant : uint32_t {
    SD15  = 0,  // 512², 4ch latent, 交叉注意力 768 维
    SD21  = 1,  // 768², 4ch latent, 交叉注意力 1024 维
    SDXL  = 2,  // 1024², 4ch latent, 交叉注意力 2048 维
    LCM   = 3,  // 兼容 SD1.5 架构，步数更少
    TURBO = 4,  // SDXL Turbo，1-4 步
};

// ── 残差块配置 ──
struct ResBlockConfig {
    int  in_channels;
    int  out_channels;
    int  num_groups;       // GroupNorm groups
    bool use_scale_shift;   // AdaGN（FiLM 条件化）
    bool use_attention;     // 该残差块后是否接自注意力
    int  attention_heads;
};

// ── 交叉注意力层配置 ──
struct CrossAttnConfig {
    int  query_dim;
    int  context_dim;       // 文本嵌入维度
    int  num_heads;
    int  head_dim;
    float dropout;
};

// ── 下采样/上采样块配置 ──
struct DownBlockConfig {
    std::vector<ResBlockConfig> res_blocks;
    bool  has_downsampler;  // 是否含 stride=2 卷积下采样
    int   downsampler_channels;
};

struct UpBlockConfig {
    std::vector<ResBlockConfig> res_blocks;
    bool  has_upsampler;    // 是否含转置卷积上采样
    int   upsampler_channels;
    bool  has_skip_connection; // 与 encoder 对应层 skip
};

// ── 完整 UNet 架构描述 ──
struct UnetArchitecture {
    UnetVariant variant = UnetVariant::SD15;

    int  in_channels    = 4;    // latent channels
    int  out_channels   = 4;
    int  model_channels = 320;  // 基础通道数
    int  time_emb_dim   = 1280; // 时间嵌入维度

    // 文本条件
    int  text_emb_dim   = 768;  // CLIP 768 / OpenCLIP 1024 / T5 2048
    bool use_cross_attn = true;

    // 块结构
    std::vector<DownBlockConfig> down_blocks;  // 通常 4 个
    std::vector<UpBlockConfig>   up_blocks;    // 通常 4 个
    ResBlockConfig               mid_block;     // 中间块

    // 注意力分辨率阈值（低于此分辨率才启用注意力）
    int  attention_resolutions[3] = {8, 16, 32};
    int  num_attention_levels     = 3;

    // 总参数量（用于预估内存）
    uint64_t total_params = 0;
};

// ── 量化参数 ──
struct QuantParams {
    UnetPrecision precision = UnetPrecision::FP16;
    // INT8 / INT4 每通道缩放因子
    std::vector<float> weight_scales;   // [out_channels]
    std::vector<float> activation_scales;// [in_channels]
    // INT4 零点（非对称量化用）
    std::vector<int8_t> weight_zeros;
    // 分组大小（INT4 group-wise）
    uint32_t int4_group_size = 128;
    // 是否对称量化
    bool symmetric = true;
};

// ── LoRA 适配器 ──
struct LoRAAdapter {
    std::string name;
    float       weight_scale = 1.0f;  // 0.0 ~ 2.0
    // LoRA 低秩矩阵（注入到 Q/K/V/O 投影）
    std::vector<float> lora_down_q;  // [rank × dim]
    std::vector<float> lora_up_q;    // [dim × rank]
    std::vector<float> lora_down_k;
    std::vector<float> lora_up_k;
    std::vector<float> lora_down_v;
    std::vector<float> lora_up_v;
    std::vector<float> lora_down_out;
    std::vector<float> lora_up_out;
    int  rank = 8;
    int  dim  = 0;  // 输入维度（自动从主权重推断）
    bool active = true;
};

// ── 推理配置 ──
struct UnetInferConfig {
    int    width        = 512;
    int    height       = 512;
    int    batch_size   = 1;
    int    num_steps    = 20;
    float  cfg_scale    = 7.5f;
    int    seed         = 42;
    // 性能
    int    num_threads  = 4;
    bool   use_npu      = false;   // 天玑 8400 NPU 路径
    bool   use_vulkan   = false;   // Vulkan Compute 路径
    bool   use_opencl   = false;   // OpenCL 路径
    // 内存
    bool   pin_to_l3    = true;   // 钉住热点权重到 L3
    bool   pin_to_slc   = true;   // 钉住到 SLC
    uint64_t memory_pool_mb = 1024;
    // 进度回调
    std::function<void(int step, int total, float eta)> on_progress;
};

// ═══════════════════════════════════════════
//  主类
// ═══════════════════════════════════════════
class Unet {
public:
    Unet();
    ~Unet();

    // ── 生命周期 ──
    bool load(const std::string& model_path, UnetVariant variant);
    bool load_from_memory(const void* data, size_t size, UnetVariant variant);
    void release();
    bool is_loaded() const { return loaded_; }

    // ── 量化 ──
    bool quantize_weights(UnetPrecision target_prec, uint32_t int4_group = 128);
    const QuantParams& get_quant_params() const { return quant_params_; }

    // ── LoRA 管理 ──
    bool attach_lora(const LoRAAdapter& lora);
    bool detach_lora(const std::string& name);
    void clear_all_lora();
    int  lora_count() const { return (int)loras_.size(); }

    // ── 核心推理 ──
    /**
     * 完整预测：输入 latents + timestep + 文本条件 → 输出噪声预测
     *
     * @param latents      [B, 4, H/8, W/8] 当前潜变量
     * @param timestep     当前时间步 (0 ~ 999)
     * @param text_emb     文本嵌入 [B, seq_len, text_emb_dim]
     * @param uncond_emb   无条件嵌入（CFG 用）
     * @param cfg_scale    CFG 引导强度
     * @param output       [B, 4, H/8, W/8] 噪声预测输出
     * @return             是否成功
     */
    bool predict(
        const float* latents,
        int          timestep,
        const float* text_emb,
        const float* uncond_emb,
        float        cfg_scale,
        float*       output
    );

    // ── 批量预测（CFG 合并，一次 forward 完成 cond + uncond）──
    bool predict_cfg_merged(
        const float* latents,
        int          timestep,
        const float* text_emb,
        float        cfg_scale,
        float*       output
    );

    // ── 天玑 8400 NPU 路径 ──
    bool enable_npu_acceleration(const std::string& dla_path);
    bool is_npu_active() const { return npu_active_; }

    // ── 内存统计 ──
    uint64_t get_weight_memory_bytes() const;
    uint64_t get_activation_memory_bytes(int width, int height) const;
    uint64_t get_total_memory_bytes(int width, int height) const;

    // ── 架构信息 ──
    const UnetArchitecture& get_architecture() const { return arch_; }
    UnetVariant get_variant() const { return arch_.variant; }

private:
    // ── 权重张量存储 ──
    struct Tensor {
        std::vector<float>  fp32_data;
        std::vector<int8_t> int8_data;   // INT8 量化后
        std::vector<uint8_t> int4_data;  // INT4 打包后（2 weights/byte）
        std::vector<float>  scales;       // 量化缩放
        std::vector<int8_t> zeros;        // 非对称量化零点
        int  dims[4];  // [out, in, kH, kW]
        UnetPrecision storage_prec = UnetPrecision::FP32;
    };

    // ── 网络层权重 ──
    // 时间嵌入
    Tensor time_embed_1_;   // [time_emb_dim, 320]
    Tensor time_embed_2_;   // [time_emb_dim, time_emb_dim]
    // 输入卷积
    Tensor input_conv_;      // [model_ch, in_ch, 3, 3]
    // 下采样块
    std::vector<std::vector<Tensor>> down_res_weights_;  // [block][layer][weight_type]
    std::vector<std::vector<Tensor>> down_attn_weights_;
    std::vector<Tensor> down_downsample_weights_;
    // 中间块
    Tensor mid_res1_weights_;
    Tensor mid_attn_weights_;
    Tensor mid_res2_weights_;
    // 上采样块
    std::vector<std::vector<Tensor>> up_res_weights_;
    std::vector<std::vector<Tensor>> up_attn_weights_;
    std::vector<Tensor> up_upsample_weights_;
    // 输出卷积
    Tensor output_conv_;     // [out_ch, model_ch, 3, 3]

    // ── 文本交叉注意力投影 ──
    Tensor text_proj_q_;     // [dim, text_emb_dim]
    Tensor text_proj_k_;
    Tensor text_proj_v_;
    Tensor text_proj_out_;

    // ── 状态 ──
    bool           loaded_     = false;
    bool           npu_active_ = false;
    UnetArchitecture arch_;
    QuantParams    quant_params_;
    std::vector<LoRAAdapter> loras_;

    // ── 工作缓冲区（避免每步重新分配）──
    struct WorkBuffers {
        std::vector<float> time_emb;       // [time_emb_dim]
        std::vector<float> hidden_states;   // 当前特征图
        std::vector<float> skip_features[4]; // 4 级 skip
        std::vector<float> attn_output;     // 注意力输出
        std::vector<float> cfg_merged;      // CFG 合并后的 latent
        // INT8 量化推理的中间缓冲区
        std::vector<int8_t> int8_hidden;
        std::vector<float>   fp32_hidden;
    };
    std::unique_ptr<WorkBuffers> work_;

    // ── NPU 句柄（dlopen 获取）──
    void* npu_handle_ = nullptr;
    void* npu_session_ = nullptr;

    // ── 内部方法 ──
    bool load_safetensors(const std::string& path);
    bool load_onnx(const std::string& path);
    bool load_gguf(const std::string& path);
    void build_architecture(UnetVariant variant);
    void allocate_work_buffers(int width, int height, int batch);

    // 核心算子
    void conv2d(const Tensor& w, const float* input, float* output,
                int batch, int in_ch, int out_ch, int h, int w,
                int ksize, int stride, int padding);
    void conv2d_int8(const Tensor& w, const float* input, float* output,
                     int batch, int in_ch, int out_ch, int h, int w,
                     int ksize, int stride, int padding);
    void group_norm(const float* input, float* output,
                    int batch, int channels, int spatial, int groups,
                    const float* gamma, const float* beta);
    void silu_activation(float* data, size_t count);
    void gelu_activation(float* data, size_t count);
    void cross_attention(const Tensor& q_proj, const Tensor& k_proj,
                        const Tensor& v_proj, const Tensor& out_proj,
                        const float* hidden, const float* text_emb,
                        float* output, int batch, int seq_len,
                        int hidden_dim, int text_seq_len, int num_heads);
    void self_attention(const float* input, float* output,
                       int batch, int seq_len, int dim, int num_heads);
    void upsample_nearest(float* output, const float* input,
                         int batch, int ch, int h_in, int w_in,
                         int scale_factor);
    void upsample_conv(float* output, const float* input,
                      const Tensor& w, int batch, int ch, int h, int w);
    void downsample_conv(float* output, const float* input,
                        const Tensor& w, int batch, int ch, int h, int w);

    // 时间步嵌入
    void timestep_embedding(float* out, int timestep, int dim);
    void sinusoidal_embedding(float* out, int timestep, int dim);

    // LoRA 注入
    void apply_lora_to_tensor(Tensor& target, const LoRAAdapter& lora,
                             const std::string& layer_name);

    // NPU 推理
    bool run_npu_inference(const float* latents, int timestep,
                          const float* text_emb, float* output);
};
