#pragma once
/**
 * ============================================================================
 *  vae_decoder.h  —  VAE 解码器（v4.0 TaiShen）
 * ============================================================================
 *
 *  功能：
 *    - 将 4ch 潜变量 (latent) 解码为 3ch RGB 图像
 *    - 支持 SD1.5 / SD2.1 / SDXL 三种 VAE
 *    - FP32 / FP16 / INT8 三精度
 *    - 天玑 8400 NPU 880 加速路径
 *    - 分块推理（大图不爆显存）
 *
 *  架构：
 *    latent[1,4,64,64] → Conv2d → [ResBlock×4 + UpSample×4] → [Conv2d→RGB]
 *                        → [GroupNorm + SiLU + Conv2d] × N
 *                        → 最终 Conv2d → [1,3,512,512]
 */
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

// ── VAE 变体 ──
enum class VaeVariant : uint32_t {
    SD15 = 0,  // 输入 64² → 输出 512²
    SD21 = 1,  // 输入 96² → 输出 768²
    SDXL = 2,  // 输入 128² → 输出 1024²
};

// ── VAE 精度 ──
enum class VaePrecision : uint32_t {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
};

// ── 分块策略 ──
struct TileStrategy {
    bool  enable_tiling   = true;   // 大图分块
    int   tile_size       = 256;    // 每块像素
    int   tile_overlap    = 16;     // 重叠像素（羽化）
    bool  use_vulkan      = false;   // Vulkan Compute 分块
    bool  use_npu         = false;   // NPU 分块
};

// ── VAE 架构描述 ──
struct VaeArchitecture {
    VaeVariant variant = VaeVariant::SD15;
    int  latent_ch     = 4;
    int  out_ch        = 3;
    int  base_ch       = 128;   // 基础通道
    int  ch_mult[4]    = {1, 2, 4, 4}; // 通道倍增
    int  num_res_blocks = 2;    // 每级残差块数
    int  latent_size   = 64;    // 输入空间尺寸
    int  output_size   = 512;   // 输出空间尺寸
    int  num_groups    = 32;    // GroupNorm groups
    float scaling_factor = 0.18215f; // SD 标准缩放因子
    uint64_t total_params = 0;
};

// ── 权重张量 ──
struct VaeTensor {
    std::vector<float>  fp32;
    std::vector<int8_t> int8;
    std::vector<float>  scales;
    int  dims[4];  // [out, in, kH, kW]
    VaePrecision storage = VaePrecision::FP32;
};

// ═══════════════════════════════════════════
//  主类
// ═══════════════════════════════════════════
class VaeDecoder {
public:
    VaeDecoder();
    ~VaeDecoder();

    // ── 生命周期 ──
    bool load(const std::string& model_path, VaeVariant variant);
    bool load_from_memory(const void* data, size_t size, VaeVariant variant);
    void release();
    bool is_loaded() const { return loaded_; }

    // ── 量化 ──
    bool quantize_to_int8();
    bool quantize_to_fp16();

    // ── 核心解码 ──
    /**
     * 完整解码：latent → RGB 图像
     *
     * @param latents  [B, 4, H/8, W/8] 潜变量（已乘 scaling_factor）
     * @param output   [B, 3, H, W] 输出 RGB（值域 [0,1]）
     * @return         是否成功
     */
    bool decode(const float* latents, float* output,
                int batch, int latent_h, int latent_w);

    // ── 分块解码（大图专用）──
    bool decode_tiled(const float* latents, float* output,
                      int batch, int latent_h, int latent_w,
                      const TileStrategy& tile);

    // ── 天玑 8400 NPU 路径 ──
    bool enable_npu(const std::string& dla_path);
    bool is_npu_active() const { return npu_active_; }

    // ── 内存 ──
    uint64_t get_weight_memory() const;
    uint64_t get_activation_memory(int out_h, int out_w) const;
    uint64_t get_total_memory(int out_h, int out_w) const;

    // ── 信息 ──
    const VaeArchitecture& get_arch() const { return arch_; }
    VaeVariant get_variant() const { return arch_.variant; }

private:
    // ── 权重 ──
    VaeTensor conv_in_;          // 输入卷积 [base_ch, latent_ch, 3, 3]
    std::vector<std::vector<VaeTensor>> res_blocks_up_;  // [level][block][weight]
    std::vector<std::vector<VaeTensor>> res_blocks_norm_; // GroupNorm 参数
    std::vector<VaeTensor> upconv_weights_;               // 转置卷积上采样
    VaeTensor conv_mid_;         // 中间卷积
    VaeTensor conv_out_;         // 输出卷积 [out_ch, base_ch, 3, 3]
    // GroupNorm 参数
    std::vector<std::vector<float>> norm_gammas_;
    std::vector<std::vector<float>> norm_betas_;

    // ── 状态 ──
    bool            loaded_     = false;
    bool            npu_active_ = false;
    VaeArchitecture arch_;
    void*           npu_handle_ = nullptr;

    // ── 工作缓冲区 ──
    struct WorkBuf {
        std::vector<float> hidden;     // 当前特征图
        std::vector<float> upsampled;  // 上采样临时
        std::vector<float> normalized;  // GroupNorm 输出
        std::vector<float> activated;  // SiLU 输出
    };
    std::unique_ptr<WorkBuf> work_;

    // ── 内部方法 ──
    void build_arch(VaeVariant variant);
    bool load_safetensors(const std::string& path);
    void allocate_work(int latent_h, int latent_w, int batch);

    // 算子
    void conv2d(const VaeTensor& w, const float* in, float* out,
                int batch, int in_ch, int out_ch, int h, int w,
                int ksize, int stride, int pad);
    void conv_transpose2d(const VaeTensor& w, const float* in, float* out,
                          int batch, int in_ch, int out_ch, int h_in, int w_in,
                          int scale_factor);
    void group_norm(const float* in, float* out, int batch, int ch,
                    int spatial, int groups, const float* gamma, const float* beta);
    void silu(float* data, size_t count);
    void upsample_nearest(float* out, const float* in,
                          int batch, int ch, int h, int w, int scale);

    // 残差块 forward
    void res_block_forward(const std::vector<VaeTensor>& weights,
                          const float* in, float* out,
                          int batch, int ch, int h, int w, int block_idx);

    // NPU 推理
    bool run_npu_decode(const float* latents, float* output,
                        int batch, int latent_h, int latent_w);
};
