#pragma once
/**
 * 图融合优化器 (v4.0 TaiShen)
 * - Conv+BN+SiLU 三件套融合 (UNet 每层省 2 次内存读写)
 * - QKV 投影融合 (Attention 1.8×)
 * - Softmax+MatMul 融合 (Flash Attention 风格在线 softmax)
 * - LUT-Diff 查表融合 (IEEE TMC 2025: 9.1× 加速, 70.9% 内存节省)
 */
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

namespace locai::v35::opt {

// ═══════════════════════════════════════════════════════════
//  融合 Pass 接口
// ═══════════════════════════════════════════════════════════

struct TensorShape { int n, c, h, w; };

// 融合后的 Conv+BN+SiLU kernel 参数
struct FusedConvBNSiLU {
    std::vector<float> weights;  // [out_c][in_c][kh][kw] (已融合 BN scale/bias)
    std::vector<float> bias;     // [out_c] (已融合 BN)
    int in_channels, out_channels, kernel_size;
    int stride, padding;
    // 原始 BN 参数 (融合后不再需要, 但保留用于调试)
    std::vector<float> bn_running_mean, bn_running_var, bn_gamma, bn_beta;
    float bn_eps = 1e-5f;
    // 验证: 融合前后数值误差应 < 1e-4
    float max_fusion_error = 0.f;
};

// QKV 融合: 3 个 [out, in] 矩阵 → 1 个 [3*out, in]
struct FusedQKV {
    std::vector<float> weights; // [3*head_dim*num_heads, in_dim]
    std::vector<float> bias;    // [3*head_dim*num_heads]
    int num_heads, head_dim, in_dim;
};

// Softmax+MatMul 融合 (Flash Attention 风格)
// 在线 softmax: 一次 pass 计算 exp(q@k^T) * v, 不需要存储 N×N 注意力矩阵
struct FusedAttnParams {
    int seq_len, num_heads, head_dim;
    float scale; // 1/sqrt(head_dim)
    bool use_online_softmax = true; // Flash Attn 风格
    int block_size = 64;            // tiling block size
};

// LUT-Diff 查表融合
// 将 GELU/SiLU/Exp 等非线性操作替换为 O(1) 查表
struct LUTParams {
    std::vector<float> table;   // 预计算表
    float input_min, input_max;
    int   table_size;
    std::string op_name; // "gelu" / "silu" / "exp" / "layernorm"
};

// ═══════════════════════════════════════════════════════════
//  融合器
// ═══════════════════════════════════════════════════════════

class GraphFusion {
public:
    GraphFusion();
    ~GraphFusion();

    // Pass 1: Conv + BatchNorm + SiLU → 单一 Fused Kernel
    // 数学: SiLU(BN(Conv(x))) = x * sigmoid(γ·(Conv(x)-μ)/√(σ²+ε)+β)
    // 融合后: SiLU(Conv'(x)) 其中 Conv' 权重已吸收 BN 参数
    FusedConvBNSiLU fuse_conv_bn_silu(
        const float* conv_w, const float* conv_b,
        int in_ch, int out_ch, int kw, int kh, int stride, int pad,
        const float* bn_mean, const float* bn_var,
        const float* bn_gamma, const float* bn_beta, float bn_eps);

    // Pass 2: Q/K/V 三个投影 → 单次 MatMul [N, 3*D]
    FusedQKV fuse_qkv_proj(
        const float* w_q, const float* w_k, const float* w_v,
        const float* b_q, const float* b_k, const float* b_v,
        int in_dim, int num_heads, int head_dim);

    // Pass 3: Softmax + MatMul → Flash Attention 在线版本
    // 不物化 N×N 注意力矩阵, 省内存 + 省带宽
    void fused_attention(
        const float* q, const float* k, const float* v,
        float* output, const FusedAttnParams& params);

    // Pass 4: LUT-Diff 查表融合 (IEEE TMC 2025)
    // 非线性操作 O(1) 查表: y = table[(x - min) / step]
    LUTParams create_lut(const std::string& op, int table_size = 1024);
    void apply_lut(const float* input, float* output, int n, const LUTParams& lut);

    // 一键全部应用
    struct FusionReport {
        int  conv_bn_silu_fused = 0;
        int  qkv_fused = 0;
        int  attn_fused = 0;
        int  lut_applied = 0;
        float total_speedup = 1.0f;
        float memory_saved_mb = 0.f;
    };
    FusionReport apply_all_passes();

    // 天玑 NPU 适配: 生成 NPU 原生子图描述
    std::string emit_mediatek_subgraph() const;
    // 骁龙 NPU 适配: 生成 QNN 图描述
    std::string emit_qualcomm_qnn_graph() const;

private:
    FusionReport report_;
    std::vector<FusedConvBNSiLU> fused_conv_cache_;
    std::vector<FusedQKV>        fused_qkv_cache_;
    std::unordered_map<std::string, LUTParams> lut_cache_;
};

} // namespace locai::v35::opt
