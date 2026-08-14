#pragma once
/**
 * Graph Fusion Optimizer
 * Conv+BN+SiLU 融合 / QKV 融合 / Flash Attention / LUT-Diff
 */
#include <vector>
#include <string>
#include <functional>

namespace locai::fusion {

// ─── 融合后的算子类型 ──────────────────────────────────
enum class FusedOpType {
    CONV_BN_SILU,        // Conv + BatchNorm + SiLU → 单 kernel
    CONV_BN_RELU,        // Conv + BatchNorm + ReLU
    QKV_PROJECTION,      // Q/K/V 三个 MatMul → 1 个
    FLASH_ATTENTION,      // Softmax + MatMul 融合
    LUT_DIFF_LOOKUP,     // 非线性查表 (IEEE TMC 2025, 9.1× 加速)
    RESIDUAL_ADD_NORM,   // Add + LayerNorm 融合
};

struct FusedKernel {
    FusedOpType type;
    std::string name;
    int         input_channels;
    int         output_channels;
    int         kernel_size;
    int         groups;          // for group norm / group conv
    bool        has_bias;
    // 融合后的权重（预计算）
    std::vector<float> fused_weights;
    std::vector<float> fused_bias;
    std::vector<float> fused_gamma;  // BN gamma
    std::vector<float> fused_beta;   // BN beta
    std::vector<float> fused_mean;   // BN running mean
    std::vector<float> fused_var;    // BN running var
    float eps = 1e-5f;
};

// ─── 图融合引擎 ────────────────────────────────────────
class GraphFusionEngine {
public:
    GraphFusionEngine();
    ~GraphFusionEngine();

    // 注册融合规则
    void register_conv_bn_silu_pattern();
    void register_qkv_fusion_pattern();
    void register_flash_attention_pattern();
    void register_lut_diff_pattern();

    // 对计算图执行融合
    int  fuse(std::vector<FusedKernel>& out_kernels);

    // 执行融合后的 kernel
    void launch(const FusedKernel& k, const float* input, float* output,
                int batch, int height, int width);

    // LUT-Diff: 预计算查表
    void build_lut(const std::string& func_name, int table_size = 4096);
    float lut_lookup(float x) const;

    // 统计
    int   fused_op_count() const { return fused_count_; }
    float speedup_estimate() const;  // 预估加速比

private:
    int fused_count_ = 0;
    std::vector<float> lut_table_;
    int lut_size_ = 0;
    std::string lut_func_;

    // 内部：Conv+BN+SiLU 权重预融合
    void fuse_conv_bn_weights(FusedKernel& k,
                               const float* conv_w, const float* conv_b,
                               const float* bn_gamma, const float* bn_beta,
                               const float* bn_mean, const float* bn_var,
                               float eps);
};

} // namespace locai::fusion
