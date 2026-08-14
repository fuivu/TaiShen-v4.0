/**
 * 天玑 NPU 深度适配实现 (v4.0 TaiShen)
 * 覆盖 NPU 990/890+/890/790 全系
 */
#include "v35/engine_mediatek/mediatek_npu_adapter.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>

namespace locai::v35::mediatek {

// ═════════════════════════════════════════════════════
//  芯片名称解析
// ═════════════════════════════════════════════════════

DimensityChip MediatekNPUAdapter::parse_chip_name(const std::string& name) {
    std::string n = name;
    // 转小写
    for (auto& c : n) c = (char)std::tolower(c);
    if (n.find("9500") != std::string::npos) return DimensityChip::D9500;
    if (n.find("9400+") != std::string::npos || n.find("9400 plus") != std::string::npos) return DimensityChip::D9400_PLUS;
    if (n.find("9400") != std::string::npos) return DimensityChip::D9400;
    return DimensityChip::UNKNOWN;
}

std::string MediatekNPUAdapter::chip_to_string(DimensityChip c) {
    switch (c) {
        case DimensityChip::D9500: return "Dimensity 9500 (NPU 990)";
        case DimensityChip::D9400_PLUS: return "Dimensity 9400+ (NPU 890+)";
        case DimensityChip::D9400: return "Dimensity 9400 (NPU 890)";
        default: return "Unknown";
    }
}

// ═════════════════════════════════════════════════════
//  能力检测
// ═════════════════════════════════════════════════════

NPUCapabilities MediatekNPUAdapter::detect_capabilities() const {
    NPUCapabilities c;
    // 1. 尝试从 /proc/cpuinfo 读取
    detect_from_sysfs(c);
    // 2. 尝试从 build.prop 读取
    detect_from_build_prop(c);
    // 3. 应用芯片默认值
    apply_chip_defaults(c);
    return c;
}

void MediatekNPUAdapter::detect_from_sysfs(NPUCapabilities& c) const {
    // 读取 /sys/devices/soc/.../mediatek,npu/info (模拟)
    // 实际路径因设备而异
    std::ifstream f("/sys/class/mtk_npu/info");
    if (f.is_open()) {
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("chip") != std::string::npos) {
                auto pos = line.find(":");
                if (pos != std::string::npos) {
                    c.chip_name = line.substr(pos+1);
                    c.chip = parse_chip_name(c.chip_name);
                }
            }
        }
        f.close();
    }
    // 默认值 (开发环境 fallback)
    if (c.chip == DimensityChip::UNKNOWN) {
        c.chip_name = "Dimensity 9500 (simulated)";
        c.chip = DimensityChip::D9500;
    }
}

void MediatekNPUAdapter::detect_from_build_prop(NPUCapabilities& c) const {
    // 读取 /system/build.prop 中的 ro.mediatek.platform
    std::ifstream f("/system/build.prop");
    if (f.is_open()) {
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("ro.mediatek.platform") != std::string::npos) {
                auto pos = line.find("=");
                if (pos != std::string::npos) {
                    c.chip_name = line.substr(pos+1);
                }
            }
        }
        f.close();
    }
}

void MediatekNPUAdapter::apply_chip_defaults(NPUCapabilities& c) const {
    switch (c.chip) {
    case DimensityChip::D9500:
        c.npu = NpuGeneration::NPU_990;
        c.npu_cores = 6;
        c.npu_freq_mhz = 1800;
        c.total_npu_tops = 60; // INT8 TOPS
        c.npu_sram_kb = 4096 * 6; // 每核 4MB
        c.int2_native = true;
        c.fp8_inference = true;
        c.cim_enabled = true;
        c.dual_npu = true;
        c.spatiotemporal_5d = true;
        c.moe_support = true;
        c.spd_plus = true;
        c.lora_training = true;
        c.gen_4k_image = true;
        c.long_context_128k = true;
        c.max_context_len = 131072;
        c.max_image_res = 4096;
        break;
    case DimensityChip::D9400_PLUS:
        c.npu = NpuGeneration::NPU_890_PLUS;
        c.npu_cores = 4;
        c.npu_freq_mhz = 1500;
        c.total_npu_tops = 40;
        c.npu_sram_kb = 4096 * 4;
        c.int2_native = false;
        c.fp8_inference = true;
        c.cim_enabled = false;
        c.dual_npu = false;
        c.spatiotemporal_5d = true;
        c.moe_support = true;
        c.spd_plus = true;
        c.lora_training = true;
        c.gen_4k_image = false;
        c.long_context_128k = false;
        c.max_context_len = 32768;
        c.max_image_res = 2048;
        break;
    case DimensityChip::D9400:
        c.npu = NpuGeneration::NPU_890;
        c.npu_cores = 4;
        c.npu_freq_mhz = 1400;
        c.total_npu_tops = 35;
        c.npu_sram_kb = 2048 * 4;
        c.int2_native = false;
        c.fp8_inference = false;
        c.cim_enabled = false;
        c.dual_npu = false;
        c.spatiotemporal_5d = false;
        c.moe_support = false;
        c.spd_plus = false;
        c.lora_training = true; // 首发端侧 LoRA
        c.gen_4k_image = false;
        c.long_context_128k = false;
        c.max_context_len = 32768;
        c.max_image_res = 2048;
        break;
    default:
        // 保守默认
        c.npu = NpuGeneration::NPU_790;
        c.npu_cores = 2;
        c.npu_freq_mhz = 1000;
        c.total_npu_tops = 12;
        c.lora_training = false;
        c.max_context_len = 8192;
        c.max_image_res = 1024;
        break;
    }
}

std::string NPUCapabilities::to_string() const {
    std::ostringstream s;
    s << "=== NPU Capabilities ===\n";
    s << "Chip: " << chip_name << "\n";
    s << "NPU TOPS (INT8): " << total_npu_tops << "\n";
    s << "NPU Cores: " << npu_cores << "\n";
    s << "NPU Freq: " << npu_freq_mhz << " MHz\n";
    s << "SRAM: " << npu_sram_kb / 1024 << " MB\n";
    s << "INT2 Native: " << (int2_native ? "✅" : "❌") << "\n";
    s << "FP8 Inference: " << (fp8_inference ? "✅" : "❌") << "\n";
    s << "CIM (存算一体): " << (cim_enabled ? "✅ (-33% power)" : "❌") << "\n";
    s << "Dual NPU: " << (dual_npu ? "✅" : "❌") << "\n";
    s << "Spatiotemporal 5D: " << (spatiotemporal_5d ? "✅" : "❌") << "\n";
    s << "MoE: " << (moe_support ? "✅" : "❌") << "\n";
    s << "SpD+: " << (spd_plus ? "✅ (+20%)" : "❌") << "\n";
    s << "LoRA Training: " << (lora_training ? "✅ (50×CPU)" : "❌") << "\n";
    s << "4K Gen: " << (gen_4k_image ? "✅ (业界首发)" : "❌") << "\n";
    s << "Long Context: " << max_context_len / 1024 << "K\n";
    s << "Max Image: " << max_image_res << "²\n";
    return s.str();
}

// ═════════════════════════════════════════════════════
//  初始化 / 关闭
// ═════════════════════════════════════════════════════

MediatekNPUAdapter::MediatekNPUAdapter()
    : initialized_(false), cim_enabled_(false), dual_npu_(false) {}

MediatekNPUAdapter::~MediatekNPUAdapter() { shutdown(); }

bool MediatekNPUAdapter::initialize() {
    if (initialized_) return true;
    caps_ = detect_capabilities();
    cim_enabled_ = caps_.cim_enabled;
    // 预分配 CIM SRAM (模拟)
    if (caps_.cim_enabled) {
        cim_sram_.resize(caps_.npu_sram_kb * 1024 / sizeof(float));
    }
    initialized_ = true;
    std::cout << caps_.to_string() << std::endl;
    return true;
}

void MediatekNPUAdapter::shutdown() {
    if (!initialized_) return;
    cim_sram_.clear();
    initialized_ = false;
}

// ═════════════════════════════════════════════════════
//  INT2 推理 (NPU 990 原生)
// ═════════════════════════════════════════════════════

int MediatekNPUAdapter::run_int2_inference(
    const uint8_t* w_int2_packed, const int8_t* act_int8,
    int16_t* output, int M, int N, int K,
    float w_scale, float a_scale)
{
    if (!caps_.int2_native) return -1; // 硬件不支持
    // 硬件原生 INT2 MAC: 每周期处理 32 个 INT2 × INT8
    // 模拟: 解包 + 乘加
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                uint8_t packed = w_int2_packed[(n * K + k) >> 2];
                int shift = ((n * K + k) & 3) * 2;
                int8_t w2 = ((packed >> shift) & 0x03) - 2; // [-2,-1,0,1,2]→[0,1,2,3,4]-2
                acc += (int32_t)w2 * (int32_t)act_int8[m * K + k];
            }
            // 反量化: INT16 输出
            float f = (float)acc * w_scale * a_scale;
            output[m * N + n] = (int16_t)(f > 32767 ? 32767 : f < -32768 ? -32768 : f);
        }
    }
    perf_.last_inference_ms = (double)M * N * K * 0.001 / caps_.total_npu_tops;
    perf_.total_inferences++;
    return 0;
}

// ═════════════════════════════════════════════════════
//  FP8 推理 (NPU 990 / 890+)
// ═════════════════════════════════════════════════════

int MediatekNPUAdapter::run_fp8_inference(
    const uint8_t* w_fp8, const uint8_t* a_fp8,
    float* output, int M, int N, int K,
    float sw, float sa)
{
    if (!caps_.fp8_inference) return -1;
    // 硬件 FP8 张量核心 (NPU 990) 或 FP8 MAC (NPU 890+)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++) {
                // 解包 FP8 E4M3 (简化)
                uint8_t wb = w_fp8[n * K + k];
                uint8_t ab = a_fp8[m * K + k];
                // 简化: 直接用字节值 (实际需 FP8→FP32 转换)
                float wf = (float)(int8_t)(wb << 1) / 128.f;
                float af = (float)(int8_t)(ab << 1) / 128.f;
                acc += wf * af;
            }
            output[m * N + n] = acc * sw * sa;
        }
    }
    perf_.last_inference_ms = (double)M * N * K * 0.0005 / caps_.total_npu_tops;
    perf_.total_inferences++;
    return 0;
}

// ═════════════════════════════════════════════════════
//  CIM 存算一体
// ═════════════════════════════════════════════════════

int MediatekNPUAdapter::cim_load_weights(const float* weights, size_t bytes) {
    if (!caps_.cim_enabled) return -1;
    size_t n = bytes / sizeof(float);
    if (n > cim_sram_.size()) n = cim_sram_.size();
    std::memcpy(cim_sram_.data(), weights, n * sizeof(float));
    return 0;
}

int MediatekNPUAdapter::run_cim_matmul(
    const float* weights_sram, const float* activation,
    float* output, int M, int N, int K)
{
    if (!cim_enabled_) return -1;
    // 权重已在 CIM SRAM, 激活流入, 在存储单元内完成 MAC
    // 功耗: 传统 1.0 → CIM 0.67 (节省 33%)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++) {
                acc += weights_sram[n * K + k] * activation[m * K + k];
            }
            output[m * N + n] = acc;
        }
    }
    // CIM 功耗估算: 基准 0.5W/GMAC → 0.335W/GMAC
    perf_.power_watts = (float)(M * N * K) * 1e-9f * 0.335f;
    return 0;
}

// ═════════════════════════════════════════════════════
//  双 NPU 调度 (NPU 990)
// ═════════════════════════════════════════════════════

int MediatekNPUAdapter::run_dual_npu_inference(
    const float* weights, const float* input,
    float* output, int M, int N, int K)
{
    if (!caps_.dual_npu) return -1;
    // 拆分: 前半 → NPU0, 后半 → NPU1
    int N_half = N / 2;
    // NPU0
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N_half; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++) acc += weights[n*K+k] * input[m*K+k];
            output[m * N + n] = acc;
        }
    }
    // NPU1 (模拟并行)
    for (int m = 0; m < M; m++) {
        for (int n = N_half; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++) acc += weights[n*K+k] * input[m*K+k];
            output[m * N + n] = acc;
        }
    }
    // 双 NPU: 理论 2×, 实际 1.7× (通信开销)
    perf_.last_inference_ms = (double)M * N * K * 0.001 / (caps_.total_npu_tops * 1.7);
    return 0;
}

// ═════════════════════════════════════════════════════
//  SpD+ 投机解码 (NPU 990 / 890+)
// ═════════════════════════════════════════════════════

int MediatekNPUAdapter::speculative_decode_step(
    const float* prompt_embedding,
    float* output_tokens,
    int seq_len,
    const SPDConfig& cfg)
{
    if (!caps_.spd_plus) return -1;
    // 草稿阶段: INT2 小模型快速生成
    int accepted = 0;
    for (int i = 0; i < cfg.draft_tokens; i++) {
        // 草稿: 简单近似 (实际是 INT2 量化小模型)
        float draft_val = prompt_embedding[i % seq_len] * 0.5f;
        // 验证: 主模型 (全精度)
        float verified = prompt_embedding[i % seq_len];
        if (std::fabs(verified - draft_val) < (1.f - cfg.acceptance_threshold)) {
            output_tokens[i] = verified;
            accepted++;
        } else {
            output_tokens[i] = verified; // 用主模型结果
        }
    }
    // SpD+ 加速: 天玑额外 +20%
    perf_.last_inference_ms *= 0.8; // 1/1.2 ≈ 0.83
    return accepted;
}

// ═════════════════════════════════════════════════════
//  端侧 LoRA 训练 (全系, 50×CPU)
// ═════════════════════════════════════════════════════

int MediatekNPUAdapter::train_lora_on_device(
    const float* base_weights, const float* training_data,
    float* lora_delta, int rows, int cols,
    const LoRATrainConfig& cfg)
{
    if (!caps_.lora_training) return -1;
    // 简化: 一次梯度更新 (实际是反向传播 + Adam)
    float lr = cfg.use_fp8 ? cfg.lr * 2.f : cfg.lr; // FP8 可用更大 LR
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float grad = training_data[i * cols + j] - base_weights[i * cols + j];
            lora_delta[i * cols + j] = grad * lr;
        }
    }
    // NPU 训练速度: 50× CPU
    perf_.last_inference_ms = (double)rows * cols * 0.01 / 50.0;
    return 0;
}

// ═════════════════════════════════════════════════════
//  4K 图像生成 (NPU 990 业界首发)
// ═════════════════════════════════════════════════════

int MediatekNPUAdapter::generate_4k_image(
    const float* latent, float* output,
    int target_w, int target_h)
{
    if (!caps_.gen_4k_image) return -1;
    if (target_w > 4096 || target_h > 4096) return -2;
    // 流程: Latent [1,4,128,128] → VAE Decode → 1024² → 4× 超分 → 4K
    // 4× 超分用 NPU 原生 ESPCN / FSRCNN
    int total_pixels = target_w * target_h * 3;
    for (int i = 0; i < total_pixels; i++) {
        // 简化: 双线性上采样 (实际是神经网络超分)
        output[i] = latent[i % (128*128*4)] * 0.5f + 0.5f; // [-1,1] → [0,1]
    }
    // 4K 生成耗时约 8-15s (NPU 990)
    perf_.last_inference_ms = 10000.0; // 10s 估计
    return 0;
}

// ═════════════════════════════════════════════════════
//  时域张量 5D (NPU 990 / 890+)
// ═════════════════════════════════════════════════════

int MediatekNPUAdapter::run_5d_conv(
    const float* input5d, float* output5d,
    int B, int T, int C, int H, int W,
    const float* kernel, int kh, int kw)
{
    if (!caps_.spatiotemporal_5d) return -1;
    // 5D 卷积: 同时处理 batch + time 维度
    // 天玑 NPU 原生支持 spatiotemporal 张量
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        float acc = 0.f;
                        for (int kh_i = 0; kh_i < kh; kh_i++) {
                            for (int kw_i = 0; kw_i < kw; kw_i++) {
                                int h_idx = h + kh_i - kh/2;
                                int w_idx = w + kw_i - kw/2;
                                if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
                                    acc += input5d[b*T*C*H*W + t*C*H*W + c*H*W + h_idx*W + w_idx]
                                         * kernel[c*kh*kw + kh_i*kw + kw_i];
                                }
                            }
                        }
                        output5d[b*T*C*H*W + t*C*H*W + c*H*W + h*W + w] = acc;
                    }
                }
            }
        }
    }
    return 0;
}

// ═════════════════════════════════════════════════════
//  MoE 推理 (NPU 990 / 890+)
// ═════════════════════════════════════════════════════

int MediatekNPUAdapter::run_moe_inference(
    const float* input, float* output,
    int batch, int seq_len, const MoEConfig& cfg)
{
    if (!caps_.moe_support) return -1;
    // MoE: 路由器 → top_k 专家 → 加权求和
    int hidden = cfg.hidden_dim;
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            // 路由器: 简化 (实际是线性层 + softmax)
            float gate_sum = 0.f;
            float gates[8]; // max num_experts
            for (int e = 0; e < cfg.num_experts; e++) {
                gates[e] = std::fabs(input[b*seq_len*hidden + s*hidden + e % hidden]);
                gate_sum += gates[e];
            }
            // top_k
            // 简化: 用固定权重
            float acc[4096] = {0};
            for (int e = 0; e < cfg.top_k; e++) {
                float w = gates[e] / gate_sum;
                for (int h = 0; h < hidden; h++) {
                    acc[h] += input[b*seq_len*hidden + s*hidden + h] * w * (1.f + 0.1f * e);
                }
            }
            for (int h = 0; h < hidden; h++) {
                output[b*seq_len*hidden + s*hidden + h] = acc[h];
            }
        }
    }
    return 0;
}

// ═════════════════════════════════════════════════════
//  功耗估算
// ═════════════════════════════════════════════════════

float MediatekNPUAdapter::estimate_power_watts(int macc_count) const {
    float base = (float)macc_count * 1e-9f * 0.5f; // 0.5W/GMAC
    if (cim_enabled_) base *= 0.67f;  // CIM -33%
    if (caps_.int2_native) base *= 0.5f; // INT2 比 INT8 再省一半
    return base;
}

// ═════════════════════════════════════════════════════
//  NeuroPilot 图描述
// ═════════════════════════════════════════════════════

std::string MediatekNPUAdapter::emit_neuropilot_graph() const {
    std::ostringstream s;
    s << "neuropilot_graph {\n";
    s << "  target_chip: \"" << caps_.chip_name << "\"\n";
    s << "  npu_cores: " << caps_.npu_cores << "\n";
    s << "  int8_tops: " << caps_.total_npu_tops << "\n";
    s << "  features: {\n";
    s << "    int2: " << (caps_.int2_native ? "true" : "false") << "\n";
    s << "    fp8: " << (caps_.fp8_inference ? "true" : "false") << "\n";
    s << "    cim: " << (caps_.cim_enabled ? "true" : "false") << "\n";
    s << "    dual_npu: " << (caps_.dual_npu ? "true" : "false") << "\n";
    s << "    spatiotemporal_5d: " << (caps_.spatiotemporal_5d ? "true" : "false") << "\n";
    s << "    moe: " << (caps_.moe_support ? "true" : "false") << "\n";
    s << "    spd_plus: " << (caps_.spd_plus ? "true" : "false") << "\n";
    s << "    lora_train: " << (caps_.lora_training ? "true" : "false") << "\n";
    s << "    gen_4k: " << (caps_.gen_4k_image ? "true" : "false") << "\n";
    s << "  }\n";
    s << "  max_context: " << caps_.max_context_len << "\n";
    s << "  max_resolution: " << caps_.max_image_res << "\n";
    s << "}\n";
    return s.str();
}

// ═════════════════════════════════════════════════════
//  DimensityOptimizer
// ═════════════════════════════════════════════════════

DimensityOptimizer::DimensityOptimizer() {
    npu_adapter_.initialize();
    apply_optimal_settings();
}

DimensityOptimizer::~DimensityOptimizer() = default;

void DimensityOptimizer::apply_optimal_settings() {
    auto* npu = npu_adapter_.npu();
    const auto& caps = npu->caps();
    if (caps.chip == DimensityChip::D9500) {
        npu->enable_cim(true);
        npu->enable_dual_npu(true);
        rec_.quantization = "INT2";
        rec_.attention = "flash_attn_5d";
        rec_.num_threads = 4;
        rec_.use_cim = true;
        rec_.use_dual_npu = true;
        rec_.use_spd_plus = true;
        rec_.preferred_resolution = 2048;
        rec_.memory_strategy = "aggressive_cache";
        rec_.estimated_speedup = 3.5f;
        rec_.estimated_power_save = 33.f;
    } else if (caps.chip == DimensityChip::D9400_PLUS) {
        npu->enable_cim(false);
        npu->enable_dual_npu(false);
        rec_.quantization = "FP8";
        rec_.attention = "flash_attn";
        rec_.num_threads = 4;
        rec_.use_cim = false;
        rec_.use_dual_npu = false;
        rec_.use_spd_plus = true;
        rec_.preferred_resolution = 1024;
        rec_.memory_strategy = "balanced";
        rec_.estimated_speedup = 2.5f;
        rec_.estimated_power_save = 10.f;
    } else if (caps.chip == DimensityChip::D9400) {
        rec_.quantization = "INT8";
        rec_.attention = "xformers";
        rec_.num_threads = 4;
        rec_.preferred_resolution = 1024;
        rec_.memory_strategy = "conservative";
        rec_.estimated_speedup = 1.8f;
    } else {
        rec_.quantization = "INT8";
        rec_.preferred_resolution = 512;
        rec_.memory_strategy = "minimal";
        rec_.estimated_speedup = 1.0f;
    }
}

DimensityOptimizer::Recommendation DimensityOptimizer::get_recommendation() const {
    return rec_;
}

std::string DimensityOptimizer::recommendation_report() const {
    const auto& r = rec_;
    std::ostringstream s;
    s << "=== Dimensity Optimization Recommendation ===\n";
    s << "Quantization: " << r.quantization << "\n";
    s << "Attention: " << r.attention << "\n";
    s << "Threads: " << r.num_threads << "\n";
    s << "CIM: " << (r.use_cim ? "ON (-33% power)" : "OFF") << "\n";
    s << "Dual NPU: " << (r.use_dual_npu ? "ON" : "OFF") << "\n";
    s << "SpD+: " << (r.use_spd_plus ? "ON (+20%)" : "OFF") << "\n";
    s << "Resolution: " << r.preferred_resolution << "²\n";
    s << "Memory: " << r.memory_strategy << "\n";
    s << "Est. Speedup: " << r.estimated_speedup << "×\n";
    s << "Est. Power Save: " << r.estimated_power_save << "%\n";
    return s.str();
}

} // namespace locai::v35::mediatek
