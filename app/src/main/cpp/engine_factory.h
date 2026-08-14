#pragma once

/**
 * engine_factory.h — 太神架构 v4.0 推理引擎工厂
 *
 * 职责：根据设备硬件能力自动选择最优推理后端
 *   - Vulkan (首选，GPU 计算)
 *   - OpenGL ES (中端 GPU 兜底)
 *   - CPU (最终兜底)
 *   - 天玑 APU / 骁龙 Hexagon (NPU 直通)
 *
 * 融合策略：
 *   - 动态稀疏激活 (Dynamic Sparse Attention)
 *   - 投机采样 (Speculative Decoding)
 *   - 异构算子融合 (Operator Fusion 2.0)
 */

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace taishen {

// ─── 后端类型枚举 ──────────────────────────
enum class BackendType {
    VULKAN,
    OPENGL,
    CPU,
    MNN,
    NCNN,
    ONNX,
    QNN,            // 骁龙 Hexagon
    MEDIATEK_APU,   // 天玑 APU
    UNKNOWN
};

// ─── 设备能力画像 ──────────────────────────
struct DeviceProfile {
    std::string manufacturer;
    std::string model;
    std::string chipset;
    std::string cpuArch;
    int totalRAMGB = 0;
    std::string gpuVendor;
    std::string gpuRenderer;
    bool hasVulkan = false;
    bool hasOpenGLES = false;
    bool hasOpenCL = false;
    std::string npuType;
    bool npuAvailable = false;
    std::vector<std::string> supportedBackends;
    std::string preferredBackend;
    bool supportsHybrid = false;
    int recommendedThreads = 4;
    float powerScore = 0.0f;  // 0.0 ~ 1.0
};

// ─── 推理请求（标准化契约） ────────────────
struct InferenceRequest {
    std::string prompt;
    std::string negativePrompt;
    int width = 512;
    int height = 512;
    int steps = 20;
    float cfgScale = 7.5f;
    long seed = -1;
    std::string samplerName = "EulerA";
    std::string backendOverride;  // 空 = 自动选择
    float strength = 1.0f;       // 图生图强度
    std::string initImagePath;   // 图生图输入
    std::string maskImagePath;   // Inpainting 蒙版
    bool useLora = false;
    std::vector<std::string> loraPaths;
    std::vector<float> loraWeights;
    // 太神优化参数
    bool enableOpFusion = true;
    bool enableSparseAttention = true;
    bool enableSpeculative = true;
    int sparseTopK = 64;
    int specDraftSteps = 2;
};

// ─── 推理结果 ──────────────────────────────
struct InferenceResult {
    bool success = false;
    std::string outputPath;
    std::string errorMessage;
    long elapsedMs = 0;
    float finalScore = 0.0f;
    int actualSteps = 0;
    std::string backendUsed;
    // 性能统计
    long modelLoadMs = 0;
    long unetMs = 0;
    long vaeMs = 0;
    long textEncodeMs = 0;
    long fusionMs = 0;       // 算子融合节省时间
    long sparseMs = 0;        // 稀疏注意力节省时间
    long specMs = 0;          // 投机采样节省时间
    float memoryUsedMB = 0.0f;
};

// ─── 引擎接口（纯虚基类） ──────────────────
class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    virtual bool initialize(const std::string& modelDir) = 0;
    virtual InferenceResult run(const InferenceRequest& req) = 0;
    virtual void shutdown() = 0;
    virtual std::string name() const = 0;
    virtual bool supportsNpu() const = 0;
    virtual std::string getBackendInfo() const = 0;
};

// ─── 工厂方法 ──────────────────────────────
class EngineFactory {
public:
    /**
     * 检测设备并选择最优后端
     */
    static DeviceProfile detectDevice();

    /**
     * 创建推理引擎（自动选择最优后端）
     */
    static std::unique_ptr<InferenceEngine> createAuto(const DeviceProfile& profile);

    /**
     * 创建指定后端的引擎
     */
    static std::unique_ptr<InferenceEngine> create(BackendType type);

    /**
     * 创建混合引擎（本地轻量 + 云端精调）
     */
    static std::unique_ptr<InferenceEngine> createHybrid(const DeviceProfile& profile);

    /**
     * 计算 DevicePowerScore (0.0 ~ 1.0)
     */
    static float computePowerScore(const DeviceProfile& profile);

    /**
     * 根据分数决定推理路径
     */
    static std::string decidePath(float score);

private:
    static BackendType selectBackend(const DeviceProfile& profile);
    static bool checkVulkanSupport();
    static bool checkOpenCLSupport();
    static bool checkNpuSupport(const std::string& chipset);
};

// ─── 算子融合 2.0 ──────────────────────────
namespace fusion {

/**
 * 融合 Conv + BatchNorm + SiLU 为单一 kernel
 * 返回融合后的权重缓冲区
 */
std::vector<uint8_t> fuseConvBnSilu(
    const uint8_t* convWeights, int convSize,
    const float* bnMean, const float* bnVar,
    const float* bnScale, const float* bnBias,
    int channels);

/**
 * QKV 投影融合：将 Q/K/V 三个矩阵乘合并为一次 GEMM
 */
std::vector<uint8_t> fuseQKV(
    const uint8_t* qWeights, const uint8_t* kWeights, const uint8_t* vWeights,
    int hiddenSize, int numHeads);

/**
 * Flash Attention 融合（Vulkan Compute Shader）
 */
bool enableFlashAttention(int headDim, int seqLen, bool hasTensorCores);

} // namespace fusion

// ─── 动态稀疏注意力 ────────────────────────
namespace sparse {

/**
 * TopK 稀疏化：保留注意力分数最高的 K 个 token
 */
void applyTopK(float* attentionScores, int seqLen, int topK);

/**
 * 局部窗口注意力：仅计算窗口内的注意力
 */
void applyLocalWindow(float* attentionScores, int seqLen, int windowSize);

/**
 * 可学习门控：通过小型网络预测注意力掩码
 */
void applyLearnedGating(float* attentionScores, int seqLen,
                        const float* gateWeights, int gateSize);

/**
 * 自适应稀疏策略选择
 */
std::string selectSparseStrategy(float availableMemoryMB, int seqLen);

} // namespace sparse

// ─── 投机采样 ──────────────────────────────
namespace speculative {

/**
 * 用草稿模型快速生成候选序列
 */
std::vector<int> draftGenerate(
    const uint8_t* draftModelWeights,
    const int* promptTokens, int promptLen,
    int numDraftTokens);

/**
 * 用目标模型验证草稿
 */
std::vector<bool> verifyDraft(
    const uint8_t* targetModelWeights,
    const int* promptTokens, int promptLen,
    const int* draftTokens, int draftLen,
    float acceptanceThreshold);

/**
 * 完整投机采样流程
 */
InferenceResult runSpeculativeDecoding(
    InferenceEngine* engine,
    const InferenceRequest& req,
    int draftSteps,
    float acceptThreshold);

} // namespace speculative

} // namespace taishen
