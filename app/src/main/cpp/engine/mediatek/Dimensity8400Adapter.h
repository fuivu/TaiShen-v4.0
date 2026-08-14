/**
 * ============================================================================
 *  Dimensity8400Adapter.h
 *  ────────────────────────────────────────────────────────────────────────────
 *  天玑 8400 专用 NPU 适配层（NPU 880 / Mali-G720 MC7）
 *
 *  ⚠️ 设计原则：
 *     本文件只定义「接口 + 数据结构 + 软逻辑」。
 *     所有真正调用 MediaTek NeuroPilot SDK 的符号都声明为 WEAK，
 *     运行时通过 dlopen() 动态加载 libneuronusdk_adapter.mtk.so 等库。
 *     如果 .so 不存在或版本不匹配 → 优雅降级到 Vulkan / CPU，
 *     绝不因为缺 .so 而崩溃。
 *
 *  芯片关键参数（来自 MediaTek 官方规格书）：
 *     • SoC        : Dimensity 8400 (MT6899 / MT6899Z)
 *     • 制程       : TSMC N4P (4nm 第二代)
 *     • CPU        : 全大核 8× Cortex-A725
 *                     1× 3.25GHz (1MB L2)
 *                     3× 3.00GHz (512KB L2)
 *                     4× 2.10GHz (256KB L2)
 *                     6MB L3 + 5MB SLC
 *     • GPU        : Mali-G720 MP7 @ 1.3GHz
 *                     Valhall 5th Gen, Vulkan 1.3, OpenCL 3.2
 *                     7 EU × 128 shader = 896 ALU
 *                     峰值 ~2329.6 GFLOPS
 *                     带宽优化 40%, 可变速率渲染 86%
 *     • NPU        : MediaTek NPU 880 (第 8 代)
 *                     整数/浮点 +20%, 能效 +18%
 *                     Stable Diffusion v1.5 提速 21%
 *                     支持 INT4 混合精度量化
 *                     支持 Diffusion Transformer (DiT)
 *                     支持 Agentic AI (DAE)
 *     • 内存       : LPDDR5X @ 8533Mbps, 4×16bit, 68.2 GB/s
 *                     最大 24GB
 *     • 工艺       : 台积电 4nm
 *
 *  ============================================================================
 */

#ifndef LOCAL_AI_PAINTER_DIMENSITY8400_ADAPTER_H
#define LOCAL_AI_PAINTER_DIMENSITY8400_ADAPTER_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <functional>

// ─────────────────────────────────────────────
//  前向声明：避免直接 #include NeuroPilot 头文件
//  （编译期不需要 SDK，运行期 dlopen 即可）
// ─────────────────────────────────────────────
typedef void* NeuronModelHandle;
typedef void* NeuronCompilationHandle;
typedef void* NeuronExecutionHandle;

namespace mediatek {
namespace dimensity8400 {

// 前向声明
class Session;

/**
 * 性能计数器（由 GetPerfCounters() 返回）。
 */
struct PerfCounters {
    uint64_t npu_active_cycles   = 0;
    uint64_t npu_idle_cycles     = 0;
    uint64_t gpu_active_cycles   = 0;
    uint64_t memory_bandwidth_gb_s = 0;
    float    avg_power_mw        = 0.0f;
    float    temperature_c       = 0.0f;
    uint32_t throttling_level    = 0;  // 0=无降频, 1=轻度, 2=重度
};

// ═══════════════════════════════════════════
//  1. 芯片能力描述符
// ═══════════════════════════════════════════

/**
 * NPU 880 支持的量化精度。
 * 天玑 8400 的 NPU 880 是第 8 代，INT4 混合精度是其杀手锏。
 */
enum class NpuPrecision : uint32_t {
    FP32    = 0,   // 全精度（调试用，NPU 上不推荐）
    FP16    = 1,   // 半精度（NPU 原生加速）
    INT8    = 2,   // 8 位对称量化（平衡精度/速度）
    INT4    = 3,   // 4 位分组量化（极致压缩，天玑 8400 特色）
    MIXED   = 4,   // INT4 权重 + FP16 激活（NeuroPilot 自动混合）
};

/**
 * GPU (Mali-G720) 计算后端选择。
 * 天玑 8400 的 Mali-G720 支持 Vulkan 1.3 + OpenCL 3.2。
 */
enum class GpuBackend : uint32_t {
    VULKAN   = 0,  // 首选：Vulkan Compute（更稳定）
    OPENCL   = 1,  // 备选：OpenCL（部分算子更快）
    AUTO     = 2,  // 让驱动自己选
};

/**
 * 芯片代际标签（用于运行时策略分支）。
 */
enum class ChipTier : uint32_t {
    UNKNOWN   = 0,
    FLAGSHIP  = 1,  // 9400 / 10000 系列
    HIGH_END  = 2,  // 8400 / 9300（我们）
    MID_RANGE = 3,  // 8300 / 8200
    LOW_END   = 4,  // 8000 / 7000
};

/**
 * 天玑 8400 的完整能力描述符。
 * 由 DetectCapabilities() 填充，供上层引擎决策。
 */
struct ChipCapabilities {
    // ── 芯片标识 ──
    ChipTier          tier            = ChipTier::UNKNOWN;
    std::string       soc_model;        // e.g. "MT6899Z"
    std::string       marketing_name;   // e.g. "Dimensity 8400"
    uint32_t          npu_generation  = 0;  // e.g. 8 (第8代 NPU)

    // ── NPU 880 ──
    bool              npu_available   = false;
    float             npu_top_int8_tops = 0.0f;  // INT8 TOPS（官方未公开，估算≈12）
    std::vector<NpuPrecision> npu_supported_precisions;

    // ── GPU (Mali-G720 MC7) ──
    bool              gpu_available   = false;
    std::string       gpu_name;         // e.g. "Mali-G720"
    uint32_t          gpu_shader_cores = 0;  // 7
    uint32_t          gpu_freq_mhz     = 0;   // 1300
    bool              gpu_supports_vulkan   = false;
    bool              gpu_supports_opencl   = false;
    bool              gpu_supports_fragment_ssr = false; // 片段着色器存储
    bool              gpu_supports_ray_tracing  = false; // 硬件光追（不用但可探测）

    // ── 内存子系统 ──
    uint64_t          total_ram_mb    = 0;
    uint64_t          l3_cache_kb     = 0;  // 6144 KB
    uint64_t          slc_cache_kb    = 0;  // 5120 KB
    uint32_t          mem_freq_mhz    = 0;  // 4266
    uint32_t          mem_channels    = 0;  // 4
    uint32_t          mem_bus_width   = 0;  // 16 bit × 4

    // ── CPU (全 A725) ──
    uint32_t          cpu_cores       = 0;  // 8
    uint32_t          cpu_max_freq_khz = 0; // 3250000
    bool              cpu_all_big_core = false; // 全大核 = true

    // ── 特性开关 ──
    bool              supports_dit        = false; // Diffusion Transformer
    bool              supports_agentic_ai = false; // DAE 引擎
    bool              supports_int4_quant = false; // INT4 混合精度
    bool              supports_memory_compress = false; // 内存硬件压缩
    bool              supports_ufs4       = false; // UFS 4.0 加速模型加载

    // ── 推荐配置（由适配层自动计算）──
    NpuPrecision      recommended_precision = NpuPrecision::INT8;
    GpuBackend        recommended_gpu       = GpuBackend::VULKAN;
    uint32_t          recommended_threads   = 4;
    uint32_t          recommended_tile_size = 512;   // UNet 分块大小
    float             estimated_sd15_speed = 0.0f;  // 秒/张 (512², 20步)
};

// ═══════════════════════════════════════════
//  2. 模型编译描述符
// ═══════════════════════════════════════════

/**
 * 模型类型枚举（决定编译策略）。
 */
enum class ModelRole : uint32_t {
    TEXT_ENCODER = 0,  // CLIP / T5 文本编码器
    UNET         = 1,  // U-Net 去噪核心（最吃 NPU）
    VAE_ENCODER  = 2,  // VAE 编码（图→潜空间）
    VAE_DECODER  = 3,  // VAE 解码（潜空间→图）
    LORA_ADAPTER = 4,  // LoRA 权重适配器
    CONTROLNET   = 5,  // ControlNet 分支
    ESRGAN       = 6,  // 超分模型
    FACE_RESTORE = 7,  // GFPGAN / CodeFormer
};

/**
 * 模型编译请求。
 * 调用方填充后传给 CompileModel()，适配层转译为 NeuroPilot 编译参数。
 */
struct ModelCompileRequest {
    ModelRole          role            = ModelRole::UNET;
    std::string        source_path;     // .onnx / .tflite / .safetensors
    std::string        output_dla_path;// 编译产物 .dla 输出路径
    NpuPrecision       target_precision = NpuPrecision::INT8;
    uint32_t           input_width      = 512;
    uint32_t           input_height     = 512;
    uint32_t           batch_size       = 1;
    // 性能 hints（NeuroPilot 会用）
    bool               prefer_low_latency  = true;
    bool               prefer_low_power     = false;
    uint32_t           max_execution_time_ms = 0; // 0 = 无限制
    // 高级：INT4 分组大小（仅 MIXED / INT4）
    uint32_t           int4_group_size      = 128;
    // 高级：算子融合开关
    bool               fuse_conv_bn         = true;
    bool               fuse_qkv_projection  = true;
    bool               fuse_softmax_matmul = true;
};

/**
 * 编译结果。
 */
struct ModelCompileResult {
    bool        success          = false;
    std::string dla_path;        // 成功时填充
    std::string error_message;
    uint64_t    compiled_size_bytes = 0;
    float       compile_time_sec    = 0.0f;
    // 编译后的内存占用预估
    uint64_t    runtime_memory_bytes = 0;
    // 实际精度（NeuroPilot 可能自动降级）
    NpuPrecision actual_precision = NpuPrecision::FP32;
};

// ═══════════════════════════════════════════
//  3. 推理会话（Session）
// ═══════════════════════════════════════════

/**
 * 推理输入张量描述。
 */
struct TensorDesc {
    std::string name;
    std::vector<int32_t> shape;
    NpuPrecision dtype;
};

/**
 * 推理会话配置。
 */
struct SessionConfig {
    std::string         dla_path;
    NpuPrecision        precision       = NpuPrecision::INT8;
    GpuBackend          gpu_fallback    = GpuBackend::VULKAN;
    uint32_t            num_threads     = 0; // 0 = 自动
    uint32_t            input_width     = 512;
    uint32_t            input_height    = 512;
    // 内存策略
    bool                use_slc_cache   = true;  // 利用 5MB SLC
    bool                use_l3_cache    = true;  // 利用 6MB L3
    uint64_t            memory_pool_mb  = 1024; // NPU 专用内存池
    // 动态形状
    bool                allow_dynamic_shape = true;
    // 功耗模式
    enum class PowerMode { HIGH = 0, BALANCED = 1, LOW = 2 };
    PowerMode           power_mode      = PowerMode::BALANCED;
};

// ═══════════════════════════════════════════
//  4. 缓存管理（利用天玑 8400 的大缓存）
// ═══════════════════════════════════════════

/**
 * 权重缓存策略（针对 6MB L3 + 5MB SLC 优化）。
 * 天玑 8400 的缓存比上代翻倍，是做权重预取的关键资源。
 */
struct WeightCacheStrategy {
    // 是否把热点权重钉在 L3 / SLC
    bool  pin_hot_weights_to_l3   = true;
    bool  pin_hot_weights_to_slc  = true;
    // LRU 容量（MB）
    uint32_t  l3_cache_capacity_mb  = 4;   // L3 = 6MB，留 2MB 给系统
    uint32_t  slc_cache_capacity_mb = 3;   // SLC = 5MB，留 2MB 给系统
    // 预取策略
    bool  prefetch_next_block     = true;   // 双缓冲预取
    uint32_t  prefetch_distance   = 2;      // 提前 2 个卷积核
    // KV Cache（文本编码器用）
    bool  enable_kv_cache         = true;
    uint32_t  kv_cache_max_tokens = 512;
};

// ═══════════════════════════════════════════
//  5. 主适配类
// ═══════════════════════════════════════════

/**
 * Dimensity8400Adapter
 *
 * 用法：
 *   auto adapter = Dimensity8400Adapter::Create();
 *   if (!adapter->IsAvailable()) { // 降级
 *       // fall back to Vulkan/CPU
 *   }
 *   auto caps = adapter->DetectCapabilities();
 *   auto session = adapter->CreateSession(session_cfg);
 *   session->LoadDla("unet.dla");
 *   session->Run(input, &output);
 *
 * 线程安全：Detect 阶段可多线程调用；Session 本身非线程安全（需外部加锁）。
 */
class Dimensity8400Adapter {
public:
    virtual ~Dimensity8400Adapter() = default;

    // ── 生命周期 ──
    static std::unique_ptr<Dimensity8400Adapter> Create();
    virtual bool Initialize() = 0;
    virtual void Shutdown() = 0;

    // ── 能力探测 ──
    virtual ChipCapabilities DetectCapabilities() = 0;
    virtual bool IsAvailable() const = 0;
    virtual std::string GetDriverVersion() const = 0;
    virtual std::string GetSdkVersion() const = 0;

    // ── 模型编译（离线 / 首次加载时）──
    virtual ModelCompileResult CompileModel(const ModelCompileRequest& req) = 0;
    virtual bool IsDlaValid(const std::string& dla_path) const = 0;

    // ── 推理会话 ──
    virtual std::unique_ptr<Session> CreateSession(const SessionConfig& cfg) = 0;

    // ── 缓存管理 ──
    virtual void ConfigureWeightCache(const WeightCacheStrategy& strategy) = 0;
    virtual void WarmupCache(const std::string& dla_path) = 0;
    virtual void ClearCache() = 0;

    // ── 性能监控 ──
    virtual PerfCounters GetPerfCounters() const = 0;

    // ── 动态降频应对 ──
    virtual void OnThermalThrottling(uint32_t level) = 0;
    virtual void SetPowerMode(SessionConfig::PowerMode mode) = 0;

protected:
    Dimensity8400Adapter() = default;
};

// ═══════════════════════════════════════════
//  6. 推理会话接口
// ═══════════════════════════════════════════

class Session {
public:
    virtual ~Session() = default;

    virtual bool LoadDla(const std::string& dla_path) = 0;
    virtual bool LoadDlaFromMemory(const void* data, size_t size) = 0;

    // 获取输入输出张量描述
    virtual std::vector<TensorDesc> GetInputTensors() const = 0;
    virtual std::vector<TensorDesc> GetOutputTensors() const = 0;

    // 同步推理
    virtual bool Run(const std::vector<const void*>& inputs,
                    std::vector<void*>& outputs) = 0;

    // 异步推理（配合双缓冲）
    using AsyncCallback = std::function<void(bool success, const std::string& error)>;
    virtual bool RunAsync(const std::vector<const void*>& inputs,
                         std::vector<void*>& outputs,
                         AsyncCallback cb) = 0;

    // 内存复用（避免每步重新分配）
    virtual void* AllocateInputBuffer(size_t size) = 0;
    virtual void  FreeInputBuffer(void* ptr) = 0;

    // 进度回调（去噪步数）
    using ProgressCallback = std::function<void(int current_step, int total_steps, float eta_sec)>;
    virtual void SetProgressCallback(ProgressCallback cb) = 0;

    // 释放
    virtual void Release() = 0;
};

// ═══════════════════════════════════════════
//  7. 量化工具函数（纯软实现，不依赖 .so）
// ═══════════════════════════════════════════

namespace quantize {

/**
 * INT8 对称量化（per-tensor）。
 * 天玑 8400 NPU 880 原生 INT8 加速，这是默认精度。
 */
void QuantizeToInt8(const float* src, int8_t* dst, size_t count,
                    float* scale_out);

/**
 * INT4 分组量化（group-wise）。
 * 天玑 8400 的杀手锏：INT4 混合精度让 SD1.5 权重从 1.6GB → 400MB。
 * group_size 推荐 128（与 NeuroPilot 默认一致）。
 */
void QuantizeToInt4Groupwise(const float* src, uint8_t* dst_packed,
                             size_t count, uint32_t group_size,
                             float* scales_out, int8_t* zeros_out);

/**
 * FP16 → INT8 动态量化（推理时在线量化激活值）。
 */
void DynamicQuantizeFp16ToInt8(const uint16_t* src, int8_t* dst,
                               size_t count, float scale);

/**
 * 计算量化后的模型大小（用于预估 NPU 内存池）。
 */
uint64_t EstimateQuantizedSize(size_t fp32_param_count, NpuPrecision prec);

} // namespace quantize

// ═══════════════════════════════════════════
//  8. 工具函数
// ═══════════════════════════════════════════

/** 检查当前进程是否运行在天玑 8400 上（读 /proc/cpuinfo + ro.board） */
bool IsRunningOnDimensity8400();

/** 读取 MediaTek 系统属性（如 ro.mediatek.platform） */
std::string GetMediatekPlatformProperty();

/** 估算 SD1.5 在 NPU 880 上的推理速度（秒/张） */
float EstimateSD15Speed(const ChipCapabilities& caps, NpuPrecision prec,
                        uint32_t steps, uint32_t width, uint32_t height);

/** 生成推荐配置（一键获取最优参数） */
SessionConfig MakeRecommendedSessionConfig(const ChipCapabilities& caps,
                                          ModelRole role);

/** 打印完整能力报告（调试用） */
void PrintCapabilities(const ChipCapabilities& caps);

} // namespace dimensity8400
} // namespace mediatek

#endif // LOCAL_AI_PAINTER_DIMENSITY8400_ADAPTER_H
