package com.localaipainter.engine

import android.graphics.Bitmap

/**
 * InferenceEngine —— 所有推理引擎的统一接口
 *
 * 定义了完整的 AI 绘画推理生命周期：
 *   初始化 → 加载模型 → 编码文本 → 推理(去噪) → 解码潜空间 → 释放
 *
 * 所有后端（QNN/MNN/NCNN/ORT/Vulkan/CPU/Neuron）必须实现此接口
 */
interface InferenceEngine {

    // ============ 生命周期 ============

    /** 初始化引擎（加载原生库、分配资源） */
    fun init()

    /** 是否已初始化 */
    fun isInitialized(): Boolean

    /** 释放所有资源 */
    fun release()

    /** 预热（执行一次空推理，触发 JIT/着色器编译） */
    fun warmup()

    // ============ 模型管理 ============

    /** 加载模型（自动检测格式） */
    fun loadModel(path: String): Boolean

    /** 卸载模型 */
    fun unloadModel()

    /** 验证模型文件完整性 */
    fun validateModel(path: String): Boolean

    /** 获取模型信息 */
    fun getModelInfo(path: String): ModelInfo

    // ============ 推理核心 ============

    /**
     * 执行一次推理步（去噪一步）
     * @param input 当前潜空间张量
     * @param width  输出宽度
     * @param height 输出高度
     * @return 预测噪声
     */
    fun runInference(input: FloatArray, width: Int, height: Int): FloatArray

    /**
     * 编码文本提示词为条件向量
     */
    fun encodeText(text: String, clipSkip: Int = 1): FloatArray

    /**
     * 解码潜空间为 Bitmap 图片
     */
    fun decodeLatent(latent: FloatArray, w: Int, h: Int): Bitmap

    /**
     * 编码图片为潜空间（图生图用）
     */
    fun encodeLatent(bitmap: Bitmap, w: Int, h: Int): FloatArray

    // ============ LoRA ============

    /** 加载单个 LoRA（带权重） */
    fun loadLora(path: String, scale: Float): Boolean

    /** 批量加载 LoRA */
    fun loadLoraBatch(loras: Map<Long, Float>): Boolean

    /** 卸载所有 LoRA */
    fun unloadLoras()

    // ============ 信息查询 ============

    /** 后端名称（如 "QNN", "Vulkan", "CPU"） */
    val backendName: String

    /** 是否支持 NPU 加速 */
    fun supportsNpu(): Boolean

    /** 当前是否可用 */
    fun isAvailable(): Boolean

    /** 当前内存占用 (MB) */
    fun getMemoryUsageMB(): Int

    /** 总内存 (MB) */
    fun getTotalMemory(): Long

    /** 支持的精度列表 */
    fun getSupportedPrecisions(): List<String>

    /** 是否支持指定精度 */
    fun supportsPrecision(precision: String): Boolean

    /** 性能统计 */
    fun getPerfStats(): EnginePerfStats

    // ============ 配置 ============

    /** 设置线程数 */
    fun setThreadCount(threads: Int)

    /** 设置功耗模式 */
    fun setPowerMode(mode: PowerMode)

    /** 启用/禁用算子融合 */
    fun setOpFusion(enabled: Boolean)

    /** 启用/禁用内存增长 */
    fun setMemoryGrowth(enabled: Boolean)
}
