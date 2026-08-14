package com.localaipainter.core

import com.localaipainter.engine.SchedulerType
import com.localaipainter.engine.ControlNetType
import com.localaipainter.lora.LoraWeight

/**
 * TaiShenInferenceRequest — 太神架构标准化推理请求包
 *
 * 这是太神内核与底层硬件/开源模块之间的**唯一契约**。
 * 底层模块（ONNX Runtime / NCNN / MNN / Vulkan / NPU）只能接收此包，
 * 禁止任何非标准调用。
 *
 * 设计原则（来自太神架构总纲）：
 *   - 标准化契约：底层只认 TaiShenInferenceRequest
 *   - 绝对控制权：错误解析、资源调度、性能归因 100% 由太神内核控制
 *   - 日志纯净度：底层日志统一转为 tai_shen_0.2 协议格式
 *
 * 红线1：旧设备日志中禁止出现 "ONNX"/"MLKit"/"TFLite"
 * 红线2：所有设备统一显示 "太神 AI 引擎 4.0"
 * 红线3：100% 流量走太神私有协议
 */
data class TaiShenInferenceRequest(
    // ════════════════════════════════════════════
    //  文本输入
    // ════════════════════════════════════════════
    val prompt: String,
    val negativePrompt: String = "",

    // ════════════════════════════════════════════
    //  图像参数
    // ════════════════════════════════════════════
    val width: Int = 512,
    val height: Int = 512,
    val steps: Int = 20,
    val cfgScale: Float = 7.5f,
    val seed: Int = -1,

    // ════════════════════════════════════════════
    //  调度器 & 后端
    // ════════════════════════════════════════════
    val scheduler: SchedulerType = SchedulerType.EULER_A,
    val backend: String = "vulkan",  // vulkan/qnn/mnn/ncnn/onnx/cpu

    // ════════════════════════════════════════════
    //  模型路径
    // ════════════════════════════════════════════
    val modelPath: String = "",
    val vaePath: String = "",
    val textEncoderPath: String = "",

    // ════════════════════════════════════════════
    //  LoRA 权重（最多5路叠加）
    // ════════════════════════════════════════════
    val loras: List<LoraWeight> = emptyList(),

    // ════════════════════════════════════════════
    //  ControlNet
    // ════════════════════════════════════════════
    val controlNetType: ControlNetType = ControlNetType.NONE,
    val controlNetImagePath: String = "",
    val controlNetStrength: Float = 1.0f,

    // ════════════════════════════════════════════
    //  量化 & 优化
    // ════════════════════════════════════════════
    val quantization: String = "int8",  // fp32/int8/int4/int2/fp8
    val useGraphFusion: Boolean = true,
    val useDoubleBuffer: Boolean = true,
    val useZeroCopy: Boolean = true,
    val useKvCache: Boolean = true,

    // ════════════════════════════════════════════
    //  性能约束（由 DevicePowerScore 决定）
    // ════════════════════════════════════════════
    val maxMemoryMb: Int = 4096,
    val maxInferenceTimeMs: Int = 30000,
    val threadCount: Int = 4,

    // ════════════════════════════════════════════
    //  协议版本（红线1：日志纯净度）
    // ════════════════════════════════════════════
    val protocolVersion: String = "tai_shen_0.2",

    // ════════════════════════════════════════════
    //  设备分段标签（红线3：流量主权）
    // ════════════════════════════════════════════
    val deviceSegment: String = "native",  // native/legacy/hybrid
) {
    companion object {
        const val PROTOCOL_VERSION = "tai_shen_0.2"
        const val MAX_LORA_COUNT = 5
    }

    /**
     * 验证请求合法性
     */
    fun validate(): ValidationResult {
        val errors = mutableListOf<String>()

        if (prompt.isBlank()) errors.add("prompt 不能为空")
        if (width <= 0 || height <= 0) errors.add("宽高必须大于0")
        if (width > 2048 || height > 2048) errors.add("宽高不能超过2048")
        if (steps <= 0 || steps > 150) errors.add("steps 范围 1-150")
        if (cfgScale < 1f || cfgScale > 30f) errors.add("cfgScale 范围 1-30")
        if (loras.size > MAX_LORA_COUNT) errors.add("LoRA 最多 $MAX_LORA_COUNT 路")
        if (threadCount <= 0 || threadCount > 16) errors.add("threadCount 范围 1-16")

        return if (errors.isEmpty()) {
            ValidationResult(true, "OK")
        } else {
            ValidationResult(false, errors.joinToString("; "))
        }
    }

    /**
     * 转换为底层引擎可消费的参数 Map
     * 所有底层模块通过此接口获取参数，禁止直接访问字段
     */
    fun toEngineParams(): Map<String, Any> {
        return mapOf(
            "prompt" to prompt,
            "negative_prompt" to negativePrompt,
            "width" to width,
            "height" to height,
            "steps" to steps,
            "cfg_scale" to cfgScale,
            "seed" to seed,
            "scheduler" to scheduler.name.lowercase(),
            "backend" to backend,
            "model_path" to modelPath,
            "vae_path" to vaePath,
            "text_encoder_path" to textEncoderPath,
            "lora_count" to loras.size,
            "lora_weights" to loras.map { it.weight },
            "lora_paths" to loras.map { it.path },
            "controlnet_type" to controlNetType.name.lowercase(),
            "controlnet_image" to controlNetImagePath,
            "controlnet_strength" to controlNetStrength,
            "quantization" to quantization,
            "graph_fusion" to useGraphFusion,
            "double_buffer" to useDoubleBuffer,
            "zero_copy" to useZeroCopy,
            "kv_cache" to useKvCache,
            "max_memory_mb" to maxMemoryMb,
            "max_time_ms" to maxInferenceTimeMs,
            "thread_count" to threadCount,
            "protocol" to protocolVersion,
            "device_segment" to deviceSegment,
        )
    }

    /**
     * 创建云端推理变体（旧设备/弱网时使用）
     * 自动设置 protocol + device_segment 标签
     */
    fun toCloudVariant(): TaiShenInferenceRequest {
        return copy(
            backend = "cloud",
            deviceSegment = "legacy",
            useGraphFusion = false,
            useDoubleBuffer = false,
            useZeroCopy = false,
        )
    }

    /**
     * 创建本地高性能变体（旗舰机时使用）
     */
    fun toLocalVariant(): TaiShenInferenceRequest {
        return copy(
            backend = "vulkan",
            deviceSegment = "native",
            useGraphFusion = true,
            useDoubleBuffer = true,
            useZeroCopy = true,
            useKvCache = true,
        )
    }

    data class ValidationResult(val isValid: Boolean, val message: String)
}
