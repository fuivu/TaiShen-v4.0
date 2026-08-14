package com.localaipainter.core

import android.content.Context
import com.localaipainter.engine.InferenceEngine

/**
 * 推理后端能力标签 —— 用于自动匹配最优后端
 */
enum class BackendCapability {
    CPU,                    // 纯 CPU
    GPU_VULKAN,             // Vulkan GPU
    GPU_OPENGL,             // OpenGL ES 3.0+ GPU
    GPU_OPENCL,             // OpenCL GPU
    NPU_QUALCOMM,           // 骁龙 Hexagon
    NPU_MEDIATEK,           // 天玑 APU / NPU 880
    NPU_HUAWEI,             // 麒麟 NPU
    NPU_SAMSUNG,            // 三星 NPU
    NPU_GENERIC,            // 通用 NPU 接口
    QUANT_INT8,             // 支持 INT8 量化
    QUANT_INT4,             // 支持 INT4 量化
    QUANT_FP16,             // 支持 FP16
    LOOPS_UNET,             // 能跑 U-Net
    LOOPS_VAE,              // 能跑 VAE
    LOOPS_TEXT_ENC           // 能跑 TextEncoder
}

/**
 * 推理后端提供者 —— 插件式注册
 *
 * 新增后端的步骤：
 *   1. 实现 InferenceEngine 接口
 *   2. 在 PluginRegistry.registerBuiltinBackends() 中调用 registerBackend()
 *   3. 在 DeviceDetector 中把芯片映射到该后端
 * 不需要修改任何已有调用代码。
 */
data class BackendProvider(
    val name: String,
    val description: String,
    val capabilities: Set<BackendCapability>,
    val priority: Int = 0, // 同能力下优先级，越高越优先
    val factory: (Context) -> InferenceEngine
) {
    fun create(context: Context): InferenceEngine = factory(context)

    fun supports(cap: BackendCapability): Boolean = capabilities.contains(cap)
}
