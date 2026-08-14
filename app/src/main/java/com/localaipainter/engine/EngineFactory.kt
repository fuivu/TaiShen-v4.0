package com.localaipainter.engine

import android.content.Context
import com.localaipainter.util.Logger

/**
 * EngineFactory —— 推理后端工厂（Kotlin 端）
 *
 * 深度融合六大推理后端：
 *   - QNN      (Qualcomm Snapdragon NPU)
 *   - MNN      (Alibaba 轻量推理)
 *   - NCNN     (Tencent 移动端推理)
 *   - ONNX RT  (Microsoft ONNX Runtime)
 *   - Vulkan   (跨平台 GPU 计算)
 *   - CPU      (兜底方案)
 *
 * 选择策略（按优先级）：
 *   1. 天玑 NPU (Dimensity 8400/9000/9200+) → 功耗最低
 *   2. 骁龙 NPU (Snapdragon 8 Gen2/3) → QNN 后端
 *   3. Vulkan GPU → 最快通用 GPU 路径
 *   4. OpenCL → 老款 GPU 兼容
 *   5. MNN / NCNN → 纯 CPU 优化
 *   6. CPU → 最后兜底
 */
object EngineFactory {

    private const val TAG = "EngineFactory"

    // Context 引用（由 Application 初始化时设置）
    private var appContext: Context? = null

    fun init(context: Context) {
        appContext = context.applicationContext
    }

    private val context: Context?
        get() = appContext

    // ─── 后端优先级列表（从高到低）──────────────
    private val backendPriority = listOf(
        "mediatek_npu",  // 天玑 NPU（最低功耗）
        "qnn",           // 骁龙 NPU
        "vulkan",        // Vulkan GPU
        "opencl",        // OpenCL GPU
        "mnn",           // MNN CPU
        "ncnn",          // NCNN CPU
        "onnx",          // ONNX Runtime
        "cpu"            // 兜底
    )

    /**
     * 自动检测并返回最佳后端名称
     */
    fun getBestBackend(): String {
        val deviceInfo = detectDevice()
        Logger.i(TAG, "设备检测: $deviceInfo")

        // 天玑优先
        if (deviceInfo.contains("mt689", ignoreCase = true) ||
            deviceInfo.contains("mt698", ignoreCase = true) ||
            deviceInfo.contains("dimensity", ignoreCase = true)) {
            Logger.i(TAG, "✅ 选择天玑 NPU 后端")
            return "mediatek_npu"
        }

        // 骁龙优先
        if (deviceInfo.contains("snapdragon", ignoreCase = true) ||
            deviceInfo.contains("sm8", ignoreCase = true)) {
            Logger.i(TAG, "✅ 选择 QNN 后端")
            return "qnn"
        }

        // 有 Vulkan 支持 → GPU 路径
        if (hasVulkanSupport()) {
            Logger.i(TAG, "✅ 选择 Vulkan GPU 后端")
            return "vulkan"
        }

        // 兜底：CPU
        Logger.i(TAG, "⚠️ 使用 CPU 兜底后端")
        return "cpu"
    }

    /**
     * 创建指定后端的推理引擎实例
     */
    fun tryCreate(context: Context, backend: String): InferenceEngine? {
        return try {
            when (backend.lowercase()) {
                "qnn" -> QnnEngine(context)
                "mnn" -> MnnEngine(context)
                "ncnn" -> NcnnEngine(context)
                "onnx" -> OnnxEngine(context)
                "vulkan" -> VulkanEngine(context)
                "mediatek_npu", "npu" -> NeuronEngine(context)
                "cpu" -> CpuEngine(context)
                else -> {
                    Logger.w(TAG, "未知后端: $backend, 尝试 CPU 兜底")
                    CpuEngine(context)
                }
            }
        } catch (e: UnsatisfiedLinkError) {
            Logger.e(TAG, "后端 $backend 加载失败: ${e.message}, 降级到 CPU")
            try { CpuEngine(context) } catch (e2: Exception) { null }
        } catch (e: Exception) {
            Logger.e(TAG, "后端 $backend 初始化异常: ${e.message}")
            null
        }
    }

    /**
     * 获取所有可用后端列表
     */
    fun getAvailableBackends(context: Context): List<String> {
        val available = mutableListOf<String>()
        for (backend in backendPriority) {
            try {
                val engine = tryCreate(context, backend)
                if (engine != null && engine.isAvailable()) {
                    available.add(backend)
                    engine.release()
                }
            } catch (_: Exception) { /* 跳过不可用后端 */ }
        }
        if (available.isEmpty()) available.add("cpu")
        return available
    }

    /**
     * 自动选择并创建最佳引擎（一步到位）
     */
    fun createBest(context: Context): InferenceEngine {
        val backend = getBestBackend()
        return tryCreate(context, backend) ?: CpuEngine(context)
    }

    /**
     * 创建混动引擎（CPU + GPU 协同）
     * 由 PluginRegistry 调用
     */
    fun createHybrid(context: Context, detector: com.localaipainter.engine.DeviceDetector.DeviceInfo): InferenceEngine {
        return try {
            // 优先 Vulkan GPU
            if (detector.hasVulkan) {
                Logger.i(TAG, "混动模式: Vulkan GPU + CPU")
                VulkanEngine(context)
            } else if (detector.hasOpenCL) {
                Logger.i(TAG, "混动模式: OpenCL GPU + CPU")
                OpenGLEngine(context)
            } else {
                Logger.i(TAG, "混动模式: CPU only")
                CpuEngine(context)
            }
        } catch (e: Exception) {
            Logger.w(TAG, "混动创建失败，降级 CPU: ${e.message}")
            CpuEngine(context)
        }
    }

    /**
     * 根据名称创建引擎（简化版，供 PluginRegistry 使用）
     */
    fun create(context: Context, name: String): InferenceEngine {
        return tryCreate(context, name) ?: CpuEngine(context)
    }

    // ─── 私有工具方法 ──────────────────────────────

    private fun detectDevice(): String {
        return try {
            val props = java.io.File("/system/build.prop")
            if (props.exists()) {
                props.readLines().find { it.contains("ro.product.board", ignoreCase = true) } ?: ""
            } else ""
        } catch (_: Exception) { "" }
    }

    private fun hasVulkanSupport(): Boolean {
        return try {
            val pm = context?.packageManager
            pm != null && android.content.pm.PackageManager::class.java
                .getMethod("hasSystemFeature", String::class.java)
                .invoke(pm, "android.hardware.vulkan") as? Boolean ?: false
        } catch (_: Exception) { false }
    }

    /**
     * 获取后端显示名称
     */
    fun getDisplayName(backend: String): String = when (backend.lowercase()) {
        "qnn" -> "Qualcomm QNN (NPU)"
        "mnn" -> "Alibaba MNN"
        "ncnn" -> "Tencent NCNN"
        "onnx" -> "ONNX Runtime"
        "vulkan" -> "Vulkan GPU"
        "opencl" -> "OpenCL GPU"
        "mediatek_npu", "npu" -> "MediaTek NPU"
        "cpu" -> "CPU (兜底)"
        else -> backend
    }
}
