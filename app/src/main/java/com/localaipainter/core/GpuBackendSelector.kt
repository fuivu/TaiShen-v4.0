package com.localaipainter.core

import android.content.Context
import com.localaipainter.util.Logger

/**
 * 🎯 GPU 后端自动选择器 v2.0
 *
 * 决策树（按优先级）：
 *   1. 用户手动锁定 → 尊重用户选择
 *   2. 芯片首选后端可用 → 用它
 *   3. 有 NPU → 用 NPU（最快）
 *   4. 有 Vulkan → 用 Vulkan GPU
 *   5. 降级到 NCNN CPU
 *   6. 最终兜底 CpuEngine
 *
 * 同时决定：精度（FP32/FP16/INT8/INT4）+ 推荐分辨率 + 推荐步数
 */
class GpuBackendSelector(private val context: Context) {

    data class Recommendation(
        val backend: String,           // "QNN" / "MNN" / "NEURON" / "VULKAN" / "NCNN" / "CPU"
        val precision: String,         // "FP32" / "FP16" / "INT8" / "INT4"
        val recommendedSteps: IntRange,
        val recommendedCfg: ClosedFloatingPointRange<Float>,
        val maxResolution: Int,        // 单边最大像素
        val recommendedResolution: Pair<Int, Int>,  // (w, h)
        val supportsLora: Boolean,
        val supportsImg2Img: Boolean,
        val expectedLatencyMs: Long,    // 512x512 估算延迟
        val reason: String            // 选择理由（展示给用户）
    )

    fun select(deviceInfo: DeviceInfo, userPreference: String? = null): Recommendation {
        // 1. 用户锁定
        if (userPreference != null && deviceInfo.supports(userPreference)) {
            Logger.i("GpuSelector", "用户锁定后端: $userPreference")
            return buildRec(userPreference, deviceInfo, "用户手动选择")
        }

        // 2. 按芯片优先
        val preferred = deviceInfo.preferredBackend
        if (deviceInfo.supports(preferred)) {
            Logger.i("GpuSelector", "芯片首选: $preferred")
            return buildRec(preferred, deviceInfo, "芯片首选后端")
        }

        // 3. NPU 优先
        if (deviceInfo.npuType != NpuType.NONE) {
            val npuBackend = when (deviceInfo.npuType) {
                NpuType.HEXAGON -> "QNN"
                NpuType.MEDIATEK_APU -> "NEURON"
                NpuType.KIRIN_NPU -> "MNN"
                NpuType.EXYNOS_NPU -> "MNN"
                NpuType.GOOGLE_TPU -> "MNN"
                else -> "MNN"
            }
            if (deviceInfo.supports(npuBackend)) {
                Logger.i("GpuSelector", "NPU 可用: $npuBackend")
                return buildRec(npuBackend, deviceInfo, "${deviceInfo.npuName} 可用")
            }
        }

        // 4. Vulkan GPU
        if (deviceInfo.hasVulkan) {
            Logger.i("GpuSelector", "Vulkan GPU 可用")
            return buildRec("VULKAN", deviceInfo, "Vulkan GPU 加速")
        }

        // 5. NCNN CPU
        Logger.i("GpuSelector", "降级到 NCNN CPU")
        return buildRec("NCNN", deviceInfo, "无 GPU/NPU，使用 NCNN CPU 优化")
    }

    private fun buildRec(backend: String, info: DeviceInfo, reason: String): Recommendation {
        val isHighEnd = info.isHighEnd
        val isLowEnd = info.isLowEnd
        val ramGB = info.totalRamMB / 1024

        // 精度选择
        val precision = when {
            backend == "QNN" || backend == "NEURON" -> if (isHighEnd) "INT8" else "INT8"
            backend == "VULKAN" || backend == "MNN" -> if (isHighEnd) "FP16" else "FP16"
            backend == "NCNN" -> if (isLowEnd) "INT8" else "FP16"
            else -> "FP16"
        }

        // 分辨率
        val maxRes = when {
            isHighEnd -> 1024
            ramGB >= 6 -> 768
            ramGB >= 4 -> 640
            else -> 512
        }
        val recRes = when {
            isHighEnd -> 768 to 768
            ramGB >= 6 -> 640 to 640
            else -> 512 to 512
        }

        // 步数
        val steps = when (backend) {
            "QNN", "NEURON" -> 4..20      // NPU 快，可用多步
            "VULKAN", "MNN" -> 8..30
            else -> 10..50               // CPU 慢，鼓励多步换质量
        }

        // 估算延迟（512x512 基准）
        val latency = when (backend) {
            "QNN" -> if (isHighEnd) 3000 else 5000
            "NEURON" -> if (isHighEnd) 4000 else 6000
            "VULKAN" -> if (isHighEnd) 8000 else 15000
            "MNN" -> if (isHighEnd) 10000 else 20000
            "NCNN" -> 30000
            else -> 60000
        }

        return Recommendation(
            backend = backend,
            precision = precision,
            recommendedSteps = steps,
            recommendedCfg = 7f..9f,
            maxResolution = maxRes,
            recommendedResolution = recRes,
            supportsLora = backend != "CPU",
            supportsImg2Img = true,
            expectedLatencyMs = latency.toLong(),
            reason = reason
        )
    }

    /**
     * 获取所有可用后端及其推荐配置（供 UI 展示）
     */
    fun getAllOptions(deviceInfo: DeviceInfo): List<Recommendation> {
        return deviceInfo.supportedBackends.map { backend ->
            buildRec(backend, deviceInfo, "可用后端")
        }.sortedBy { it.expectedLatencyMs }  // 最快的排前面
    }

    /**
     * 校验用户选择是否可行，不可行时返回建议
     */
    fun validateChoice(backend: String, precision: String, resolution: Int, deviceInfo: DeviceInfo): ValidationResult {
        val issues = mutableListOf<String>()
        if (!deviceInfo.supports(backend)) issues.add("后端 $backend 在此设备上不可用")
        if (resolution > buildRec(backend, deviceInfo, "").maxResolution)
            issues.add("分辨率 $resolution 超出推荐上限")
        if (precision == "INT4" && backend == "CPU") issues.add("INT4 在纯 CPU 上效果差")
        return ValidationResult(issues.isEmpty(), issues)
    }

    data class ValidationResult(val valid: Boolean, val issues: List<String>)
}
