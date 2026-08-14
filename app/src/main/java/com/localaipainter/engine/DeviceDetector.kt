package com.localaipainter.engine

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import com.localaipainter.util.Logger

/**
 * DeviceDetector (engine 层) —— 完整版硬件检测
 *
 * 深度融合检测能力：
 *   - SoC 识别（骁龙/天玑/麒麟/Exynos/Google Tensor）
 *   - GPU 检测（Adreno/Mali/PowerVR/Apple）
 *   - NPU 识别（Hexagon/APU/Neural Engine）
 *   - Vulkan / OpenCL / OpenGL ES 支持检测
 *   - 内存大小检测
 *   - 推荐推理后端选择
 *   - 混动模式支持判断
 */
class DeviceDetector(private val context: Context) {

    companion object {
        private const val TAG = "DeviceDetector"
    }

    data class DeviceInfo(
        val manufacturer: String,
        val model: String,
        val chipset: String,
        val cpuArch: String,
        val totalRAMGB: Int,
        val gpuVendor: String,
        val gpuRenderer: String,
        val hasVulkan: Boolean,
        val hasOpenGLES: Boolean,
        val hasOpenCL: Boolean,
        val npuType: String,        // "Hexagon" / "APU" / "NeuralEngine" / "DaVinci" / ""
        val npuAvailable: Boolean,
        val supportedBackends: List<String>,
        val preferredBackend: String,
        val supportsHybrid: Boolean,
        val recommendedThreads: Int,
    )

    fun detect(): DeviceInfo {
        Logger.d(TAG, "=== 开始硬件检测 ===")

        val manufacturer = Build.MANUFACTURER ?: "Unknown"
        val model = Build.MODEL ?: "Unknown"
        val chipset = detectChipset()
        val cpuArch = Build.SUPPORTED_ABIS?.firstOrNull() ?: "arm64-v8a"
        val totalRAMGB = getTotalRAMGB()
        val gpuInfo = detectGPU()
        val hasVulkan = checkVulkanSupport()
        val hasOpenGLES = checkOpenGLESSupport()
        val hasOpenCL = checkOpenCLSupport()
        val npuInfo = detectNPU(chipset)
        val backends = determineBackends(chipset, hasVulkan, hasOpenCL, npuInfo.second)
        val preferred = selectPreferredBackend(backends, npuInfo.second)
        val supportsHybrid = hasVulkan && Runtime.getRuntime().availableProcessors() >= 4
        val recommendedThreads = Runtime.getRuntime().availableProcessors().coerceIn(2, 8)

        val info = DeviceInfo(
            manufacturer = manufacturer,
            model = model,
            chipset = chipset,
            cpuArch = cpuArch,
            totalRAMGB = totalRAMGB,
            gpuVendor = gpuInfo.first,
            gpuRenderer = gpuInfo.second,
            hasVulkan = hasVulkan,
            hasOpenGLES = hasOpenGLES,
            hasOpenCL = hasOpenCL,
            npuType = npuInfo.first,
            npuAvailable = npuInfo.second,
            supportedBackends = backends,
            preferredBackend = preferred,
            supportsHybrid = supportsHybrid,
            recommendedThreads = recommendedThreads,
        )

        Logger.i(TAG, "设备: $manufacturer $model")
        Logger.i(TAG, "芯片: $chipset | GPU: ${gpuInfo.first} ${gpuInfo.second}")
        Logger.i(TAG, "NPU: ${npuInfo.first} (${if (npuInfo.second) "可用" else "不可用"})")
        Logger.i(TAG, "Vulkan=$hasVulkan OpenCL=$hasOpenCL OpenGL=$hasOpenGLES")
        Logger.i(TAG, "推荐后端: $preferred | 混动: $supportsHybrid")
        Logger.i(TAG, "内存: ${totalRAMGB}GB | 推荐线程: $recommendedThreads")

        return info
    }

    // ===== 芯片识别 =====

    private fun detectChipset(): String {
        val board = Build.BOARD?.lowercase() ?: ""
        val hardware = Build.HARDWARE?.lowercase() ?: ""
        val socModel = try {
            @Suppress("PrivateApi")
            Class.forName("android.os.SystemProperties")
                .getMethod("get", String::class.java)
                .invoke(null, "ro.soc.model") as? String ?: ""
        } catch (_: Exception) { "" }

        // 天玑系列
        if (board.contains("mt689") || socModel.contains("dimensity", ignoreCase = true)) {
            return when {
                board.contains("mt6983") || socModel.contains("9000") -> "MediaTek Dimensity 9000"
                board.contains("mt6985") || socModel.contains("9200") -> "MediaTek Dimensity 9200"
                board.contains("mt6895") || socModel.contains("8300") -> "MediaTek Dimensity 8300"
                board.contains("mt6897") || socModel.contains("8400") -> "MediaTek Dimensity 8400"
                board.contains("mt6989") || socModel.contains("9400") -> "MediaTek Dimensity 9400"
                else -> "MediaTek Dimensity (Unknown)"
            }
        }

        // 骁龙系列
        if (board.contains("sm8") || hardware.contains("qcom") || socModel.contains("snapdragon", ignoreCase = true)) {
            return when {
                board.contains("sm8650") || socModel.contains("8 gen 3") -> "Qualcomm Snapdragon 8 Gen 3"
                board.contains("sm8550") || socModel.contains("8 gen 2") -> "Qualcomm Snapdragon 8 Gen 2"
                board.contains("sm8450") || socModel.contains("8 gen 1") -> "Qualcomm Snapdragon 8 Gen 1"
                board.contains("sm8475") || socModel.contains("8+") -> "Qualcomm Snapdragon 8+ Gen 1"
                else -> "Qualcomm Snapdragon (Unknown)"
            }
        }

        // 麒麟/海思
        if (hardware.contains("kirin") || board.contains("hi3")) {
            return "HiSilicon Kirin"
        }

        // Google Tensor
        if (board.contains("gs") || manufacturer.contains("google")) {
            return "Google Tensor"
        }

        // Exynos
        if (board.contains("exynos") || hardware.contains("samsung")) {
            return "Samsung Exynos"
        }

        return socModel.ifBlank { board.ifBlank { "Unknown SoC" } }
    }

    // ===== GPU 检测 =====

    private fun detectGPU(): Pair<String, String> {
        return try {
            val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? android.app.ActivityManager
            val configInfo = activityManager?.deviceConfigurationInfo
            val reqGlEsVersion = configInfo?.reqGlEsVersion ?: 0
            val glVersion = "${reqGlEsVersion shr 16}.${reqGlEsVersion and 0xFFFF}"

            // 通过 OpenGL 扩展字符串检测 GPU 厂商
            val vendor = when {
                tryReadBuildProp("ro.hardware.egl")?.contains("adreno", ignoreCase = true) == true -> "Qualcomm Adreno"
                tryReadBuildProp("ro.hardware.egl")?.contains("mali", ignoreCase = true) == true -> "ARM Mali"
                tryReadBuildProp("ro.hardware.egl")?.contains("powervr", ignoreCase = true) == true -> "Imagination PowerVR"
                else -> "Unknown GPU"
            }

            Pair(vendor, "OpenGL ES $glVersion")
        } catch (e: Exception) {
            Logger.w(TAG, "GPU 检测失败: ${e.message}")
            Pair("Unknown", "Unknown")
        }
    }

    // ===== Vulkan 支持 =====

    private fun checkVulkanSupport(): Boolean {
        return try {
            val pm = context.packageManager
            pm.hasSystemFeature(PackageManager.FEATURE_VULKAN)
        } catch (e: Exception) {
            Logger.w(TAG, "Vulkan 检测异常: ${e.message}")
            false
        }
    }

    // ===== OpenGL ES 支持 =====

    private fun checkOpenGLESSupport(): Boolean {
        return try {
            val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? android.app.ActivityManager
            val info = activityManager?.deviceConfigurationInfo
            info != null && info.reqGlEsVersion >= 0x20000
        } catch (e: Exception) { false }
    }

    // ===== OpenCL 支持 =====

    private fun checkOpenCLSupport(): Boolean {
        // OpenCL 支持通常通过系统库存在性判断
        return try {
            val libPaths = listOf(
                "/system/lib64/libOpenCL.so",
                "/system/vendor/lib64/libOpenCL.so",
                "/system/lib/libOpenCL.so"
            )
            libPaths.any { java.io.File(it).exists() }
        } catch (e: Exception) { false }
    }

    // ===== NPU 检测 =====

    private fun detectNPU(chipset: String): Pair<String, Boolean> {
        val lower = chipset.lowercase()
        return when {
            lower.contains("dimensity") || lower.contains("mediatek") -> {
                // 天玑 APU：检查 NeuroPilot 库
                val hasNPU = try {
                    java.io.File("/system/lib64/libneuropilot.so").exists() ||
                    java.io.File("/system/vendor/lib64/libapu.so").exists() ||
                    java.io.File("/system/lib64/libmtkneu.so").exists()
                } catch (_: Exception) { false }
                Pair("MediaTek APU", hasNPU)
            }
            lower.contains("snapdragon") || lower.contains("qualcomm") -> {
                // 骁龙 Hexagon：检查 QNN 库
                val hasNPU = try {
                    java.io.File("/system/lib64/libQnnHtp.so").exists() ||
                    java.io.File("/system/vendor/lib64/libQnnHtp.so").exists()
                } catch (_: Exception) { false }
                Pair("Qualcomm Hexagon DSP/NPU", hasNPU)
            }
            lower.contains("kirin") || lower.contains("hisilicon") -> {
                Pair("Huawei DaVinci NPU", false) // 受限，暂不支持
            }
            lower.contains("tensor") -> {
                Pair("Google Edge TPU", false) // 待适配
            }
            else -> Pair("", false)
        }
    }

    // ===== 后端选择 =====

    private fun determineBackends(
        chipset: String,
        hasVulkan: Boolean,
        hasOpenCL: Boolean,
        npuAvailable: Boolean
    ): List<String> {
        val backends = mutableListOf<String>()
        val lower = chipset.lowercase()

        if (npuAvailable) {
            when {
                lower.contains("dimensity") -> backends.add("mediatek_npu")
                lower.contains("qualcomm") || lower.contains("snapdragon") -> backends.add("qnn")
            }
        }
        if (hasVulkan) backends.add("vulkan")
        if (hasOpenCL) backends.add("opencl")
        backends.add("mnn")
        backends.add("ncnn")
        backends.add("onnx")
        backends.add("cpu")

        return backends
    }

    private fun selectPreferredBackend(backends: List<String>, npuAvailable: Boolean): String {
        return if (npuAvailable && backends.isNotEmpty()) backends.first() else "vulkan"
    }

    // ===== 工具方法 =====

    private fun getTotalRAMGB(): Int {
        return try {
            val mi = android.app.ActivityManager.MemoryInfo()
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
            am.getMemoryInfo(mi)
            (mi.totalMem / (1024 * 1024 * 1024)).toInt().coerceAtLeast(1)
        } catch (e: Exception) { 4 }
    }

    private fun tryReadBuildProp(key: String): String? {
        return try {
            @Suppress("PrivateApi")
            Class.forName("android.os.SystemProperties")
                .getMethod("get", String::class.java)
                .invoke(null, key) as? String
        } catch (_: Exception) { null }
    }
}
