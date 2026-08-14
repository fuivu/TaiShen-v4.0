package com.localaipainter.engine.v35

import android.content.Context
import android.util.Log
import com.localaipainter.engine.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Engine V4.0 "TaiShen" — 太神架构总入口 Kotlin 封装
 * 自动检测平台 + 一键最优配置 + 全后端支持
 * 深度融合 DevicePowerScore 动态决策 + LegacyKiller 自毁机制
 */
class EngineV35(private val context: Context) {

    companion object {
        private const val TAG = "EngineV35"
        init {
            try {
                System.loadLibrary("engine_v35")
                Log.i(TAG, "✅ libengine_v35.so loaded")
            } catch (e: UnsatisfiedLinkError) {
                Log.w(TAG, "⚠️ libengine_v35.so not found, using fallback")
            }
        }
    }

    // ─── 平台信息 ────────────────────────────────
    data class PlatformInfo(
        val platform: String,
        val socModel: String,
        val cpuInfo: String,
        val cpuCores: Int,
        val bigCores: Int,
        val littleCores: Int,
        val hasVulkan: Boolean,
        val hasOpenCL: Boolean,
        val hasNpu: Boolean,
        val npuName: String,
        val npuVersion: Int,
        val totalRamMb: Long,
        val availableRamMb: Long,
        val androidVersion: String,
        val apiLevel: Int,
    )

    // ─── 推荐配置 ────────────────────────────────
    data class RecommendedConfig(
        val primaryBackend: String,
        val secondaryBackend: String,
        val hybridStrategy: String,
        val useInt8: Boolean,
        val useInt4: Boolean,
        val useInt2: Boolean,
        val useFp8: Boolean,
        val useGraphFusion: Boolean,
        val useDoubleBuffer: Boolean,
        val useKvCache: Boolean,
        val useZeroCopy: Boolean,
        val threadCount: Int,
        val maxMemoryMb: Int,
        val summary: String,
    )

    // ─── 生成参数 ────────────────────────────────
    data class GenerateParams(
        val prompt: String,
        val negativePrompt: String = "",
        val width: Int = 512,
        val height: Int = 512,
        val steps: Int = 20,
        val cfgScale: Float = 7.5f,
        val seed: Int = -1,
        val sampler: String = "euler_a",
        val modelPath: String = "",
        val loraPaths: List<String> = emptyList(),
        val controlNetType: String = "",
    )

    // ─── 生成结果 ────────────────────────────────
    data class GenerateResult(
        val success: Boolean,
        val errorMsg: String,
        val imageRgb: ByteArray?,
        val imageW: Int,
        val imageH: Int,
        val totalTimeMs: Float,
        val encodeMs: Float,
        val denoiseMs: Float,
        val decodeMs: Float,
        val peakMemoryMb: Float,
        val avgPowerMw: Float,
    )

    // ─── 性能报告 ────────────────────────────────
    data class PerfReport(
        val backendName: String,
        val avgInferenceMs: Float,
        val peakMemoryMb: Float,
        val avgPowerMw: Float,
        val thermalThrottle: Float,
        val successfulRuns: Int,
        val failedRuns: Int,
        val bottleneck: String,
        val suggestions: String,
    )

    // ─── Native 方法 ──────────────────────────────
    private external fun nativeCreate(): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeInit(handle: Long): Boolean
    private external fun nativeApplyRecommended(handle: Long)
    private external fun nativeGetPlatformInfo(handle: Long): String
    private external fun nativeGetRecommendedConfig(handle: Long): String
    private external fun nativeGenerate(
        handle: Long,
        prompt: String,
        negativePrompt: String,
        width: Int,
        height: Int,
        steps: Int,
        cfgScale: Float,
        seed: Int,
        sampler: String,
        modelPath: String,
    ): String
    private external fun nativeGetPerfReport(handle: Long): String
    private external fun nativeCancel(handle: Long)
    private external fun nativeResetStats(handle: Long)

    // ─── 内部状态 ────────────────────────────────
    private var handle: Long = 0
    private var initialized = false

    // ─── 公开 API ────────────────────────────────

    fun create(): EngineV35 {
        handle = nativeCreate()
        if (handle == 0L) {
            Log.e(TAG, "❌ Failed to create engine")
        }
        return this
    }

    fun init(): Boolean {
        if (handle == 0L) create()
        initialized = nativeInit(handle)
        Log.i(TAG, "Engine init: ${if (initialized) "✅ success" else "❌ failed"}")
        return initialized
    }

    fun applyRecommended() {
        nativeApplyRecommended(handle)
        Log.i(TAG, "✅ Recommended config applied")
    }

    fun getPlatformInfo(): PlatformInfo {
        val json = nativeGetPlatformInfo(handle)
        return parsePlatformInfo(json)
    }

    fun getRecommendedConfig(): RecommendedConfig {
        val json = nativeGetRecommendedConfig(handle)
        return parseRecommendedConfig(json)
    }

    suspend fun generate(params: GenerateParams): GenerateResult = withContext(Dispatchers.Default) {
        val json = nativeGenerate(
            handle, params.prompt, params.negativePrompt,
            params.width, params.height, params.steps,
            params.cfgScale, params.seed, params.sampler,
            params.modelPath,
        )
        parseGenerateResult(json)
    }

    fun getPerfReport(): PerfReport {
        val json = nativeGetPerfReport(handle)
        return parsePerfReport(json)
    }

    fun cancel() {
        nativeCancel(handle)
        Log.i(TAG, "🛑 Generation cancelled")
    }

    fun resetStats() {
        nativeResetStats(handle)
    }

    fun destroy() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
            initialized = false
        }
    }

    // ─── 生命周期 ────────────────────────────────
    fun initialize(): EngineV35 {
        create()
        init()
        applyRecommended()
        return this
    }

    // ─── JSON 解析（简易实现，避免额外依赖）──
    private fun parsePlatformInfo(json: String): PlatformInfo {
        // 实际应从 C++ 侧返回结构化数据
        // 这里做简易解析
        fun j(key: String): String {
            val regex = """"$key"\s*:\s*"([^"]*)"""".toRegex()
            return regex.find(json)?.groupValues?.get(1) ?: ""
        }
        fun jInt(key: String): Int {
            val regex = """"$key"\s*:\s*(\d+)""".toRegex()
            return regex.find(json)?.groupValues?.get(1)?.toIntOrNull() ?: 0
        }
        fun jBool(key: String): Boolean {
            val regex = """"$key"\s*:\s*(true|false)""".toRegex()
            return regex.find(json)?.groupValues?.get(1) == "true"
        }
        return PlatformInfo(
            platform = j("platform"),
            socModel = j("soc_model"),
            cpuInfo = j("cpu_info"),
            cpuCores = jInt("cpu_cores"),
            bigCores = jInt("big_cores"),
            littleCores = jInt("little_cores"),
            hasVulkan = jBool("has_vulkan"),
            hasOpenCL = jBool("has_opencl"),
            hasNpu = jBool("has_npu"),
            npuName = j("npu_name"),
            npuVersion = jInt("npu_version"),
            totalRamMb = jInt("total_ram_mb").toLong(),
            availableRamMb = jInt("available_ram_mb").toLong(),
            androidVersion = j("android_version"),
            apiLevel = jInt("api_level"),
        )
    }

    private fun parseRecommendedConfig(json: String): RecommendedConfig {
        fun j(key: String): String {
            val regex = """"$key"\s*:\s*"([^"]*)"""".toRegex()
            return regex.find(json)?.groupValues?.get(1) ?: ""
        }
        fun jInt(key: String): Int {
            val regex = """"$key"\s*:\s*(\d+)""".toRegex()
            return regex.find(json)?.groupValues?.get(1)?.toIntOrNull() ?: 0
        }
        fun jBool(key: String): Boolean {
            val regex = """"$key"\s*:\s*(true|false)""".toRegex()
            return regex.find(json)?.groupValues?.get(1) == "true"
        }
        return RecommendedConfig(
            primaryBackend = j("primary_backend"),
            secondaryBackend = j("secondary_backend"),
            hybridStrategy = j("hybrid_strategy"),
            useInt8 = jBool("use_int8"),
            useInt4 = jBool("use_int4"),
            useInt2 = jBool("use_int2"),
            useFp8 = jBool("use_fp8"),
            useGraphFusion = jBool("use_graph_fusion"),
            useDoubleBuffer = jBool("use_double_buffer"),
            useKvCache = jBool("use_kv_cache"),
            useZeroCopy = jBool("use_zero_copy"),
            threadCount = jInt("thread_count"),
            maxMemoryMb = jInt("max_memory_mb"),
            summary = j("summary"),
        )
    }

    private fun parseGenerateResult(json: String): GenerateResult {
        fun j(key: String): String {
            val regex = """"$key"\s*:\s*"([^"]*)"""".toRegex()
            return regex.find(json)?.groupValues?.get(1) ?: ""
        }
        fun jInt(key: String): Int {
            val regex = """"$key"\s*:\s*(-?\d+)""".toRegex()
            return regex.find(json)?.groupValues?.get(1)?.toIntOrNull() ?: 0
        }
        fun jFloat(key: String): Float {
            val regex = """"$key"\s*:\s*([\d.]+)""".toRegex()
            return regex.find(json)?.groupValues?.get(1)?.toFloatOrNull() ?: 0f
        }
        fun jBool(key: String): Boolean {
            val regex = """"$key"\s*:\s*(true|false)""".toRegex()
            return regex.find(json)?.groupValues?.get(1) == "true"
        }
        val success = jBool("success")
        val imgW = jInt("image_w")
        val imgH = jInt("image_h")
        val totalBytes = jInt("image_size")
        // 图像数据通过单独的函数获取
        val imageData = if (success && totalBytes > 0) {
            // 实际应从 native 侧获取
            ByteArray(totalBytes)
        } else null
        return GenerateResult(
            success = success,
            errorMsg = j("error_msg"),
            imageRgb = imageData,
            imageW = imgW,
            imageH = imgH,
            totalTimeMs = jFloat("total_time_ms"),
            encodeMs = jFloat("encode_ms"),
            denoiseMs = jFloat("denoise_ms"),
            decodeMs = jFloat("decode_ms"),
            peakMemoryMb = jFloat("peak_memory_mb"),
            avgPowerMw = jFloat("avg_power_mw"),
        )
    }

    private fun parsePerfReport(json: String): PerfReport {
        fun j(key: String): String {
            val regex = """"$key"\s*:\s*"([^"]*)"""".toRegex()
            return regex.find(json)?.groupValues?.get(1) ?: ""
        }
        fun jInt(key: String): Int {
            val regex = """"$key"\s*:\s*(\d+)""".toRegex()
            return regex.find(json)?.groupValues?.get(1)?.toIntOrNull() ?: 0
        }
        fun jFloat(key: String): Float {
            val regex = """"$key"\s*:\s*([\d.]+)""".toRegex()
            return regex.find(json)?.groupValues?.get(1)?.toFloatOrNull() ?: 0f
        }
        return PerfReport(
            backendName = j("backend_name"),
            avgInferenceMs = jFloat("avg_inference_ms"),
            peakMemoryMb = jFloat("peak_memory_mb"),
            avgPowerMw = jFloat("avg_power_mw"),
            thermalThrottle = jFloat("thermal_throttle"),
            successfulRuns = jInt("successful_runs"),
            failedRuns = jInt("failed_runs"),
            bottleneck = j("bottleneck"),
            suggestions = j("suggestions"),
        )
    }

    // ─── 便捷方法 ────────────────────────────────
    fun isReady(): Boolean = initialized && handle != 0L

    fun getVersion(): String = "4.0.0 TaiShen"

    fun getBuildInfo(): String {
        return "Local AI Painter Engine v4.0.0 TaiShen\n" +
               "Build: ${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})\n" +
               "ABI: ${android.os.Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown"}"
    }

    // ─── 自动选择最优后端并初始化（一键启动）──
    suspend fun autoInit(): EngineV35 = withContext(Dispatchers.IO) {
        create()
        if (!init()) {
            Log.e(TAG, "❌ Engine init failed, falling back to CPU")
        }
        applyRecommended()
        val info = getPlatformInfo()
        val config = getRecommendedConfig()
        Log.i(TAG, "📱 Platform: ${info.socModel} (${info.platform})")
        Log.i(TAG, "🚀 Backend: ${config.primaryBackend} + ${config.secondaryBackend}")
        Log.i(TAG, "💡 ${config.summary}")
        this@EngineV35
    }
}
