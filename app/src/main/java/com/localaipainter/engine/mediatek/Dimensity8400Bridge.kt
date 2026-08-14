/**
 * ============================================================================
 *  Dimensity8400Bridge.kt
 *  ────────────────────────────────────────────────────────────────────────────
 *  Kotlin 侧桥接：对接 C++ Dimensity8400Adapter
 *
 *  用法：
 *      val ok = Dimensity8400Bridge.init()
 *      if (ok) {
 *          val caps = Dimensity8400Bridge.getCapabilities()
 *          val cfg  = Dimensity8400Bridge.makeRecommendedConfig(ModelRole.UNET)
 *          // ...
 *      } else {
 *          // STUB 模式 / 降级到 Vulkan / CPU
 *      }
 *
 *  重要：所有 native 调用都包了 try-catch UnsatisfiedLinkError，
 *  缺 .so 时返回安全默认值，绝不崩溃。
 *  ============================================================================
 */
package com.localaipainter.engine.mediatek

import com.localaipainter.util.Logger

/**
 * 天玑 8400 专用 NPU 适配桥接器。
 *
 * 底层通过 dlopen 动态加载 NeuroPilot SDK 的 .so。
 * 如果 .so 不存在 → 进入 STUB 模式（所有查询返回默认值）。
 */
object Dimensity8400Bridge {

    private const val TAG = "D8400Bridge"
    private var initialized = false

    // ═════════════════════════════════════
    //  初始化
    // ═════════════════════════════════════

    fun init(): Boolean {
        if (initialized) return true
        return try {
            System.loadLibrary("dimensity8400_adapter")
            val ok = nativeInit()
            initialized = ok
            if (ok) Logger.i(TAG, "✅ Dimensity 8400 NPU 初始化成功")
            else     Logger.w(TAG, "⚠️ Dimensity 8400 进入 STUB 模式（缺 .so）")
            ok
        } catch (e: UnsatisfiedLinkError) {
            Logger.w(TAG, "⚠️ libdimensity8400_adapter.so 未找到 → STUB 模式", e)
            initialized = false
            false
        }
    }

    fun shutdown() {
        try { nativeShutdown() } catch (_: Throwable) {}
        initialized = false
    }

    fun isAvailable(): Boolean = try {
        init()
        nativeIsAvailable()
    } catch (_: Throwable) { false }

    // ═════════════════════════════════════
    //  能力查询
    // ═════════════════════════════════════

    /**
     * 获取完整能力描述符。
     * 失败时返回基于天玑 8400 官方规格的默认值。
     */
    fun getCapabilities(): ChipCapabilities {
        return try {
            val json = nativeGetCapabilities()
            parseCapabilities(json)
        } catch (e: Throwable) {
            Logger.w(TAG, "getCapabilities fallback → default D8400 profile", e)
            defaultCapabilities()
        }
    }

    fun getDriverVersion(): String = try { nativeGetDriverVersion() } catch (_: Throwable) { "stub" }
    fun getSdkVersion(): String   = try { nativeGetSdkVersion()   } catch (_: Throwable) { "stub" }

    // ═════════════════════════════════════
    //  模型编译
    // ═════════════════════════════════════

    fun compileModel(req: ModelCompileRequest): ModelCompileResult {
        return try {
            val role = req.role.ordinal
            val prec = req.targetPrecision.ordinal
            nativeCompileModel(role, req.sourcePath, req.outputDlaPath,
                              prec, req.inputWidth, req.inputHeight, req.batchSize)
        } catch (e: Throwable) {
            Logger.e(TAG, "compileModel failed", e)
            ModelCompileResult(success = false, error = e.message ?: "unknown")
        }
    }

    // ═════════════════════════════════════
    //  缓存管理
    // ═════════════════════════════════════

    fun configureCache(cfg: CacheStrategy) {
        try {
            nativeConfigureCache(
                cfg.pinL3, cfg.pinSlc,
                cfg.l3Mb, cfg.slcMb,
                cfg.prefetch, cfg.prefetchDist,
                cfg.kvCache, cfg.kvMaxTokens
            )
        } catch (e: Throwable) { Logger.w(TAG, "configureCache failed", e) }
    }

    fun warmupCache(dlaPath: String) {
        try { nativeWarmupCache(dlaPath) } catch (e: Throwable) {}
    }

    fun clearCache() {
        try { nativeClearCache() } catch (_: Throwable) {}
    }

    // ═════════════════════════════════════
    //  性能监控
    // ═════════════════════════════════════

    fun getPerfCounters(): PerfCounters {
        return try {
            val arr = nativeGetPerfCounters()
            PerfCounters(
                npuActiveCycles  = arr[0].toLong(),
                gpuActiveCycles  = arr[2].toLong(),
                memBandwidthGbs = arr[3],
                avgPowerMw      = arr[4],
                temperatureC    = arr[5],
                throttlingLevel = 0
            )
        } catch (_: Throwable) { PerfCounters() }
    }

    fun onThermalThrottling(level: Int) {
        try { nativeOnThermalThrottling(level) } catch (_: Throwable) {}
    }

    fun setPowerMode(mode: PowerMode) {
        try { nativeSetPowerMode(mode.ordinal) } catch (_: Throwable) {}
    }

    // ═════════════════════════════════════
    //  工具函数
    // ═════════════════════════════════════

    fun estimateSpeed(precision: NpuPrecision, steps: Int, w: Int, h: Int): Float =
        try { nativeEstimateSpeed(precision.ordinal, steps, w, h) }
        catch (_: Throwable) { 3.5f }

    fun estimateQuantizedSize(paramCount: Long, precision: NpuPrecision): Long =
        try { nativeEstimateQuantizedSize(paramCount, precision.ordinal) }
        catch (_: Throwable) { paramCount * 2 }

    fun makeRecommendedConfig(role: ModelRole): SessionConfig {
        val caps = getCapabilities()
        // 基于能力描述符 + 角色生成推荐配置
        return SessionConfig(
            precision    = caps.recommendedPrecision,
            gpuFallback  = GpuBackend.VULKAN,
            numThreads   = if (role == ModelRole.UNET) 4 else 2,
            inputWidth   = 512,
            inputHeight  = 512,
            useSlcCache  = caps.supportsInt4Quant,
            useL3Cache   = true,
            memoryPoolMb = when (role) {
                ModelRole.UNET         -> 1536
                ModelRole.VAE_DECODER  -> 512
                ModelRole.TEXT_ENCODER -> 256
                else                   -> 1024
            },
            powerMode    = PowerMode.BALANCED
        )
    }

    // ═════════════════════════════════════
    //  JNI 声明
    // ═════════════════════════════════════

    private external fun nativeInit(): Boolean
    private external fun nativeShutdown()
    private external fun nativeIsAvailable(): Boolean
    private external fun nativeGetCapabilities(): String
    private external fun nativeGetDriverVersion(): String
    private external fun nativeGetSdkVersion(): String
    private external fun nativeCompileModel(
        role: Int, srcPath: String, outPath: String,
        precision: Int, width: Int, height: Int, batchSize: Int
    ): ModelCompileResult
    private external fun nativeConfigureCache(
        pinL3: Boolean, pinSlc: Boolean,
        l3Mb: Int, slcMb: Int,
        prefetch: Boolean, prefetchDist: Int,
        kvCache: Boolean, kvMaxTokens: Int
    )
    private external fun nativeWarmupCache(dlaPath: String)
    private external fun nativeClearCache()
    private external fun nativeGetPerfCounters(): FloatArray
    private external fun nativeOnThermalThrottling(level: Int)
    private external fun nativeSetPowerMode(mode: Int)
    private external fun nativeEstimateSpeed(prec: Int, steps: Int, w: Int, h: Int): Float
    private external fun nativeEstimateQuantizedSize(paramCount: Long, prec: Int): Long

    // ═════════════════════════════════════
    //  解析 / 默认值
    // ═════════════════════════════════════

    private fun parseCapabilities(json: String): ChipCapabilities {
        // 简易 JSON 解析（不引入外部库）
        fun g(key: String): String? {
            val idx = json.indexOf("\"$key\"")
            if (idx < 0) return null
            val colon = json.indexOf(':', idx) + 1
            val start = json.indexOfFirst { it != ' ' && it != '\t' } 
            // 简化：用正则不靠谱，手撸
            return null
        }
        // 由于 C++ 侧输出的是紧凑 JSON，这里用最小解析
        val c = ChipCapabilities()
        c.socModel = json.str("soc") ?: "MT6899"
        c.marketingName = json.str("name") ?: "Dimensity 8400"
        c.npuGeneration = json.int("npu_gen") ?: 8
        c.npuAvailable = json.bool("npu_avail") ?: false
        c.gpuName = json.str("gpu") ?: "Mali-G720"
        c.gpuShaderCores = json.int("gpu_shaders") ?: 7
        c.gpuFreqMhz = json.int("gpu_freq") ?: 1300
        c.gpuSupportsVulkan = json.bool("gpu_vk") ?: true
        c.gpuSupportsOpenCL = json.bool("gpu_cl") ?: true
        c.totalRamMb = json.long("ram_mb") ?: 8192L
        c.l3CacheKb = json.int("l3_kb") ?: 6144
        c.slcCacheKb = json.int("slc_kb") ?: 5120
        c.memFreqMhz = json.int("mem_mhz") ?: 4266
        c.cpuCores = json.int("cpu_cores") ?: 8
        c.cpuMaxFreqKhz = json.int("cpu_max") ?: 3250000
        c.allBigCore = json.bool("all_big") ?: true
        c.supportsInt4Quant = json.bool("int4") ?: true
        c.supportsDiT = json.bool("dit") ?: true
        c.supportsAgenticAi = json.bool("dae") ?: true
        c.supportsUfs4 = json.bool("ufs4") ?: true
        c.recommendedPrecision = when (json.int("rec_prec") ?: 3) {
            0 -> NpuPrecision.FP32
            1 -> NpuPrecision.FP16
            2 -> NpuPrecision.INT8
            3 -> NpuPrecision.INT4
            else -> NpuPrecision.MIXED
        }
        c.recommendedGpu = GpuBackend.VULKAN
        c.recommendedThreads = json.int("rec_threads") ?: 4
        c.estimatedSd15Speed = json.float("est_sd15") ?: 2.8f
        return c
    }

    private fun defaultCapabilities(): ChipCapabilities {
        // 天玑 8400 官方规格默认值
        return ChipCapabilities(
            socModel = "MT6899",
            marketingName = "Dimensity 8400",
            npuGeneration = 8,
            npuAvailable = false, // STUB 模式
            gpuName = "Mali-G720",
            gpuShaderCores = 7,
            gpuFreqMhz = 1300,
            gpuSupportsVulkan = true,
            gpuSupportsOpenCL = true,
            totalRamMb = 8192L,
            l3CacheKb = 6144,
            slcCacheKb = 5120,
            memFreqMhz = 4266,
            cpuCores = 8,
            cpuMaxFreqKhz = 3250000,
            allBigCore = true,
            supportsInt4Quant = true,
            supportsDiT = true,
            supportsAgenticAi = true,
            supportsUfs4 = true,
            recommendedPrecision = NpuPrecision.INT4,
            recommendedGpu = GpuBackend.VULKAN,
            recommendedThreads = 4,
            estimatedSd15Speed = 2.8f
        )
    }

    // ═════════════════════════════════════
    //  String 扩展（迷你 JSON 解析）
    // ═════════════════════════════════════

    private fun String.str(key: String): String? {
        val pattern = "\"$key\":\""
        val i = indexOf(pattern); if (i < 0) return null
        val s = i + pattern.length
        val e = indexOf('"', s); if (e < 0) return null
        return substring(s, e)
    }
    private fun String.int(key: String): Int? {
        val pattern = "\"$key\":"
        val i = indexOf(pattern); if (i < 0) return null
        val s = indexOfFirst { it.isDigit() || it == '-' } 
        if (s < 0) return null
        val e = indexOfFirst { !it.isDigit() && it != '-' }
        return try { substring(s, if (e < 0) length else e).toInt() } catch (_: Throwable) { null }
    }
    private fun String.long(key: String): Long? {
        return int(key)?.toLong()
    }
    private fun String.float(key: String): Float? {
        val pattern = "\"$key\":"
        val i = indexOf(pattern); if (i < 0) return null
        val s = i + pattern.length
        val e = indexOfFirst { !it.isDigit() && it != '.' && it != '-' }
        return try { substring(s, if (e < 0) length else e).toFloat() } catch (_: Throwable) { null }
    }
    private fun String.bool(key: String): Boolean? {
        val pattern = "\"$key\":"
        val i = indexOf(pattern); if (i < 0) return null
        val s = i + pattern.length
        return when {
            startsWith("true", s)  -> true
            startsWith("false", s) -> false
            else -> null
        }
    }
    private inline fun String.indexOfFirst(pred: (Char) -> Boolean): Int {
        for (i in 0 until length) if (pred(this[i])) return i
        return -1
    }
    private inline fun String.startsWith(s: String, offset: Int): Boolean {
        if (offset + s.length > length) return false
        for (i in s.indices) if (this[offset + i] != s[i]) return false
        return true
    }
}
