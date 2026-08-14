package com.localaipainter.engine

import android.content.Context
import android.graphics.Bitmap
import com.localaipainter.util.Logger

/**
 * 🔧 推理引擎抽象基类 —— 提供通用实现，减少子类重复代码
 *
 * 子类只需实现核心方法（predictNoise / decodeLatent / encodeText / backendName / supportsNpu），
 * 其余通用逻辑（生命周期、内存统计、性能采集、LoRA 管理）已在此实现。
 */
abstract class BaseEngine(
    protected val context: Context,
    protected val engineTag: String
) : InferenceEngine {

    companion object { private const val TAG_SUFFIX = "Engine" }

    // ---- 生命周期 ----
    protected var initialized = false
    protected var currentPrecision = "FP16"
    protected var currentDeviceType = "CPU"
    protected var modelLoaded = false

    // ---- 统计 ----
    protected var totalInferences = 0L
    protected var totalTimeMs = 0L
    protected var peakMemoryMB = 0L
    protected var cacheHits = 0L
    protected var cacheMisses = 0L

    // ---- LoRA ----
    protected val loadedLoras = linkedMapOf<Long, Float>()

    // ---- 配置 ----
    protected var threadCount = Runtime.getRuntime().availableProcessors()
    protected var powerMode = PowerMode.BALANCED
    protected var opFusionEnabled = true
    protected var memoryGrowthEnabled = true

    // ============ InferenceEngine 接口实现 ============

    override val backendName: String
        get() = backendName()

    abstract fun backendName(): String
    abstract fun supportsNpu(): Boolean

    override fun init() {
        initialized = true
        Logger.i(engineTag, "引擎初始化完成")
    }

    override fun isInitialized(): Boolean = initialized

    override fun release() {
        loadedLoras.clear()
        modelLoaded = false
        initialized = false
        Logger.i(engineTag, "引擎已释放 (推理${totalInferences}次, 平均${if(totalInferences>0) totalTimeMs/totalInferences else 0}ms)")
    }

    override fun warmup() {
        if (!modelLoaded) { Logger.w(engineTag, "预热失败：模型未加载"); return }
        Logger.d(engineTag, "预热开始...")
        Logger.d(engineTag, "预热完成")
    }

    // ---- 模型管理 ----

    override fun loadModel(path: String): Boolean =
        loadModel(path, currentDeviceType, currentPrecision)

    open fun loadModel(path: String, backend: String, precision: String): Boolean {
        currentPrecision = precision
        currentDeviceType = backend
        modelLoaded = validateModel(path)
        Logger.i(engineTag, "loadModel path=$path backend=$backend precision=$precision -> $modelLoaded")
        return modelLoaded
    }

    override fun unloadModel() {
        modelLoaded = false
        Logger.i(engineTag, "模型已卸载")
    }

    override fun validateModel(path: String): Boolean {
        val f = java.io.File(path)
        if (!f.exists()) { Logger.w(engineTag, "模型不存在: $path"); return false }
        if (f.length() < 1024) { Logger.w(engineTag, "模型文件过小: ${f.length()}B"); return false }
        return true
    }

    override fun getModelInfo(path: String): ModelInfo {
        val f = java.io.File(path)
        return ModelInfo(
            name = f.nameWithoutExtension,
            format = f.extension.uppercase(),
            fileSizeMB = f.length() / (1024 * 1024),
            isValid = validateModel(path)
        )
    }

    // ---- 推理核心（子类必须重写） ----

    abstract override fun runInference(input: FloatArray, width: Int, height: Int): FloatArray

    // ---- LoRA ----

    override fun loadLora(path: String, scale: Float): Boolean {
        val id = path.hashCode().toLong()
        loadedLoras[id] = scale.coerceIn(-2f, 2f)
        Logger.i(engineTag, "LoRA 加载: ${java.io.File(path).name} scale=$scale")
        return true
    }

    override fun loadLoraBatch(loras: Map<Long, Float>): Boolean {
        loras.forEach { (id, scale) -> loadedLoras[id] = scale }
        Logger.i(engineTag, "批量加载 ${loras.size} 个 LoRA")
        return true
    }

    override fun unloadLoras() {
        val count = loadedLoras.size
        loadedLoras.clear()
        Logger.i(engineTag, "卸载 $count 个 LoRA")
    }

    // ---- 信息查询 ----

    override fun isAvailable(): Boolean = modelLoaded && initialized

    override fun getMemoryUsageMB(): Int {
        val used = ((Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024)).toInt()
        if (used > peakMemoryMB) peakMemoryMB = used.toLong()
        return used
    }

    override fun getTotalMemory(): Long =
        Runtime.getRuntime().maxMemory() / (1024 * 1024)

    override fun supportsPrecision(precision: String): Boolean =
        getSupportedPrecisions().contains(precision)

    override fun getSupportedPrecisions(): List<String> =
        listOf("FP32", "FP16", "INT8")

    override fun getPerfStats(): EnginePerfStats {
        val avg = if (totalInferences > 0) totalTimeMs.toDouble() / totalInferences else 0.0
        val hitRate = if (cacheHits + cacheMisses > 0)
            cacheHits.toFloat() / (cacheHits + cacheMisses) else 0f
        return EnginePerfStats(
            totalInferences = totalInferences,
            totalTimeMs = totalTimeMs,
            avgTimeMs = avg,
            peakMemoryMB = peakMemoryMB,
            cacheHitRate = hitRate
        )
    }

    // ---- 配置 ----

    override fun setThreadCount(threads: Int) {
        threadCount = threads.coerceIn(1, Runtime.getRuntime().availableProcessors())
        Logger.d(engineTag, "线程数=$threadCount")
    }

    override fun setPowerMode(mode: PowerMode) {
        powerMode = mode
        Logger.d(engineTag, "功率模式=$mode")
    }

    override fun setOpFusion(enabled: Boolean) {
        opFusionEnabled = enabled
        Logger.d(engineTag, "算子融合=$enabled")
    }

    override fun setMemoryGrowth(enabled: Boolean) {
        memoryGrowthEnabled = enabled
        Logger.d(engineTag, "内存增长=$enabled")
    }

    // ---- 默认实现（子类应重写） ----

    override fun encodeText(text: String, clipSkip: Int): FloatArray {
        Logger.v(engineTag, "encodeText '${text.take(20)}...' skip=$clipSkip")
        return FloatArray(768) { kotlin.random.Random(text.hashCode() + it).nextFloat() * 2 - 1 }
    }

    override fun decodeLatent(latent: FloatArray, w: Int, h: Int): Bitmap {
        Logger.v(engineTag, "decodeLatent ${w}x$h")
        return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
    }

    override fun encodeLatent(bitmap: Bitmap, w: Int, h: Int): FloatArray {
        val px = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(px, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val latent = FloatArray(px.size * 3)
        for (i in px.indices) {
            latent[i*3]   = (((px[i] shr 16) and 0xFF) / 127.5f) - 1f
            latent[i*3+1] = (((px[i] shr 8) and 0xFF) / 127.5f) - 1f
            latent[i*3+2] = ((px[i] and 0xFF) / 127.5f) - 1f
        }
        return latent
    }

    // ============ 推理计时辅助 ============

    protected inline fun <T> timeOperation(tag: String, block: () -> T): T {
        val start = System.currentTimeMillis()
        val result = block()
        val elapsed = System.currentTimeMillis() - start
        totalInferences++
        totalTimeMs += elapsed
        if (elapsed > 5000) Logger.w(engineTag, "$tag 慢操作: ${elapsed}ms")
        else Logger.v(engineTag, "$tag: ${elapsed}ms")
        return result
    }

    protected fun recordCacheHit() { cacheHits++ }
    protected fun recordCacheMiss() { cacheMisses++ }
}
