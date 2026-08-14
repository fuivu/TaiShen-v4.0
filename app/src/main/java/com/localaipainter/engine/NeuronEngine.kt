package com.localaipainter.engine

import android.content.Context
import android.graphics.Bitmap
import com.localaipainter.util.Logger

/**
 * MediaTek Neuron / NPU 880 引擎 —— 天玑真·NPU
 *
 * ⚠️ 重要前提：
 *   - 需要 MediaTek NeuroPilot SDK（适配天玑 8400/9000/9300/9400）
 *   - 模型必须用 ncc-tflite 离线编译为 .dla 文件
 *   - .dla 文件存放于 assets/ 或 files 目录
 *   - 运行时通过 Neuron Adapter API 加载 .dla
 *
 * 当前是"框架代码 + JNI 桩"，编译能过、运行能加载，
 * 真正的 NPU 推理需要把 NeuroPilot SDK 的 .so 放进 jniLibs。
 *
 * 扩展：未来天玑 10000+ 若改 API，只需改 native 层，本类不变。
 */
class NeuronEngine(context: Context) : BaseEngine(context, "NEURON") {

    private var nativeHandle: Long = 0L
    private var textEncHandle: Long = 0L
    private var unetHandle: Long = 0L
    private var vaeHandle: Long = 0L
    private var dlaLoaded = false

    override fun backendName(): String = "Neuron NPU 880 (天玑)"

    override fun supportsNpu(): Boolean = true  // 这就是天玑真 NPU

    override fun getSupportedPrecisions(): List<String> =
        listOf("FP32", "FP16", "INT8", "INT4")  // NPU 880 全精度支持

    override fun init() {
        super.init()
        try {
            System.loadLibrary("neuron_wrapper")
            nativeHandle = nativeCreate()
            if (nativeHandle == 0L) {
                Logger.e(engineTag, "Neuron 创建失败（驱动不支持？）")
                initialized = false
                return
            }
            // 查询 NPU 信息
            val npuInfo = nativeGetNpuInfo(nativeHandle)
            Logger.i(engineTag, "✅ Neuron NPU 初始化成功: $npuInfo")
        } catch (e: UnsatisfiedLinkError) {
            Logger.e(engineTag, "Neuron 库加载失败（未集成 NeuroPilot SDK）", e)
            initialized = false
        }
    }

    override fun loadModel(path: String, backend: String, precision: String): Boolean {
        if (!initialized) init()
        if (nativeHandle == 0L) return false

        currentPrecision = precision
        currentDeviceType = "NPU"

        return timeOperation("loadModel") {
            val modelType = detectModelType(path)
            val handle = when (modelType) {
                "text_encoder" -> {
                    textEncHandle = nativeLoadDla(nativeHandle, path, "CPU", precision)
                    textEncHandle
                }
                "unet" -> {
                    // U-Net 是 NPU 的主战场
                    unetHandle = nativeLoadDla(nativeHandle, path, "NPU", precision)
                    unetHandle
                }
                "vae" -> {
                    vaeHandle = nativeLoadDla(nativeHandle, path, "NPU", precision)
                    vaeHandle
                }
                else -> nativeLoadDla(nativeHandle, path, "NPU", precision)
            }
            modelLoaded = handle != 0L
            dlaLoaded = modelLoaded
            if (modelLoaded) {
                Logger.i(engineTag, "DLA 加载成功: ${java.io.File(path).name} ($modelType) precision=$precision")
            } else {
                Logger.e(engineTag, "DLA 加载失败: $path（确认是 ncc-tflite 编译的 .dla？）")
            }
            modelLoaded
        }
    }

    override fun encodeText(text: String, clipSkip: Int): FloatArray {
        if (textEncHandle == 0L) return super.encodeText(text, clipSkip)
        return timeOperation("encodeText") {
            try { nativeEncodeText(textEncHandle, text, clipSkip) }
            catch (e: Throwable) { Logger.e(engineTag, "encodeText", e); FloatArray(768) }
        }
    }

    override fun predictNoise(
        latent: FloatArray, textEmb: FloatArray, timestep: Int, w: Int, h: Int
    ): FloatArray {
        if (unetHandle == 0L) return super.predictNoise(latent, textEmb, timestep, w, h)
        return timeOperation("predictNoise") {
            try { nativePredictNoise(unetHandle, latent, textEmb, timestep, w, h) }
            catch (e: Throwable) {
                Logger.e(engineTag, "predictNoise step=$timestep", e)
                latent.copyOf()  // 返回原 latent，避免管线崩溃
            }
        }
    }

    override fun decodeLatent(latent: FloatArray, w: Int, h: Int): Bitmap {
        if (vaeHandle == 0L) return super.decodeLatent(latent, w, h)
        return timeOperation("decodeLatent") {
            try { nativeDecodeLatent(vaeHandle, latent, w, h) }
            catch (e: Throwable) {
                Logger.e(engineTag, "decodeLatent", e)
                Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            }
        }
    }

    override fun encodeLatent(bitmap: Bitmap, w: Int, h: Int): FloatArray {
        if (vaeHandle == 0L) return super.encodeLatent(bitmap, w, h)
        return timeOperation("encodeLatent") {
            try { nativeEncodeLatent(vaeHandle, bitmap, w, h) }
            catch (e: Throwable) { Logger.e(engineTag, "encodeLatent", e); FloatArray(w * h * 3 / 64) }
        }
    }

    override fun warmup() {
        if (!modelLoaded || !dlaLoaded) return
        super.warmup()
        timeOperation("warmup") {
            val dummy = FloatArray(4 * 64 * 64)
            val emb = FloatArray(768)
            for (i in 0 until 3) predictNoise(dummy, emb, i, 512, 512)
            Logger.i(engineTag, "✅ NPU 预热完成（3 次推理）")
        }
    }

    override fun release() {
        try {
            if (nativeHandle != 0L) {
                nativeRelease(nativeHandle)
                nativeHandle = 0L
            }
            textEncHandle = 0L; unetHandle = 0L; vaeHandle = 0L
            dlaLoaded = false
        } catch (t: Throwable) { Logger.w(engineTag, "release: ${t.message}") }
        super.release()
    }

    // ---- 私有工具 ----

    private fun detectModelType(path: String): String {
        val n = java.io.File(path).name.lowercase()
        return when {
            n.contains("text") || n.contains("clip") -> "text_encoder"
            n.contains("vae") -> "vae"
            n.contains("unet") || n.contains("diffusion") -> "unet"
            else -> "unknown"
        }
    }

    // ---- JNI 声明（对接 NeuroPilot SDK）----

    private external fun nativeCreate(): Long
    private external fun nativeGetNpuInfo(handle: Long): String
    private external fun nativeLoadDla(
        handle: Long, dlaPath: String, device: String, precision: String
    ): Long
    private external fun nativeEncodeText(handle: Long, text: String, clipSkip: Int): FloatArray
    private external fun nativePredictNoise(
        handle: Long, latent: FloatArray, textEmb: FloatArray, timestep: Int, w: Int, h: Int
    ): FloatArray
    private external fun nativeDecodeLatent(handle: Long, latent: FloatArray, w: Int, h: Int): Bitmap
    private external fun nativeEncodeLatent(handle: Long, bitmap: Bitmap, w: Int, h: Int): FloatArray
    private external fun nativeRelease(handle: Long)
}
