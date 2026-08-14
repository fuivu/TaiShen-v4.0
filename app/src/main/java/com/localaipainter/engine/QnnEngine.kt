package com.localaipainter.engine

import android.content.Context
import android.graphics.Bitmap
import com.localaipainter.util.Logger

/**
 * 高通 QNN 引擎 —— 骁龙 8 Gen 2/3 首选
 *
 * 特点：
 *   - 直接调用 Hexagon NPU，整数 INT8/INT4 推理
 *   - 速度最快（骁龙 8 Gen 3 上 512x512 约 3-5s）
 *   - 模型需离线编译为 .onnx → QNN .so
 *   - 天玑不可用（驱动不存在）
 *
 * 扩展：新增算子只需在 QNN 侧注册，本类无需修改。
 */
class QnnEngine(context: Context) : BaseEngine(context, "QNN") {

    private var nativeHandle: Long = 0L
    private var textEncHandle: Long = 0L
    private var unetHandle: Long = 0L
    private var vaeHandle: Long = 0L

    override fun backendName(): String = "QNN (骁龙 Hexagon NPU)"

    override fun supportsNpu(): Boolean = true

    override fun getSupportedPrecisions(): List<String> =
        listOf("FP32", "FP16", "INT8", "INT4")

    override fun init() {
        super.init()
        try {
            System.loadLibrary("qnn_wrapper")
            nativeHandle = nativeCreate(currentPrecision)
            if (nativeHandle == 0L) {
                Logger.e(engineTag, "QNN 创建失败")
                initialized = false
                return
            }
            Logger.i(engineTag, "✅ QNN 初始化成功 (handle=$nativeHandle)")
        } catch (e: UnsatisfiedLinkError) {
            Logger.e(engineTag, "QNN 库加载失败（非骁龙设备？）", e)
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
                    textEncHandle = nativeLoadModel(nativeHandle, path, "CPU", precision)
                    textEncHandle
                }
                "unet" -> {
                    unetHandle = nativeLoadModel(nativeHandle, path, "NPU", precision)
                    unetHandle
                }
                "vae" -> {
                    vaeHandle = nativeLoadModel(nativeHandle, path, "NPU", precision)
                    vaeHandle
                }
                else -> nativeLoadModel(nativeHandle, path, "NPU", precision)
            }
            modelLoaded = handle != 0L
            if (modelLoaded) {
                Logger.i(engineTag, "模型加载: ${java.io.File(path).name} ($modelType) precision=$precision")
            } else {
                Logger.e(engineTag, "模型加载失败: $path")
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
            catch (e: Throwable) { Logger.e(engineTag, "predictNoise step=$timestep", e); latent.copyOf() }
        }
    }

    override fun decodeLatent(latent: FloatArray, w: Int, h: Int): Bitmap {
        if (vaeHandle == 0L) return super.decodeLatent(latent, w, h)
        return timeOperation("decodeLatent") {
            try { nativeDecodeLatent(vaeHandle, latent, w, h) }
            catch (e: Throwable) { Logger.e(engineTag, "decodeLatent", e); Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888) }
        }
    }

    override fun encodeLatent(bitmap: Bitmap, w: Int, h: Int): FloatArray {
        if (vaeHandle == 0L) return super.encodeLatent(bitmap, w, h)
        return timeOperation("encodeLatent") {
            try { nativeEncodeLatent(vaeHandle, bitmap, w, h) }
            catch (e: Throwable) { Logger.e(engineTag, "encodeLatent", e); FloatArray(w * h * 3 / 64) }
        }
    }

    override fun setPowerMode(mode: PowerMode) {
        super.setPowerMode(mode)
        val m = when (mode) {
            PowerMode.LOW_POWER -> 0
            PowerMode.BALANCED -> 1
            PowerMode.HIGH_PERFORMANCE -> 2
            PowerMode.MAX_PERFORMANCE -> 3
        }
        if (nativeHandle != 0L) nativeSetPowerMode(nativeHandle, m)
    }

    override fun release() {
        try {
            if (nativeHandle != 0L) {
                nativeRelease(nativeHandle)
                nativeHandle = 0L
            }
            textEncHandle = 0L; unetHandle = 0L; vaeHandle = 0L
        } catch (t: Throwable) { Logger.w(engineTag, "release: ${t.message}") }
        super.release()
    }

    private fun detectModelType(path: String): String {
        val n = java.io.File(path).name.lowercase()
        return when {
            n.contains("text") || n.contains("clip") -> "text_encoder"
            n.contains("vae") -> "vae"
            n.contains("unet") || n.contains("diffusion") -> "unet"
            else -> "unknown"
        }
    }

    // ---- JNI ----
    private external fun nativeCreate(precision: String): Long
    private external fun nativeLoadModel(handle: Long, path: String, device: String, precision: String): Long
    private external fun nativeEncodeText(handle: Long, text: String, clipSkip: Int): FloatArray
    private external fun nativePredictNoise(handle: Long, latent: FloatArray, textEmb: FloatArray, timestep: Int, w: Int, h: Int): FloatArray
    private external fun nativeDecodeLatent(handle: Long, latent: FloatArray, w: Int, h: Int): Bitmap
    private external fun nativeEncodeLatent(handle: Long, bitmap: Bitmap, w: Int, h: Int): FloatArray
    private external fun nativeSetPowerMode(handle: Long, mode: Int)
    private external fun nativeRelease(handle: Long)
}
