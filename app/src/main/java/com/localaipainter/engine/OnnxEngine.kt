package com.localaipainter.engine

import android.content.Context
import android.graphics.Bitmap
import com.localaipainter.util.Logger

/**
 * ONNX Runtime 引擎 —— 通用兜底
 *
 * 特点：
 *   - 微软 ONNX Runtime，跨平台
 *   - 支持 CPU / Vulkan / CUDA（桌面）
 *   - 模型格式最通用（.onnx）
 *   - 速度中等，兼容性强
 *
 * 当 MNN/NCNN/QNN 都不可用时，ONNX 是最后一道防线。
 */
class OnnxEngine(context: Context) : BaseEngine(context, "ONNX") {

    private var nativeHandle: Long = 0L
    private var textEncHandle: Long = 0L
    private var unetHandle: Long = 0L
    private var vaeHandle: Long = 0L

    override fun backendName(): String = "ONNX Runtime ($currentDeviceType)"

    override fun supportsNpu(): Boolean = false  // ONNX 不直接支持 NPU

    override fun getSupportedPrecisions(): List<String> =
        listOf("FP32", "FP16", "INT8")

    override fun init() {
        super.init()
        try {
            System.loadLibrary("onnx_wrapper")
            nativeHandle = nativeCreate(currentPrecision)
            if (nativeHandle == 0L) {
                Logger.e(engineTag, "ONNX 创建失败")
                initialized = false
                return
            }
            Logger.i(engineTag, "✅ ONNX Runtime 初始化成功")
        } catch (e: UnsatisfiedLinkError) {
            Logger.e(engineTag, "ONNX 库加载失败", e)
            initialized = false
        }
    }

    override fun loadModel(path: String, backend: String, precision: String): Boolean {
        if (!initialized) init()
        if (nativeHandle == 0L) return false

        currentPrecision = precision
        currentDeviceType = if (backend == "GPU") "VULKAN" else "CPU"

        return timeOperation("loadModel") {
            val modelType = detectModelType(path)
            val handle = when (modelType) {
                "text_encoder" -> {
                    textEncHandle = nativeLoadModel(nativeHandle, path, "CPU", precision)
                    textEncHandle
                }
                "unet" -> {
                    unetHandle = nativeLoadModel(nativeHandle, path, currentDeviceType, precision)
                    unetHandle
                }
                "vae" -> {
                    vaeHandle = nativeLoadModel(nativeHandle, path, "GPU", precision)
                    vaeHandle
                }
                else -> nativeLoadModel(nativeHandle, path, currentDeviceType, precision)
            }
            modelLoaded = handle != 0L
            if (modelLoaded) Logger.i(engineTag, "模型加载: ${java.io.File(path).name} ($modelType)")
            else Logger.e(engineTag, "模型加载失败: $path")
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
            catch (e: Throwable) { Logger.e(engineTag, "predictNoise", e); latent.copyOf() }
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

    override fun release() {
        try {
            if (nativeHandle != 0L) { nativeRelease(nativeHandle); nativeHandle = 0L }
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

    // ---- JNI ---
    private external fun nativeCreate(precision: String): Long
    private external fun nativeLoadModel(handle: Long, path: String, device: String, precision: String): Long
    private external fun nativeEncodeText(handle: Long, text: String, clipSkip: Int): FloatArray
    private external fun nativePredictNoise(handle: Long, latent: FloatArray, textEmb: FloatArray, timestep: Int, w: Int, h: Int): FloatArray
    private external fun nativeDecodeLatent(handle: Long, latent: FloatArray, w: Int, h: Int): Bitmap
    private external fun nativeEncodeLatent(handle: Long, bitmap: Bitmap, w: Int, h: Int): FloatArray
    private external fun nativeRelease(handle: Long)
}
