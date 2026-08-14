package com.localaipainter.engine

import android.graphics.Bitmap
import com.localaipainter.engine.SchedulerType

/**
 * JNI 桥接层 — 对应 C++ 端 jni_bridge.cpp
 *
 * 所有 native 方法通过 System.loadLibrary("engine_factory") 加载
 */
object JNIBridge {

    // ============ 生命周期 ============

    external fun nativeInit(
        modelPath: String,
        vaePath: String,
        loraPaths: Array<String>,
        loraWeights: FloatArray,
        schedulerType: String,
        steps: Int,
        cfgScale: Float,
        width: Int,
        height: Int,
        seed: Long,
        useGpu: Boolean,
        threads: Int,
        quantMode: Int,    // 0=FP32, 1=FP16, 2=INT8, 3=INT4
        powerMode: Int,     // 0=MAX, 1=HIGH, 2=BALANCED, 3=SAVE
    ): Boolean

    external fun nativeGenerate(
        prompt: String,
        negativePrompt: String,
        outputPath: String,
    ): Boolean

    external fun nativeStop(): Boolean

    external fun nativeUnload(): Boolean

    // ============ 进度回调 ============

    external fun nativeSetProgressCallback(callback: ProgressCallback)

    interface ProgressCallback {
        fun onStep(step: Int, totalSteps: Int, progress: Float, previewPath: String?)
        fun onComplete(success: Boolean, outputPath: String?, message: String?)
        fun onError(errorCode: Int, message: String?)
    }

    // ============ 模型验证 ============

    external fun nativeValidateModel(modelPath: String): Int  // 0=OK, 1=INVALID, 2=UNSUPPORTED, 3=CORRUPT

    external fun nativeGetModelInfo(modelPath: String): String  // JSON

    // ============ LoRA ============

    external fun nativeLoadLoRA(path: String, weight: Float): Boolean

    external fun nativeUnloadLoRA(path: String): Boolean

    external fun nativeSetLoRAWeights(weights: FloatArray): Boolean

    // ============ ControlNet ============

    external fun nativeSetControlNet(
        type: Int,          // 0=CANNY,1=OPENPOSE,2=DEPTH,3=SCRIBBLE,4=MLSD,5=SEG
        imagePath: String,
        strength: Float,
    ): Boolean

    external fun nativeClearControlNet(): Boolean

    // ============ 超分 ============

    external fun nativeUpscale(
        inputPath: String,
        outputPath: String,
        modelPath: String,
        scale: Int,
        tileSize: Int,
    ): Boolean

    // ============ 人脸修复 ============

    external fun nativeFaceRestore(
        inputPath: String,
        outputPath: String,
        modelType: Int,     // 0=GFPGAN,1=CODEFORMER,2=RESTOREFORMER
        fidelity: Float,
    ): Boolean

    // ============ 设备检测 ============

    external fun nativeDetectDevice(): String  // JSON DeviceInfo

    external fun nativeGetBestBackend(): Int  // 0=CPU,1=Vulkan,2=OpenCL,3=NNAPI,4=Metal

    external fun nativeBenchmark(backend: Int): Float  // 返回 iterations/sec

    // ============ 缓存管理 ============

    external fun nativeClearCache(): Boolean

    external fun nativeSetGpuCacheEnabled(enabled: Boolean): Boolean

    external fun nativeGetMemoryStats(): String  // JSON

    // ============ 工具 ============

    external fun nativeGetVersion(): String

    external fun nativeGetSupportedSchedulers(): Array<String>

    external fun nativeGetSupportedModels(): Array<String>

    // ============ 常量 ============

    const val QUANT_FP32 = 0
    const val QUANT_FP16 = 1
    const val QUANT_INT8 = 2
    const val QUANT_INT4 = 3

    const val POWER_MAX = 0
    const val POWER_HIGH = 1
    const val POWER_BALANCED = 2
    const val POWER_SAVE = 3

    const val BACKEND_CPU = 0
    const val BACKEND_VULKAN = 1
    const val BACKEND_OPENCL = 2
    const val BACKEND_NNAPI = 3

    const val MODEL_TYPE_SD15 = 0
    const val MODEL_TYPE_SD21 = 1
    const val MODEL_TYPE_SDXL = 2
    const val MODEL_TYPE_LCM = 3
    const val MODEL_TYPE_PIXART = 4
    const val MODEL_TYPE_KOLORS = 5
}
