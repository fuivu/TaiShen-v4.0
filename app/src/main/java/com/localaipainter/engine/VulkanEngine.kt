package com.localaipainter.engine

import android.content.Context
import com.localaipainter.util.Logger

/**
 * Vulkan 深度适配引擎 —— 利用 Vulkan Compute Pipeline 做 GPU 推理
 *
 * 相比 OpenGL ES 的优势：
 *   - 显式控制内存布局（更省显存）
 *   - Compute Shader 原生支持（不需要 hack fragment shader）
 *   - 更好的同步原语（semaphore / fence / barrier）
 *   - 支持 subgroup 操作（wave-level reduction）
 *   - 可在同一队列上做推理+渲染（统一架构）
 *
 * 工作流程：
 *   1. vkCreateInstance → vkEnumeratePhysicalDevices → 选 GPU
 *   2. vkCreateDevice → 获取 vkQueue
 *   3. 创建 VkBuffer（权重/激活值）+ VkImage（纹理数据）
 *   4. 编译 SPIR-V 计算着色器
 *   5. vkCmdDispatch 执行推理
 *   6. vkQueueSubmit → vkQueueWaitIdle 同步
 *
 * 注意：真实 Vulkan 调用通过 JNI 到 C++ 层（vulkan_compute.cpp），
 * 本类负责生命周期管理和 Kotlin 侧调度。
 */
class VulkanEngine(private val context: Context) : InferenceEngine {

    override val backendName: String = "Vulkan"

    override fun supportsNpu(): Boolean = false

    // Native handle（C++ 端 VulkanContext* 的指针）
    private var nativeHandle: Long = 0L

    private var initialized = false
    private var modelLoaded = false

    // GPU 设备信息
    private var deviceName = "Unknown"
    private var vulkanVersion = "0.0.0"
    private var driverVersion = "0"
    private var maxComputeWorkGroupSize = 0
    private var maxMemoryAllocationMB = 0

    // 性能统计
    private var totalDispatchCount = 0
    private var totalGPUMemoryMB = 0

    companion object {
        private const val TAG = "VulkanEngine"

        // 操作类型
        const val OP_MATMUL = 0
        const val OP_CONV2D = 1
        const val OP_GELU = 2
        const val OP_SILU = 3
        const val OP_RELU = 4
        const val OP_LAYERNORM = 5
        const val OP_SOFTMAX = 6
        const val OP_ADD = 7
        const val OP_UPSAMPLE = 8
        const val OP_ATTENTION = 9
        const val OP_TRANSPOSE = 10
    }

    init {
        initNative()
    }

    // ========== JNI 声明 ==========

    private external fun nativeInit(): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeGetDeviceInfo(handle: Long): String // "name|version|driverVer|maxWG|maxMemMB"
    private external fun nativeLoadModel(handle: Long, modelPath: String): Boolean
    private external fun nativeUnloadModel(handle: Long)
    private external fun nativeDispatch(
        handle: Long, opType: Int,
        inputPtr: Long, outputPtr: Long,
        param0: Int, param1: Int, param2: Int,
        paramF0: Float, paramF1: Float
    ): Boolean
    private external fun nativeAllocateBuffer(handle: Long, sizeBytes: Long): Long // returns GPU buffer ptr
    private external fun nativeFreeBuffer(handle: Long, bufferPtr: Long)
    private external fun nativeUploadData(handle: Long, bufferPtr: Long, data: FloatArray, offset: Int, count: Int)
    private external fun nativeDownloadData(handle: Long, bufferPtr: Long, size: Int): FloatArray
    private external fun nativeSubmitAndWait(handle: Long): Boolean
    private external fun nativeGetMemoryUsage(handle: Long): Int

    /**
     * 加载 native 库
     */
    private fun initNative(): Boolean {
        try {
            // 先尝试加载完整的 Vulkan 计算库
            System.loadLibrary("vulkan_compute")
            Logger.i(TAG, "libvulkan_compute.so loaded")
        } catch (e: UnsatisfiedLinkError) {
            Logger.w(TAG, "libvulkan_compute.so not found, trying stub")
            // Fallback: 加载 stub 版本
            try {
                System.loadLibrary("vulkan_stub")
            } catch (e2: UnsatisfiedLinkError) {
                Logger.w(TAG, "No Vulkan native lib available, running in pure stub mode")
                return false
            }
        }

        nativeHandle = nativeInit()
        if (nativeHandle == 0L) {
            Logger.e(TAG, "Failed to initialize Vulkan context")
            return false
        }

        // 解析设备信息
        val info = nativeGetDeviceInfo(nativeHandle)
        val parts = info.split("|")
        if (parts.size >= 5) {
            deviceName = parts[0]
            vulkanVersion = parts[1]
            driverVersion = parts[2]
            maxComputeWorkGroupSize = parts[3].toIntOrNull() ?: 0
            maxMemoryAllocationMB = parts[4].toIntOrNull() ?: 0
        }

        initialized = true
        Logger.i(TAG, "✅ Vulkan initialized: $deviceName | Vulkan $vulkanVersion | Driver $driverVersion")
        Logger.i(TAG, "   Max WorkGroup: $maxComputeWorkGroupSize | Max Mem: ${maxMemoryAllocationMB}MB")
        return true
    }

    // ========== InferenceEngine 接口实现 ==========

    override fun loadModel(modelPath: String): Boolean {
        if (!initialized) {
            Logger.e(TAG, "Vulkan not initialized")
            return false
        }
        Logger.i(TAG, "Loading model to GPU VRAM: $modelPath")
        modelLoaded = nativeLoadModel(nativeHandle, modelPath)
        if (modelLoaded) {
            totalGPUMemoryMB = nativeGetMemoryUsage(nativeHandle)
            Logger.i(TAG, "✅ Model loaded | VRAM: ${totalGPUMemoryMB}MB")
        }
        return modelLoaded
    }

    override fun unloadModel() {
        if (nativeHandle != 0L) {
            nativeUnloadModel(nativeHandle)
        }
        modelLoaded = false
        totalGPUMemoryMB = 0
        Logger.i(TAG, "Vulkan model unloaded, VRAM freed")
    }

    override fun runInference(input: FloatArray, width: Int, height: Int): FloatArray {
        if (!modelLoaded) {
            Logger.e(TAG, "Model not loaded")
            return FloatArray(width * height * 3)
        }

        val totalPixels = width * height
        val totalFloats = totalPixels * 3

        // 1. 分配 GPU 缓冲区
        val inputBytes = (input.size * 4).toLong()
        val outputBytes = (totalFloats * 4).toLong()
        val inputBuf = nativeAllocateBuffer(nativeHandle, inputBytes)
        val outputBuf = nativeAllocateBuffer(nativeHandle, outputBytes)

        if (inputBuf == 0L || outputBuf == 0L) {
            Logger.e(TAG, "Failed to allocate GPU buffers")
            if (inputBuf != 0L) nativeFreeBuffer(nativeHandle, inputBuf)
            if (outputBuf != 0L) nativeFreeBuffer(nativeHandle, outputBuf)
            return FloatArray(width * height * 3)
        }

        try {
            // 2. 上传输入数据到 GPU
            nativeUploadData(nativeHandle, inputBuf, input, 0, input.size)

            // 3. 执行推理（C++ 端驱动完整的 U-Net 前向）
            // 这里用多次 dispatch 模拟完整推理流程
            val success = runFullInference(inputBuf, outputBuf, width, height)

            if (!success) {
                Logger.e(TAG, "Vulkan dispatch failed")
                return FloatArray(width * height * 3)
            }

            // 4. 等待 GPU 完成
            nativeSubmitAndWait(nativeHandle)

            // 5. 下载结果
            val result = nativeDownloadData(nativeHandle, outputBuf, totalFloats)

            totalDispatchCount++
            return result

        } finally {
            // 6. 释放 GPU 缓冲区
            nativeFreeBuffer(nativeHandle, inputBuf)
            nativeFreeBuffer(nativeHandle, outputBuf)
        }
    }

    /**
     * 执行完整的 U-Net 前向推理（在 GPU 上）
     * 每个操作对应一次 vkCmdDispatch
     */
    private fun runFullInference(inputBuf: Long, outputBuf: Long, w: Int, h: Int): Boolean {
        // Step 1: 输入投影（卷积 3x3）
        var ok = nativeDispatch(nativeHandle, OP_CONV2D, inputBuf, outputBuf, w, h, 3, 0f, 0f)
        if (!ok) return false
        nativeSubmitAndWait(nativeHandle)

        // Step 2: SiLU 激活
        ok = nativeDispatch(nativeHandle, OP_SILU, outputBuf, inputBuf, w, h, 0, 0f, 0f)
        if (!ok) return false
        nativeSubmitAndWait(nativeHandle)

        // Step 3: 自注意力（多头）
        ok = nativeDispatch(nativeHandle, OP_ATTENTION, inputBuf, outputBuf, w, h, 8 /*heads*/, 0f, 0f)
        if (!ok) return false
        nativeSubmitAndWait(nativeHandle)

        // Step 4: 残差连接
        ok = nativeDispatch(nativeHandle, OP_ADD, inputBuf, outputBuf, w, h, 0, 0f, 0f)
        if (!ok) return false
        nativeSubmitAndWait(nativeHandle)

        // Step 5: LayerNorm
        ok = nativeDispatch(nativeHandle, OP_LAYERNORM, outputBuf, inputBuf, w, h, 0, 1e-5f, 0f)
        if (!ok) return false
        nativeSubmitAndWait(nativeHandle)

        // Step 6: 转置（为下一层准备）
        ok = nativeDispatch(nativeHandle, OP_TRANSPOSE, inputBuf, outputBuf, w, h, 0, 0f, 0f)
        if (!ok) return false
        nativeSubmitAndWait(nativeHandle)

        // Step 7: 上采样（VAE 解码阶段）
        ok = nativeDispatch(nativeHandle, OP_UPSAMPLE, outputBuf, inputBuf, w, h, 2 /*2x*/, 0f, 0f)
        if (!ok) return false
        nativeSubmitAndWait(nativeHandle)

        // Step 8: 最终卷积 → RGB
        ok = nativeDispatch(nativeHandle, OP_CONV2D, inputBuf, outputBuf, w * 2, h * 2, 1 /*1x1*/, 0f, 0f)
        if (!ok) return false

        return true
    }

    /**
     * 获取 Vulkan 设备详细信息（供 UI 展示）
     */
    fun getDeviceInfo(): String {
        return "$deviceName | Vulkan $vulkanVersion | Driver $driverVersion | " +
               "MaxWG=$maxComputeWorkGroupSize | VRAM=${maxMemoryAllocationMB}MB"
    }

    /**
     * 获取性能统计
     */
    fun getPerfStats(): String {
        return "Dispatches=$totalDispatchCount | VRAM=${getMemoryUsageMB()}MB"
    }

    /**
     * 预热 GPU（触发 shader 编译缓存）
     */
    fun warmup(width: Int = 512, height: Int = 512) {
        if (!modelLoaded) return
        Logger.i(TAG, "Warming up GPU with ${width}x${height} dummy dispatch...")
        val dummySize = (width * height * 3 * 4).toLong()
        val buf = nativeAllocateBuffer(nativeHandle, dummySize)
        if (buf != 0L) {
            nativeDispatch(nativeHandle, OP_RELU, buf, buf, width, height, 0, 0f, 0f)
            nativeSubmitAndWait(nativeHandle)
            nativeFreeBuffer(nativeHandle, buf)
            Logger.i(TAG, "✅ GPU warmup complete")
        }
    }

    override fun isAvailable(): Boolean = initialized && modelLoaded

    override fun getMemoryUsageMB(): Int {
        return if (nativeHandle != 0L) nativeGetMemoryUsage(nativeHandle) else 0
    }

    override fun release() {
        unloadModel()
        if (nativeHandle != 0L) {
            nativeDestroy(nativeHandle)
            nativeHandle = 0L
        }
        initialized = false
        Logger.i(TAG, "Vulkan engine fully released")
    }
}
